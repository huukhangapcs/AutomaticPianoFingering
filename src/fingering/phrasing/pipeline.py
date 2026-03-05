"""
Phrase-Aware Fingering Pipeline (v2) — Pianist-Inspired Architecture.

Orchestrates 4 recognition layers + 4 fingering layers:

  RECOGNITION (top-down + bottom-up):
    Layer 0: ScoreProfile    — texture, key, markings (global pre-read)
    Layer 1: MotifEngine     — recurring motifs → A-B-A form (top-down)
    Layer 2: HarmonicSkeleton— chord root + cadence map (bottom-up context)
    Layer 3: PhraseSelector  — merge forced + confirmed boundaries

  FINGERING:
    Layer A: PhraseBoundaryDetector  — bottom-up boundary signals
    Layer B: PhraseIntentAnalyzer    — musical intent + tension arc
    Layer C: PhraseScopedDP          — Viterbi DP with pattern constraints
    Layer D: CrossPhraseStitch       — smooth cross-phrase transitions
"""
from __future__ import annotations
from typing import List, Optional

from fingering.models.note_event import NoteEvent
from fingering.phrasing.phrase import Phrase
from fingering.phrasing.boundary_detector import PhraseBoundaryDetector
from fingering.phrasing.intent_analyzer import PhraseIntentAnalyzer
from fingering.phrasing.phrase_dp import PhraseScopedDP
from fingering.phrasing.cross_phrase import CrossPhraseStitch
from fingering.phrasing.score_profile import ScoreProfile, build_score_profile
from fingering.phrasing.motif_engine import MotifEngine, Section
from fingering.phrasing.harmonic_skeleton import HarmonicSkeleton, build_harmonic_skeleton
from fingering.phrasing.phrase_selector import PhraseSelector, PhraseSelectorConfig


class PhraseAwareFingering:
    """
    High-level pipeline: NoteEvents → fingering list (int 1–5 per note).

    Two operating modes:
      - grand_staff=False (default): single-hand, uses only bottom-up signals
      - grand_staff=True: pass both_hands=(rh, lh) to run(), enables all 4 layers

    Usage (single hand):
        paf = PhraseAwareFingering()
        fingering = paf.run(notes)

    Usage (grand staff, full top-down):
        paf = PhraseAwareFingering()
        rh_fingering = paf.run(rh_notes, companion_notes=lh_notes)
        lh_fingering = paf.run(lh_notes, companion_notes=rh_notes)
    """

    def __init__(
        self,
        boundary_threshold: float = 0.40,
        time_sig_numerator: int = 4,
        min_phrase_notes: int = 3,
        max_phrase_measures: int = 16,
        use_motif_engine: bool = True,
    ):
        self.boundary_threshold = boundary_threshold
        self.use_motif_engine = use_motif_engine

        self.detector  = PhraseBoundaryDetector(
            boundary_threshold=boundary_threshold,
            time_sig_numerator=time_sig_numerator,
            min_phrase_notes=min_phrase_notes,
            max_phrase_measures=max_phrase_measures,
        )
        self.analyzer  = PhraseIntentAnalyzer()
        self.dp_solver = PhraseScopedDP()
        self.stitcher  = CrossPhraseStitch()
        self.motif_engine = MotifEngine()
        self.phrase_selector = PhraseSelector(
            PhraseSelectorConfig(max_phrase_measures=max_phrase_measures)
        )

    # ──────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────

    def run(
        self,
        notes: List[NoteEvent],
        companion_notes: Optional[List[NoteEvent]] = None,
    ) -> List[int]:
        """
        Full pianist-inspired fingering pipeline.

        Args:
            notes: Primary hand notes to finger.
            companion_notes: The other hand's notes (optional).
                When provided, enables harmonic skeleton + motif engine.

        Returns:
            List[int] — finger 1–5 per note, same order as input.
        """
        if not notes:
            return []

        # ---- Recognition Layers (top-down + bottom-up) ----
        phrases = self._detect_phrases(notes, companion_notes)

        if not phrases:
            return [1] * len(notes)

        # ---- Fingering Layers (B → C → D) ----
        return self._assign_fingering(phrases)

    def run_and_annotate(self, notes: List[NoteEvent], companion_notes=None) -> List[NoteEvent]:
        """Run pipeline and write finger assignments back to NoteEvents."""
        fingering = self.run(notes, companion_notes=companion_notes)
        for note, finger in zip(notes, fingering):
            note.finger = finger
        return notes

    def get_phrases(
        self,
        notes: List[NoteEvent],
        companion_notes: Optional[List[NoteEvent]] = None,
    ) -> List[Phrase]:
        """Expose detected + analyzed phrases for debugging."""
        phrases = self._detect_phrases(notes, companion_notes)
        return self.analyzer.analyze_all(phrases)

    # ──────────────────────────────────────────────────────────────
    # Recognition: Layer 0-3
    # ──────────────────────────────────────────────────────────────

    def _detect_phrases(
        self,
        notes: List[NoteEvent],
        companion: Optional[List[NoteEvent]],
    ) -> List[Phrase]:
        """
        Run the 4-layer recognition pipeline:
           0: ScoreProfile  → global texture
           1: MotifEngine   → sections (top-down forced boundaries)
           2: HarmonicSkeleton → cadences
           3: PhraseSelector → merge into final phrases
        """
        rh = notes
        lh = companion or []

        # Layer 0: build global score profile
        score_profile = build_score_profile(rh, lh)

        # Layer 1: motif-based form detection
        sections: List[Section] = []
        if self.use_motif_engine and len(notes) >= 8:
            sections = self.motif_engine.infer_sections(notes)

        # Layer 2: harmonic skeleton
        harmonic: HarmonicSkeleton = build_harmonic_skeleton(rh, lh)

        # Layer A: bottom-up boundary signals (existing detector)
        bottom_up_signals = self.detector._compute_signals(notes)
        self.detector._apply_phrase_length_prior(bottom_up_signals, notes)

        # Layer 3: PhraseSelector merge
        if sections:
            # Top-down driven: use PhraseSelector to merge
            boundary_indices = self.phrase_selector.select(
                notes=notes,
                bottom_up_signals=bottom_up_signals,
                sections=sections,
                harmonic=harmonic,
                score_profile=score_profile,
            )
            phrases = self.detector._build_phrases_from_indices(
                notes, boundary_indices
            )
        else:
            # No sections detected: fall back to pure bottom-up
            phrases = self.detector.detect(notes)

        return phrases

    # ──────────────────────────────────────────────────────────────
    # Fingering: Layers B → C → D
    # ──────────────────────────────────────────────────────────────

    def _assign_fingering(self, phrases: List[Phrase]) -> List[int]:
        """Layers B (intent) + C (DP) + D (stitch) → flat finger list."""
        analyzed = self.analyzer.analyze_all(phrases)

        all_fingering: List[int] = []
        prev_phrase: Optional[Phrase] = None
        prev_fingering: Optional[List[int]] = None

        for phrase in analyzed:
            stitch = None
            if prev_phrase is not None and prev_fingering is not None:
                stitch = self.stitcher.compute_constraint(
                    prev_phrase, phrase, prev_fingering
                )
            fingering = self.dp_solver.solve(phrase, stitch_constraint=stitch)
            all_fingering.extend(fingering)
            prev_phrase = phrase
            prev_fingering = fingering

        return all_fingering
