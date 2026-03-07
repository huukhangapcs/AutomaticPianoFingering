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
from fingering.phrasing.period_detector import PeriodDetector
from fingering.phrasing.fingering_auditor import FingeredAuditor
from fingering.phrasing.comfort_checker import is_too_hard, phrase_difficulty


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
        max_phrase_measures: int = 12,   # Fix 5: synced with PhraseSelector cap
        use_motif_engine: bool = True,
        tonic_pc: Optional[int] = None,  # Fix 1: from MusicXML key sig
    ):
        self.boundary_threshold = boundary_threshold
        self.use_motif_engine = use_motif_engine
        self.tonic_pc = tonic_pc  # Fix 1

        self.detector  = PhraseBoundaryDetector(
            boundary_threshold=boundary_threshold,
            time_sig_numerator=time_sig_numerator,
            min_phrase_notes=min_phrase_notes,
            max_phrase_measures=max_phrase_measures,    # Fix 5
        )
        self.analyzer  = PhraseIntentAnalyzer()
        self.dp_solver = PhraseScopedDP()
        self.stitcher  = CrossPhraseStitch()
        self.motif_engine = MotifEngine()
        self.phrase_selector = PhraseSelector(
            PhraseSelectorConfig(max_phrase_measures=max_phrase_measures)  # Fix 5
        )
        self.period_detector = PeriodDetector()  # Fix 4
        self.auditor = FingeredAuditor()           # Post-DP validation

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
        is_left_hand = len(notes) > 0 and notes[0].hand == 'left'

        # Layer 0: build global score profile (Fix 1: with key sig tonic if available)
        score_profile = build_score_profile(rh, lh, tonic_pc_override=self.tonic_pc)

        if is_left_hand:
            # ----- Left Hand: Pattern-Aware Segmentation (Phase 3C) -----
            # Pianist thinks of LH accompaniment in harmonic units (2-4 measures),
            # not individual measures. Group by detected patterns first.
            phrases = self._segment_lh_pattern_aware(notes)
            return phrases

        # ----- Right Hand: Full Phrase Segmentation -----
        # Layer 1: motif-based form detection (Fix 2: phrase-level motif lengths)
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
            phrases = self.detector.detect(notes)

        # Fix 4: Run PeriodDetector — annotate phrases with antecedent/consequent labels
        if score_profile.tonic_pc is not None:
            self.period_detector.tonic_pc = score_profile.tonic_pc
        periods, _ = self.period_detector.detect(phrases, harmonic)
        # Annotate phrases with period membership
        for period in periods:
            period.antecedent.period_role = 'antecedent'
            period.consequent.period_role = 'consequent'

        return phrases

    # ──────────────────────────────────────────────────────────────
    # Fingering: Layers B → C → D
    # ──────────────────────────────────────────────────────────────

    def _assign_fingering(self, phrases: List[Phrase]) -> List[int]:
        """Layers B (intent) + C (DP) + D (stitch) + Comfort-check + Audit."""
        analyzed = self.analyzer.analyze_all(phrases)

        all_fingering: List[int] = []
        prev_phrase: Optional[Phrase] = None
        prev_fingering: Optional[List[int]] = None

        for phrase in analyzed:
            stitch = None
            if prev_phrase is not None and prev_fingering is not None:
                # Calculate actual rest gap between phrases (in seconds)
                # beats_per_sec = bpm / 60; default 120 BPM if not available
                bpm = getattr(self, 'bpm', 120.0)
                beats_per_sec = bpm / 60.0
                if prev_phrase.notes and phrase.notes:
                    gap_beats = phrase.notes[0].onset - prev_phrase.notes[-1].offset
                    rest_sec = max(0.0, gap_beats / beats_per_sec)
                else:
                    rest_sec = 0.0
                stitch = self.stitcher.compute_constraint(
                    prev_phrase, phrase, prev_fingering, rest_sec=rest_sec
                )
            fingering = self.dp_solver.solve(phrase, stitch_constraint=stitch)

            # ── Comfort Check + Strict Re-solve ──────────────────────────────
            # After DP, evaluate ergonomic difficulty of the phrase.
            # If the result is too hard for a human pianist, re-solve
            # with STRICT mode (stronger span + weak-pair penalties).
            # Take the strict result only if it is not harder than original.
            if is_too_hard(phrase.notes, fingering):
                strict_fingering = self.dp_solver.solve(
                    phrase, stitch_constraint=stitch, strict=True
                )
                if phrase_difficulty(phrase.notes, strict_fingering) < \
                   phrase_difficulty(phrase.notes, fingering):
                    fingering = strict_fingering
            # ─────────────────────────────────────────────────────────────

            # ── Audit + Repair ──────────────────────────────────────────
            hand = phrase.notes[0].hand if phrase.notes else 'right'
            report = self.auditor.audit(phrase.notes, fingering, hand=hand)
            if not report.is_clean:
                fingering = self.auditor.repair(phrase.notes, fingering, hand=hand)
            # ───────────────────────────────────────────

            all_fingering.extend(fingering)
            prev_phrase = phrase
            prev_fingering = fingering

        return all_fingering

    # ──────────────────────────────────────────────────────────────
    # Phase 3C: LH Pattern-Aware Segmentation
    # ──────────────────────────────────────────────────────────────

    def _segment_lh_pattern_aware(self, notes: List[NoteEvent]) -> List[Phrase]:
        """
        Segment the Left Hand using detected accompaniment patterns.

        Pianist thinking: LH accompaniment is thought of in harmonic units —
        e.g., one Alberti cycle per measure, one waltz unit per beat group.
        The hand stays in position for the whole pattern unit, so segmenting
        at pattern boundaries (not arbitrary measure lines) is more natural.

        Strategy:
          1. Detect Alberti, waltz, arpeggio, and partial-scale patterns via PatternLibrary.
          2. Group detected pattern units into phrases of 2–4 measures.
          3. If no pattern is detected (e.g., single bass notes or chords), fall back to
             segmenting every 2 measures (more natural than every 1 measure).

        Returns:
            List[Phrase] — LH phrases sized by musical unit, not measure lines.
        """
        from fingering.phrasing.pattern_library import PatternLibrary
        from fingering.phrasing.boundary_detector import detect_arc_type

        if not notes:
            return []

        n = len(notes)
        lib = PatternLibrary()

        # ── Step 1: Detect all LH patterns ──────────────────────────────
        matches = lib.find_all(notes, hand='left')
        lh_patterns = {
            m.pattern for m in matches
            if m.pattern in ('alberti_bass', 'waltz_bass', 'arpeggio_asc', 'arpeggio_desc')
        }

        has_accompaniment_pattern = bool(lh_patterns)

        # ── Step 2: Determine phrase grouping size ───────────────────────
        # Group measures: Alberti uses 4-note units (1 per beat × 4 = 1 measure typically)
        # → natural phrase = 2 measures of Alberti
        # Waltz uses 3-note units (1 per beat × 3 = 1 measure in 3/4)
        # → natural phrase = 2 measures of waltz
        # No pattern → 2-measure grouping (double the original 1-measure cut)
        if 'alberti_bass' in lh_patterns or 'waltz_bass' in lh_patterns:
            phrase_measure_size = 2   # One harmonic unit = 2 measures
        else:
            phrase_measure_size = 2   # Default: 2-measure groups (better than 1)

        # ── Step 3: Build measures list ──────────────────────────────────
        # Collect unique measure numbers and their note ranges
        measures: dict[int, list] = {}
        for i, note in enumerate(notes):
            m = note.measure
            if m not in measures:
                measures[m] = []
            measures[m].append(i)

        sorted_measures = sorted(measures.keys())

        # ── Step 4: Group measures into phrases ──────────────────────────
        # Collect boundaries: last note index of each phrase group
        boundary_indices: list[int] = []
        for group_start in range(0, len(sorted_measures), phrase_measure_size):
            group_end = min(group_start + phrase_measure_size, len(sorted_measures))
            last_measure = sorted_measures[group_end - 1]
            last_note_idx = max(measures[last_measure])
            boundary_indices.append(last_note_idx)

        # Ensure we always end at the last note
        if not boundary_indices or boundary_indices[-1] != n - 1:
            boundary_indices.append(n - 1)

        return self.detector._build_phrases_from_indices(notes, boundary_indices)

