"""
Layer 3: Phrase Selector — Top-Down + Bottom-Up Merge

Merges forced section boundaries (from MotifEngine) with bottom-up
signal candidates (from PhraseBoundaryDetector) and cadence evidence
(from HarmonicSkeleton) to produce the final phrase segmentation.

Decision priority:
  1. Section changes from MotifEngine (FORCE — highest priority)
  2. Bottom-up candidates aligned with cadences (STRONG CONFIRM)
  3. Strong bottom-up candidates alone (WEAK CONFIRM)
  4. Fallback: forced measure segmentation (if phrase still too long)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Set
from fingering.models.note_event import NoteEvent
from fingering.phrasing.phrase import Phrase, PhraseBoundarySignal
from fingering.phrasing.motif_engine import Section
from fingering.phrasing.harmonic_skeleton import HarmonicSkeleton, Cadence
from fingering.phrasing.score_profile import ScoreProfile


@dataclass
class PhraseSelectorConfig:
    """Tunable parameters for phrase selection."""
    max_phrase_measures: int = 16   # Hard cap — never create phrase > this
    min_phrase_measures: int = 2    # Never split below this
    cadence_alignment_tolerance: int = 1   # measures of tolerance for cadence match
    confirm_threshold: float = 0.45        # Min bottom-up score to confirm without cadence
    weak_threshold: float = 0.35           # Min score with cadence support


class PhraseSelector:
    """
    Merges top-down and bottom-up evidence into final phrase boundaries.

    The key insight:
    - Top-down (MotifEngine sections) tells us WHERE phrase groups ARE
    - Bottom-up (boundary signals) tells us WHERE natural breaks occur
    - Cadences CONFIRM bottom-up candidates within section structure
    """

    def __init__(self, config: Optional[PhraseSelectorConfig] = None):
        self.config = config or PhraseSelectorConfig()

    def select(
        self,
        notes: List[NoteEvent],
        bottom_up_signals: List[PhraseBoundarySignal],
        sections: List[Section],
        harmonic: HarmonicSkeleton,
        score_profile: ScoreProfile,
    ) -> List[int]:
        """
        Returns a list of note indices where phrase boundaries should be placed.

        The boundary at index i means:
          - notes[:i+1] → phrase 1
          - notes[i+1:] → phrase 2 (and so on)
        """
        if not notes:
            return []

        cfg = self.config

        # --- Step 1: Convert section changes to FORCED boundaries ---
        forced_measures: Set[int] = set()
        for s in sections:
            forced_measures.add(s.start_measure)

        # --- Step 2: Evaluate bottom-up candidates ---
        # For each candidate, compute an enhanced score using cadence evidence
        confirmed_indices: Set[int] = set()
        weak_indices: Set[int] = set()

        for sig in bottom_up_signals:
            idx = sig.position
            if idx >= len(notes):
                continue
            note = notes[idx]
            measure = note.measure

            # Skip if within min_phrase_measures of a forced boundary
            # (avoids tiny splinters between forced + confirmed boundaries)
            in_forced_zone = any(
                abs(measure - fm) < cfg.min_phrase_measures
                for fm in forced_measures
            )

            base_score = sig.boundary_score()

            # Check for nearby cadence to boost confidence
            cadence = harmonic.cadence_at(measure, cfg.cadence_alignment_tolerance)
            cadence_boost = cadence.strength * 0.20 if cadence else 0.0
            enhanced_score = min(1.0, base_score + cadence_boost)

            if in_forced_zone:
                continue   # Don't clutter near section boundaries

            if enhanced_score >= cfg.confirm_threshold:
                confirmed_indices.add(idx)
            elif cadence and enhanced_score >= cfg.weak_threshold:
                weak_indices.add(idx)

        # --- Step 3: Convert forced measures to note indices ---
        forced_indices: Set[int] = set()
        for i, note in enumerate(notes):
            if note.measure in forced_measures and (
                i == 0 or notes[i - 1].measure not in forced_measures
            ):
                # First note of the forced measure = boundary predecessor
                if i > 0:
                    forced_indices.add(i - 1)

        # --- Step 4: Merge all boundary sets ---
        all_boundaries = sorted(forced_indices | confirmed_indices | weak_indices)

        # --- Step 5: Apply minimum phrase length filter ---
        filtered: List[int] = []
        last_boundary = -1
        for idx in all_boundaries:
            note = notes[idx]
            if last_boundary == -1:
                # First boundary: check it's not in the first min_phrase_measures
                if note.measure - notes[0].measure >= cfg.min_phrase_measures - 1:
                    filtered.append(idx)
                    last_boundary = idx
            else:
                prev_note = notes[last_boundary]
                gap_measures = note.measure - prev_note.measure
                if gap_measures >= cfg.min_phrase_measures:
                    filtered.append(idx)
                    last_boundary = idx

        # --- Step 6: Force-split any remaining over-long phrases ---
        final: List[int] = []
        prev_split = 0
        for boundary in filtered + [len(notes) - 1]:
            phrase_slice = notes[prev_split: boundary + 1]
            if not phrase_slice:
                continue
            n_measures = phrase_slice[-1].measure - phrase_slice[0].measure + 1
            if n_measures > cfg.max_phrase_measures:
                # Find best split point in middle (prefer cadence)
                mid_m = phrase_slice[0].measure + cfg.max_phrase_measures // 2
                best_split = _find_split_near(notes, prev_split, boundary, mid_m, harmonic)
                if best_split is not None and best_split not in final:
                    final.append(best_split)

            if boundary < len(notes) - 1:
                final.append(boundary)
            prev_split = boundary + 1

        return sorted(set(final))


# ──────────────────────────────────────────────────────────────
# Helper
# ──────────────────────────────────────────────────────────────

def _find_split_near(
    notes: List[NoteEvent],
    start_idx: int,
    end_idx: int,
    target_measure: int,
    harmonic: HarmonicSkeleton,
    search_window: int = 4,
) -> Optional[int]:
    """
    Find the best split point near target_measure in notes[start_idx:end_idx].
    Prefers notes where a cadence occurs nearby.
    """
    best_idx = None
    best_score = -1.0

    for i in range(start_idx, min(end_idx + 1, len(notes))):
        note = notes[i]
        dist = abs(note.measure - target_measure)
        if dist > search_window:
            continue

        score = max(0.0, 1.0 - dist / search_window)
        cadence = harmonic.cadence_at(note.measure, 1)
        if cadence:
            score += cadence.strength * 0.5

        if score > best_score:
            best_score = score
            best_idx = i

    return best_idx
