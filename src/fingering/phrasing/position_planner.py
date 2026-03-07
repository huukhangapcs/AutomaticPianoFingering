"""
Position Planner — Phase 2.8

Pre-pass that identifies stable hand positions before the Viterbi DP runs.

Core insight (from review.txt):
  Pianist tư duy theo 2 tầng:
    Layer 1: Hand Position Planning   ← this module
    Layer 2: Fingering inside position ← PhraseScopedDP

The DP currently optimises pairwise (note_i, note_{i+1}) transitions.
It has no memory that "I've been in E5-position for 3 notes, so I'm
committed to it." A wrong anchor at note 0 cascades through the phrase.

This planner:
  1. Scans sliding windows of 4–8 notes
  2. Finds the thumb_mm that covers all notes in the window under some finger
  3. Returns per-note anchor suggestions (anchor_mm[i])
  4. phrase_dp.py adds POSITION_ANCHOR_REWARD when the chosen finger
     matches the suggested anchor — making stable positions "sticky" across
     multiple notes rather than just pairwise.

Example — m10 (G5, E5, A5, G5):
  Planner finds anchor=1034mm (E5 thumb position)
  Covers: G5/f3, E5/f1, A5/f4, G5/f3
  DP receives -2.0 reward for each note that matches → cumulative -8.0
  vs shifting anchor (F5 thumb): each pair costs +shift_cost
  → DP strongly prefers E5-anchor chain.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional

from fingering.models.note_event import NoteEvent
from fingering.core.keyboard import physical_key_position_mm, _WHITE_KEY_WIDTH_MM as _WK
from fingering.phrasing.hand_position import _FINGER_OFFSET_MM, _FINGER_TOLERANCE_MM

# ──────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────

# Tolerance: how close a note must be to an expected finger position
# to be considered "covered" by that anchor.
# Now uses per-finger _FINGER_TOLERANCE_MM from hand_position.py.
# Kept here as fallback default only.
_COVERAGE_TOLERANCE_MM = 15.0

# Window sizes to try (largest first → prefer stable long positions)
_WINDOW_SIZES = [8, 6, 5, 4]

# Minimum coverage fraction to accept an anchor as "good"
_MIN_COVERAGE_FRACTION = 0.75

# Penalty for using high fingers (prefer low fingers when coverage equal)
_FINGER_PENALTY = [0.0, 0.0, 0.1, 0.2, 0.5, 1.0]  # index by finger (1-5)


# ──────────────────────────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────────────────────────

@dataclass
class PositionAnchor:
    """A stable hand position covering a contiguous note segment."""
    start_idx: int        # first note index (inclusive)
    end_idx: int          # last note index (inclusive)
    thumb_mm: float       # physical position of thumb (f1)
    coverage: float       # fraction of notes covered (0–1)
    finger_map: dict      # {note_idx: finger} for covered notes


# ──────────────────────────────────────────────────────────────────
# PositionPlanner
# ──────────────────────────────────────────────────────────────────

class PositionPlanner:
    """
    Pre-pass: infer stable hand positions for a phrase before DP runs.

    Usage:
        planner = PositionPlanner()
        anchor_mm = planner.plan(notes, hand='right')
        # anchor_mm[i] = suggested thumb_mm for note i, or None
    """

    def plan(
        self,
        notes: List[NoteEvent],
        hand: str = 'right',
    ) -> List[Optional[float]]:
        """
        Return per-note anchor suggestions.

        anchor_mm[i]:
          float  → the thumb_mm of the recommended position for note i
          None   → no stable anchor found, DP is free

        The anchors are non-overlapping: once a segment is assigned an
        anchor, those notes are not re-considered.
        """
        n = len(notes)
        anchor_mm: List[Optional[float]] = [None] * n
        covered = [False] * n

        # Greedy: process windows from left to right, largest first
        i = 0
        while i < n:
            if covered[i]:
                i += 1
                continue

            best: Optional[PositionAnchor] = None

            for w in _WINDOW_SIZES:
                end = min(i + w, n)
                window = notes[i:end]
                anchor = self._best_anchor(window, i, hand)
                if anchor is not None:
                    if best is None or anchor.coverage > best.coverage:
                        best = anchor

            if best is not None and best.coverage >= _MIN_COVERAGE_FRACTION:
                # Apply this anchor to the window
                for idx in range(best.start_idx, best.end_idx + 1):
                    anchor_mm[idx] = best.thumb_mm
                    covered[idx] = True
                i = best.end_idx + 1
            else:
                i += 1

        return anchor_mm

    # ──────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────

    def _best_anchor(
        self,
        window: List[NoteEvent],
        start_idx: int,
        hand: str,
    ) -> Optional[PositionAnchor]:
        """
        Find the best thumb_mm that covers the most notes in the window.

        Candidate thumb positions: for each note and each finger (1–5),
        compute the implied thumb_mm.  Then score each candidate by how
        many notes it covers.
        """
        if not window:
            return None

        candidates: set[float] = set()
        for note in window:
            note_mm = physical_key_position_mm(note.pitch)
            for f in range(1, 6):
                if hand == 'right':
                    t_mm = note_mm - (f - 1) * _WK
                else:
                    t_mm = note_mm + (f - 1) * _WK
                candidates.add(round(t_mm, 1))

        best_anchor: Optional[PositionAnchor] = None
        best_score = -1.0

        for thumb_mm in candidates:
            finger_map, coverage = self._score_anchor(
                thumb_mm, window, start_idx, hand
            )
            if not finger_map:
                continue
            # Penalty for high average finger number
            avg_finger_penalty = (
                sum(_FINGER_PENALTY[f] for f in finger_map.values())
                / max(1, len(finger_map))
            )
            # Temporal ordering score: reward monotonic/stepwise finger sequences
            temporal = self._temporal_order_score(finger_map)
            # Combined score: coverage weighted most, temporal adds quality signal
            score = coverage * 0.70 + temporal * 0.20 - avg_finger_penalty * 0.10

            if score > best_score:
                best_score = score
                end_idx = start_idx + len(window) - 1
                best_anchor = PositionAnchor(
                    start_idx=start_idx,
                    end_idx=end_idx,
                    thumb_mm=thumb_mm,
                    coverage=coverage,
                    finger_map=finger_map,
                )

        return best_anchor

    def _temporal_order_score(self, finger_map: dict) -> float:
        """
        Score how naturally the finger sequence flows over time.

        Rewards:
          - Monotonic runs (1→2→3→4→5 or reverse): very natural
          - Stepwise changes (|Δf| = 1): natural adjacent finger motion
        Penalises:
          - Large irregular jumps (e.g. 1→5→2→4): awkward temporal pattern

        Returns a score in [0, 1].
        """
        if not finger_map or len(finger_map) < 2:
            return 1.0
        fingers = [finger_map[i] for i in sorted(finger_map)]
        natural = sum(
            1 for a, b in zip(fingers, fingers[1:])
            if abs(b - a) <= 2   # stepwise or skip-one = still natural
        )
        return natural / (len(fingers) - 1)

    def _score_anchor(
        self,
        thumb_mm: float,
        window: List[NoteEvent],
        start_idx: int,
        hand: str,
    ) -> tuple[dict, float]:
        """
        Score an anchor position against the window.

        Uses anatomical finger offsets (_FINGER_OFFSET_MM) and per-finger
        tolerances (_FINGER_TOLERANCE_MM) for more accurate coverage scoring.

        Returns:
            finger_map: {global_note_idx: best_finger} for covered notes
            coverage: fraction of notes covered (0.0–1.0)
        """
        finger_map: dict[int, int] = {}

        for local_i, note in enumerate(window):
            note_mm = physical_key_position_mm(note.pitch)
            global_i = start_idx + local_i

            best_f = None
            best_diff = float('inf')
            for f in range(1, 6):
                offset = _FINGER_OFFSET_MM[f]
                if hand == 'right':
                    expected_mm = thumb_mm + offset
                else:
                    expected_mm = thumb_mm - offset
                diff = abs(note_mm - expected_mm)
                tolerance = _FINGER_TOLERANCE_MM[f]   # per-finger tolerance
                if diff <= tolerance and diff < best_diff:
                    best_diff = diff
                    best_f = f

            if best_f is not None:
                finger_map[global_i] = best_f

        coverage = len(finger_map) / len(window) if window else 0.0
        return finger_map, coverage
