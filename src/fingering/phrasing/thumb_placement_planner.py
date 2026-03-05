"""
Thumb Placement Planner — Phase 3A improvement.

Addresses the largest error category: THUMB_MISS (47.8% of wrong predictions).

Root cause: The Viterbi DP uses a local thumb-under REWARD at note transitions,
but has no global plan for *where* the thumb should land across the phrase.
Pianist thinking: "Where does my thumb land so my hand stays comfortable for
the REST of the phrase?" — this is global planning, not local scoring.

Algorithm
---------
For a stepwise run (half/whole steps in the same direction), detect the
natural thumb landing points — i.e., every time a thumb crossing would happen
in standard scale technique — and inject those as forced constraints *before*
the DP runs.

Rules:
  RH ascending:   thumb crosses under after finger 3 (→ lands 3 notes later)
                   → every 3rd note in stepwise run gets finger=1
  RH descending:  finger crosses over thumb → same cadence mirrored
  LH descending:  thumb crosses under after finger 3
  LH ascending:   finger crosses over

Skips:
  - Notes that already have a PatternLibrary or ChordHeuristic constraint
  - Notes landing on black keys (thumb on black = high penalty, avoid)
  - Short stepwise runs (< 3 notes consecutive) — no crossing needed
"""

from __future__ import annotations
from typing import List, Dict, Optional
from fingering.models.note_event import NoteEvent


_STEPWISE_SEMITONES = {1, 2}    # Half or whole step
_THUMB = 1
_MIN_RUN_LENGTH = 4             # Minimum stepwise notes to trigger planning


def _is_stepwise(a: NoteEvent, b: NoteEvent) -> bool:
    """True if interval between a and b is a half or whole step."""
    return abs(b.pitch - a.pitch) in _STEPWISE_SEMITONES


def _detect_stepwise_runs(notes: List[NoteEvent]) -> List[tuple[int, int, bool]]:
    """
    Returns list of (start_idx, end_idx_exclusive, is_ascending) for stepwise runs.
    A run must be at least _MIN_RUN_LENGTH notes in the same direction.
    """
    if len(notes) < 2:
        return []

    runs = []
    i = 0
    n = len(notes)

    while i < n - 1:
        if not _is_stepwise(notes[i], notes[i + 1]):
            i += 1
            continue

        ascending = notes[i + 1].pitch > notes[i].pitch
        j = i + 1
        while j < n - 1 and _is_stepwise(notes[j], notes[j + 1]):
            next_asc = notes[j + 1].pitch > notes[j].pitch
            if next_asc != ascending:
                break
            j += 1

        run_len = j - i + 1
        if run_len >= _MIN_RUN_LENGTH:
            runs.append((i, j + 1, ascending))
        i = j  # continue from end of run

    return runs


class ThumbPlacementPlanner:
    """
    Pre-DP pass: injects thumb landing positions as forced constraints.

    Usage:
        planner = ThumbPlacementPlanner()
        forced = planner.plan(notes, hand='right', existing_forced=forced)
        # Pass updated `forced` dict into PhraseScopedDP.solve()

    Priority: PatternLibrary constraints beat planner constraints.
    The planner never overrides an already-constrained position.
    """

    def plan(
        self,
        notes: List[NoteEvent],
        hand: str,
        existing_forced: Dict[int, int],
    ) -> Dict[int, int]:
        """
        Compute thumb landing positions for stepwise runs and inject them.

        Returns an updated forced constraints dict.
        """
        forced = dict(existing_forced)
        runs = _detect_stepwise_runs(notes)

        for start, end, ascending in runs:
            thumb_positions = self._thumb_landing_positions(
                notes, start, end, ascending, hand
            )
            for idx in thumb_positions:
                # Never override existing constraints (PatternLibrary/ChordHeuristic wins)
                if idx not in forced:
                    note = notes[idx]
                    # Avoid placing thumb on a black key
                    if not note.is_black:
                        forced[idx] = _THUMB

        return forced

    def _thumb_landing_positions(
        self,
        notes: List[NoteEvent],
        start: int,
        end: int,
        ascending: bool,
        hand: str,
    ) -> List[int]:
        """
        Compute note indices where thumb should land within a stepwise run.

        Standard piano technique:
          - RH ascending + LH descending: thumb crosses UNDER.
            Pattern: 1-2-3-[thumb]-2-3-4-[thumb]-...
            → thumb lands every 3-4 notes depending on key (3 after white, 4 near black clusters)
          - RH descending + LH ascending: fingers cross OVER thumb.
            Pattern: 5-4-3-2-[thumb]-3-2-[thumb]-...
            → thumb lands at mirror positions

        We use a conservative cadence of 3 notes per thumb crossing
        (the minimum; actual scale patterns vary from 3 to 5).
        The PatternLibrary handles exact scale contexts — this planner
        handles generic stepwise runs where PatternLibrary didn't match.
        """
        run_notes = notes[start:end]
        length = len(run_notes)
        positions = []

        # Determine if this hand/direction uses thumb-under or finger-over
        # Both result in thumb landing points, just from different directions.
        # RH ascending:  thumb-under → thumb lands after finger 3
        # RH descending: finger-over → thumb lands as "pivot" for the next group
        # LH descending: thumb-under (mirror of RH ascending)
        # LH ascending:  finger-over

        # All four cases: thumb appears roughly every 3 notes in a run.
        # Start from position 2 (index within run, 0-based) = after two fingers.
        # This simulates the "1-2-3-thumb" or "thumb-2-3-thumb" cadence.

        # For ascending RH: first thumb at offset 3 within run (0-indexed local)
        # For descending RH: first thumb at offset 2 (pinky-side start → thumb end)
        if (hand == 'right' and ascending) or (hand == 'left' and not ascending):
            # Thumb-under: 1-2-3-[1]-2-3-4-[1]  → thumb at local offset 3, then +3 or +4
            offset = 3
        else:
            # Finger-over (descending RH): 5-4-3-2-[1]-4-3-[1] → thumb at offset 4
            offset = 4

        idx = offset
        while idx < length:
            global_idx = start + idx
            # Skip if this position's note is a black key
            if not run_notes[idx].is_black:
                positions.append(global_idx)
            # Advance: after thumb, next group is 3 fingers, then thumb again
            idx += 3

        return positions


def apply_thumb_constraints(
    notes: List[NoteEvent],
    hand: str,
    existing_forced: Dict[int, int],
) -> Dict[int, int]:
    """
    Convenience wrapper for use inside PhraseScopedDP.solve().

    Example:
        forced = apply_thumb_constraints(notes, hand, forced)
    """
    return ThumbPlacementPlanner().plan(notes, hand, existing_forced)
