"""
SimpleFingering — Physics-First Piano Fingering via Viterbi DP.

Model:
  HandState = thumb_mm  (single float: physical position of thumb on keyboard)

  Knowing (note, finger) → thumb_mm = key_pos_mm(note.pitch) - (f-1) * WK_MM
  Knowing thumb_mm → all 5 finger positions are determined.

Cost function (4 components):
  1. SHIFT    — how much did the hand (thumb) move? Lazy First: free zone ±12mm
  2. STRETCH  — physical span vs comfortable/max span for this finger pair
  3. CROSSING — thumb-under / finger-over:  natural?  or wrong direction?
  4. DIRECTION— does the finger choice align with note-stream direction?
  5. OFOK     — same finger on different pitch: always penalized

Segmentation:
  Split at HandResetType.FULL/SOFT points. Each segment runs an independent
  Viterbi — no cross-segment stitch cost.

Look-ahead:
  Direction of a segment (ASCENDING / DESCENDING / STATIC) is computed from
  the first N notes. Used to bias init_cost at segment start.
"""

from __future__ import annotations
from enum import Enum, auto
from typing import List, Tuple

from fingering.models.note_event import NoteEvent
from fingering.core.keyboard import (
    physical_key_position_mm,
    span_cost,
    _WHITE_KEY_WIDTH_MM,
)
from fingering.phrasing.hand_reset import classify_reset, HandResetType


# ──────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────

# "Free zone" for hand shift — moving less than this is essentially free.
# ~half a white key width; the hand naturally rocks this much.
_FREE_ZONE_MM = 12.0

# Cost weight for hand shift beyond free zone.
_SHIFT_WEIGHT = 0.5

# Penalty for same finger on different pitch (OFOK).
_OFOK_PENALTY = 15.0

# Penalty for moving in the wrong cross direction.
_WRONG_CROSS_PENALTY = 8.0

# Soft nudge for direction misalignment (not a hard rule).
_DIRECTION_NUDGE = 2.0

# Large number used as "effectively impossible" (not float('inf') to keep DP stable).
_IMPOSSIBLE = 1000.0

# Look-ahead window for direction detection.
_LOOKAHEAD_WINDOW = 8


class Direction(Enum):
    ASCENDING  = auto()
    DESCENDING = auto()
    STATIC     = auto()


# ──────────────────────────────────────────────────────────────
# thumb_mm helper
# ──────────────────────────────────────────────────────────────

def _thumb_mm(pitch: int, finger: int, hand: str = 'right') -> float:
    """
    Given a note (pitch) played by `finger`, return the physical position
    of the thumb (mm from keyboard origin).

    Right hand: thumb is the lowest finger, sits LEFT of higher fingers.
      thumb_mm = key_pos - (finger - 1) * WK_MM

    Left hand: thumb plays the highest pitch key, sits RIGHT of other fingers.
      thumb_mm = key_pos + (finger - 1) * WK_MM
    """
    pos = physical_key_position_mm(pitch)
    offset = (finger - 1) * _WHITE_KEY_WIDTH_MM
    if hand == 'right':
        return pos - offset
    else:
        return pos + offset


# ──────────────────────────────────────────────────────────────
# Direction detection
# ──────────────────────────────────────────────────────────────

def _detect_direction(notes: List[NoteEvent], window: int = _LOOKAHEAD_WINDOW) -> Direction:
    """
    Classify the overall direction of the first `window` notes.
    net_move > 2 → ASCENDING, < -2 → DESCENDING, else STATIC.
    """
    pitches = [n.pitch for n in notes[:window]]
    if len(pitches) < 2:
        return Direction.STATIC
    net = sum(
        1 if pitches[i + 1] > pitches[i] else -1 if pitches[i + 1] < pitches[i] else 0
        for i in range(len(pitches) - 1)
    )
    if net > 2:
        return Direction.ASCENDING
    if net < -2:
        return Direction.DESCENDING
    return Direction.STATIC


# ──────────────────────────────────────────────────────────────
# Cost functions
# ──────────────────────────────────────────────────────────────

def _init_cost(note: NoteEvent, finger: int, direction: Direction) -> float:
    """
    Cost of starting a segment with `finger` on `note`.

    Biases: thumb(1) on black key is bad; direction determines preferred finger range.
    """
    cost = 0.0

    # Thumb on black key at start of segment
    if note.is_black and finger == 1:
        cost += 5.0

    # Direction bias: give the hand room to move towards the melody's direction
    if direction == Direction.ASCENDING:
        # Prefer starting low (finger 1-2) so we can climb
        cost += (finger - 1) * 0.5
    elif direction == Direction.DESCENDING:
        # Prefer starting high (finger 4-5) so we can descend
        cost += (5 - finger) * 0.5

    return cost


def _transition_cost(
    note_a: NoteEvent, f_a: int,
    note_b: NoteEvent, f_b: int,
    hand: str,
    direction: Direction,
) -> float:
    """
    Physical cost of moving from (note_a, f_a) → (note_b, f_b).

    Components:
      1. SHIFT    — thumb movement (lazy first: free zone ±12mm)
      2. STRETCH  — physical finger span (3-zone model from keyboard.py)
      3. CROSSING — is the cross direction correct for the hand?
      4. DIRECTION— does finger choice align with stream direction?
      5. OFOK     — same finger, different pitch
    """
    cost = 0.0

    # ── 1. SHIFT (lazy first) ─────────────────────────────────────────────
    thumb_a = _thumb_mm(note_a.pitch, f_a, hand)
    thumb_b = _thumb_mm(note_b.pitch, f_b, hand)
    shift = abs(thumb_b - thumb_a)
    if shift > _FREE_ZONE_MM:
        excess = (shift - _FREE_ZONE_MM) / _WHITE_KEY_WIDTH_MM  # in WK units
        cost += excess ** 2 * _SHIFT_WEIGHT

    # ── 2. STRETCH (3-zone span model) ───────────────────────────────────
    span_mm = abs(physical_key_position_mm(note_b.pitch)
                  - physical_key_position_mm(note_a.pitch))
    cost += span_cost(span_mm, f_a, f_b)

    # ── 3. CROSSING ───────────────────────────────────────────────────────
    ascending = note_b.pitch > note_a.pitch

    if hand == 'right':
        if ascending and f_b < f_a and f_b != 1:
            # Going up but using a lower finger that isn't thumb-under → wrong
            cost += _WRONG_CROSS_PENALTY
        if not ascending and f_b > f_a and f_a != 1:
            # Going down but using a higher finger without thumb as anchor → wrong
            cost += _WRONG_CROSS_PENALTY
    else:  # left hand — mirror image
        if ascending and f_b > f_a and f_b != 1:
            cost += _WRONG_CROSS_PENALTY
        if not ascending and f_b < f_a and f_a != 1:
            cost += _WRONG_CROSS_PENALTY

    # ── 4. DIRECTION alignment (soft nudge, not hard rule) ────────────────
    if direction == Direction.ASCENDING:
        if hand == 'right' and f_b < f_a and f_b != 1:
            cost += _DIRECTION_NUDGE
        if hand == 'left' and f_b > f_a and f_b != 1:
            cost += _DIRECTION_NUDGE
    elif direction == Direction.DESCENDING:
        if hand == 'right' and f_b > f_a and f_a != 1:
            cost += _DIRECTION_NUDGE
        if hand == 'left' and f_b < f_a and f_a != 1:
            cost += _DIRECTION_NUDGE

    # ── 5. OFOK (one finger, one key — reuse penalty) ────────────────────
    if f_a == f_b and note_a.pitch != note_b.pitch:
        cost += _OFOK_PENALTY

    return cost


# ──────────────────────────────────────────────────────────────
# Segmentation
# ──────────────────────────────────────────────────────────────

def _segment_at_resets(
    notes: List[NoteEvent],
    bpm: float,
) -> List[List[NoteEvent]]:
    """
    Split the note stream at FULL reset points.

    A FULL reset (rest >= 0.4s) starts a new independent segment.
    A SOFT reset is NOT a split — it just removes the stitch cost within the DP
    by setting a flag (handled in solve() as a "free" first note in subsegment).
    """
    if not notes:
        return []

    segments: List[List[NoteEvent]] = []
    current: List[NoteEvent] = [notes[0]]

    for i in range(1, len(notes)):
        reset = classify_reset(notes, idx=i - 1, bpm=bpm)
        if reset == HandResetType.FULL:
            segments.append(current)
            current = [notes[i]]
        else:
            current.append(notes[i])

    if current:
        segments.append(current)

    return segments


# ──────────────────────────────────────────────────────────────
# Viterbi DP
# ──────────────────────────────────────────────────────────────

def _viterbi(
    notes: List[NoteEvent],
    hand: str,
    direction: Direction,
    first_finger_hint: int | None = None,
) -> List[int]:
    """
    Run Viterbi DP over a segment, returning the optimal finger sequence.

    Args:
        notes:             Segment of NoteEvents.
        hand:              'right' or 'left'.
        direction:         Pre-computed direction for this segment.
        first_finger_hint: If set, bias but don't force the first finger.
    """
    n = len(notes)
    if n == 0:
        return []
    if n == 1:
        # Single note: pick the finger with lowest init cost
        costs = [_init_cost(notes[0], f, direction) for f in range(1, 6)]
        return [1 + costs.index(min(costs))]

    INF = float('inf')

    # dp[f-1] = min cost to reach current note with finger f
    # bt[i][f-1] = best previous finger at step i
    dp_prev = [_init_cost(notes[0], f, direction) for f in range(1, 6)]
    if first_finger_hint is not None:
        # Soft hint: boost non-hint fingers slightly
        for f in range(1, 6):
            if f != first_finger_hint:
                dp_prev[f - 1] += 1.0

    bt: List[List[int]] = []  # bt[i] = list of best prev fingers (1-indexed)

    for i in range(1, n):
        dp_curr = [INF] * 5
        bt_curr = [1] * 5

        for f_curr in range(1, 6):
            best_cost = INF
            best_prev = 1

            for f_prev in range(1, 6):
                c = dp_prev[f_prev - 1] + _transition_cost(
                    notes[i - 1], f_prev,
                    notes[i],     f_curr,
                    hand, direction,
                )
                if c < best_cost:
                    best_cost = c
                    best_prev = f_prev

            dp_curr[f_curr - 1] = best_cost
            bt_curr[f_curr - 1] = best_prev

        dp_prev = dp_curr
        bt.append(bt_curr)

    # Backtrack
    fingers = [0] * n
    fingers[n - 1] = 1 + dp_prev.index(min(dp_prev))

    for i in range(n - 2, -1, -1):
        fingers[i] = bt[i][fingers[i + 1] - 1]

    return fingers


# ──────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────

class SimpleFingering:
    """
    Physics-first piano fingering via segmented Viterbi DP.

    Usage:
        sf = SimpleFingering()
        fingers = sf.solve(notes, bpm=120.0)
    """

    def __init__(self, bpm: float = 120.0):
        self.bpm = bpm

    def solve(
        self,
        notes: List[NoteEvent],
        bpm: float | None = None,
    ) -> List[int]:
        """
        Assign fingers 1–5 to each note.

        Args:
            notes: Sorted list of NoteEvents (single hand).
            bpm:   Tempo override (default: self.bpm).

        Returns:
            List[int] of length len(notes), fingers 1–5.
        """
        if not notes:
            return []

        effective_bpm = bpm if bpm is not None else self.bpm
        hand = notes[0].hand if notes[0].hand in ('right', 'left') else 'right'

        segments = _segment_at_resets(notes, effective_bpm)

        result: List[int] = []
        for seg in segments:
            direction = _detect_direction(seg)
            fingers = _viterbi(seg, hand, direction)
            result.extend(fingers)

        return result
