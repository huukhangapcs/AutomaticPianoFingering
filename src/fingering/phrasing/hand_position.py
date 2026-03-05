"""
Hand State Tracking — Phase 3B (v2: Full Hand State Model).

Pianist thinking: a pianist establishes a "hand position" — all 5 fingers
hovering over 5 specific keys. When the next note falls under any of those
fingers, the hand stays put (zero cost). Only when a note lies OUTSIDE the
current span does the hand reposition.

v1 flaw: HandPositionTracker.infer() stored anchor_pitch = MIDI pitch of the
note being played. shift_cost() then compared those MIDI pitches directly,
causing false "large shifts" for in-position jumps like E5(f1)→A5(f4).

v2 fix: HandState stores thumb_mm — the physical position (mm) of the thumb
regardless of which finger is currently active. Two assignments sharing the
same thumb position have shift_cost = 0, even if the played pitches are far
apart.

Example:
    f1=E5 → thumb_mm = E5_mm
    f4=A5 → thumb_mm = A5_mm - 3*WHITE_KEY_MM = E5_mm  (same position!)
    → shift_cost = 0  ✅  (hand did NOT move)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Dict

from fingering.models.note_event import NoteEvent
from fingering.core.keyboard import physical_key_position_mm, _WHITE_KEY_WIDTH_MM

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Constants
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_WK = _WHITE_KEY_WIDTH_MM   # 23.5 mm per white-key step

# Tolerance: ±15mm (~0.6 WK) — notes this close to the expected finger
# position are considered "in-position" (micro-adjustments are free).
_IN_POSITION_TOLERANCE_MM = 15.0

# Shift below this threshold is free (hand micro-adjusts without repositioning).
# ~1 white key = 23.5mm.  We allow half a white key of free drift.
_FREE_SHIFT_MM = 12.0

# White keys (pitch class 0-indexed from C):
_WHITE_KEY_PCS = {0, 2, 4, 5, 7, 9, 11}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# HandState
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class HandState:
    """
    Full physical state of the hand at one moment in time.

    `thumb_mm` = physical center position (mm) of the thumb (finger 1).
    All other fingers follow from this anchor via white-key offsets:

        Right Hand (ascending keyboard direction = right):
            f1 = thumb_mm + 0*WK
            f2 = thumb_mm + 1*WK
            f3 = thumb_mm + 2*WK
            f4 = thumb_mm + 3*WK
            f5 = thumb_mm + 4*WK

        Left Hand (mirror — pinky is to the left of thumb):
            f1 = thumb_mm + 0*WK
            f2 = thumb_mm - 1*WK
            f3 = thumb_mm - 2*WK
            f4 = thumb_mm - 3*WK
            f5 = thumb_mm - 4*WK
    """
    thumb_mm: float   # mm position of thumb (f1)
    hand: str         # 'right' | 'left'

    def finger_mm(self, f: int) -> float:
        """Physical position (mm) of finger f in this hand state."""
        offset = (f - 1) * _WK
        if self.hand == 'left':
            offset = -offset
        return self.thumb_mm + offset

    def distance_to(self, other: 'HandState') -> float:
        """Physical distance (mm) the thumb must travel to reach `other`."""
        return abs(self.thumb_mm - other.thumb_mm)

    def is_in_position(self, note: NoteEvent, finger: int) -> bool:
        """
        True if `note` can be played with `finger` without moving the hand.

        We compare the note's physical key position to where finger `f` is
        expected to be in this hand state.  If the difference is within
        _IN_POSITION_TOLERANCE_MM, the hand stays put.
        """
        expected_mm = self.finger_mm(finger)
        actual_mm   = physical_key_position_mm(note.pitch)
        return abs(actual_mm - expected_mm) <= _IN_POSITION_TOLERANCE_MM

    def all_finger_positions(self) -> dict[int, float]:
        """Return {finger: mm} for all 5 fingers."""
        return {f: self.finger_mm(f) for f in range(1, 6)}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# HandPositionTracker  (v2)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class HandPositionTracker:
    """
    Infers HandState from (note, finger) pairs and scores repositioning cost.

    Stateless — safe to use as a shared singleton across DP calls.
    """

    def infer(self, note: NoteEvent, finger: int) -> HandState:
        """
        Reconstruct the hand state given that `note` is played with `finger`.

        For RH: thumb (f1) is to the LEFT of finger f.
            thumb_mm = note_mm - (f-1) * WK

        For LH: thumb (f1) is to the RIGHT of finger f (keyboard mirror).
            thumb_mm = note_mm + (f-1) * WK
        """
        hand     = getattr(note, 'hand', 'right')
        note_mm  = physical_key_position_mm(note.pitch)
        offset   = (finger - 1) * _WK
        if hand == 'right':
            thumb_mm = note_mm - offset
        else:
            thumb_mm = note_mm + offset
        return HandState(thumb_mm=thumb_mm, hand=hand)

    def shift_cost(self, prev: HandState, curr: HandState) -> float:
        """
        Cost of moving the hand from `prev` to `curr` position.

        Uses thumb_mm delta (actual physical distance) instead of MIDI pitch.

        Free zone  : ≤ _FREE_SHIFT_MM (~½ white key) — micro-adjustments.
        Quadratic  : excess² × 0.5 / WK²  — large shifts are very expensive.

        Examples (WK = 23.5 mm):
            shift =  0 mm → cost 0.0   (perfectly in position)
            shift = 12 mm → cost 0.0   (within free zone)
            shift = 35 mm → cost 0.5×(23.5/23.5)² = 0.5  (1 WK excess)
            shift = 94 mm → cost 0.5×(82/23.5)² ≈ 6.1  (4 WK excess = octave shift)
        """
        shift_mm = prev.distance_to(curr)
        if shift_mm <= _FREE_SHIFT_MM:
            return 0.0
        excess_mm = shift_mm - _FREE_SHIFT_MM
        # Normalise to white-key units so the scale is comparable to other costs
        excess_wk = excess_mm / _WK
        return excess_wk ** 2 * 0.5


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Five-Finger Segment Detection  (unchanged from v1)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Maximum semitone span covered by 5 fingers comfortably (RH/LH both)
_FIVE_FINGER_MAX_SPAN = 9   # Allow up to minor 6th (9 semitones) for flexibility

# Minimum run length to trigger five-finger segment bypass
_FIVE_FINGER_MIN_LEN = 4


@dataclass
class FiveFingerSegment:
    """A segment where the hand doesn't need to cross or shift."""
    start_idx: int
    end_idx: int         # exclusive
    fingers: List[int]
    ascending: bool


def detect_five_finger_segments(
    notes: List[NoteEvent],
    hand: str = 'right',
) -> List[FiveFingerSegment]:
    """
    Detect contiguous runs that fit comfortably under 5 fingers.

    Criteria:
      - At least _FIVE_FINGER_MIN_LEN notes
      - Purely stepwise motion (half or whole steps only, same direction)
      - Total pitch span ≤ _FIVE_FINGER_MAX_SPAN semitones
      - No need for thumb crossing
    """
    segments = []
    n = len(notes)
    i = 0

    while i < n - 1:
        ascending = notes[i + 1].pitch > notes[i].pitch

        j = i + 1
        while j < n - 1:
            diff = notes[j + 1].pitch - notes[j].pitch
            if ascending and diff in (1, 2):
                j += 1
            elif not ascending and diff in (-1, -2):
                j += 1
            else:
                break

        end = j + 1
        length = end - i

        if length >= _FIVE_FINGER_MIN_LEN:
            total_span = abs(notes[end - 1].pitch - notes[i].pitch)
            if total_span <= _FIVE_FINGER_MAX_SPAN:
                if hand == 'right':
                    if ascending:
                        fingers = list(range(1, min(6, length + 1)))
                    else:
                        fingers = list(range(min(5, length), 0, -1))
                else:
                    if ascending:
                        fingers = list(range(min(5, length), 0, -1))
                    else:
                        fingers = list(range(1, min(6, length + 1)))

                while len(fingers) < length:
                    fingers.append(fingers[-1])

                segments.append(FiveFingerSegment(
                    start_idx=i,
                    end_idx=end,
                    fingers=fingers[:length],
                    ascending=ascending,
                ))
                i = end
                continue

        i += 1

    return segments


def apply_five_finger_constraints(
    notes: List[NoteEvent],
    hand: str,
    existing_forced: Dict[int, int],
    skip_indices: Optional[set] = None,
) -> Dict[int, int]:
    """
    Inject five-finger segment fingerings as forced constraints.
    Only applied where no existing constraint is present.
    """
    forced = dict(existing_forced)
    _skip = skip_indices or set()
    segments = detect_five_finger_segments(notes, hand=hand)

    for seg in segments:
        for local_i, finger in enumerate(seg.fingers):
            global_i = seg.start_idx + local_i
            if global_i not in forced and global_i not in _skip:
                forced[global_i] = finger

    return forced
