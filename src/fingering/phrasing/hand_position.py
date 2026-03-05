"""
Hand Position State Tracking — Phase 3B improvement.

Pianist thinking: A pianist establishes a "hand position" (5-finger anchor)
and only shifts it when necessary. This creates natural ergonomic efficiency
because the hand stays balanced, reducing unnecessary movement.

Current gap:
  The Viterbi DP penalizes per-note span between consecutive notes but has
  no notion of the hand's "current position." A large jump at note N-5 can 
  place the hand poorly for notes N through N+10, but the DP only sees pairs.

This module provides:
  1. `HandPosition` — dataclass representing the current 5-finger anchor
  2. `HandPositionTracker` — infers position from (note, finger) pairs and
     computes shift penalties
  3. `detect_five_finger_segments` — identifies sequences where the entire
     run fits under 5 fingers with no crossing needed (fast bypass of DP)
  4. `apply_five_finger_constraints` — injects the bypass fingering as forced
     constraints into the DP dict, or returns a full override
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
from fingering.models.note_event import NoteEvent

# White keys, 0-indexed from C (relative to any octave)
# MIDI pitch % 12: 0=C, 2=D, 4=E, 5=F, 7=G, 9=A, 11=B
# Black keys: 1=C#, 3=D#, 6=F#, 8=G#, 10=A#
_WHITE_KEY_PCS = {0, 2, 4, 5, 7, 9, 11}

# Maximum semitone span covered by 5 fingers comfortably (RH/LH both)
# 8 semitones = perfect 5th (C-G), typical comfortable 5-finger range
_FIVE_FINGER_MAX_SPAN = 9   # Allow up to minor 6th (9 semitones) for flexibility

# Minimum run length to trigger five-finger segment bypass
_FIVE_FINGER_MIN_LEN = 4

# Position shift threshold: shifts > this many semitones are penalized
_SHIFT_SEMITONE_THRESHOLD = 3


@dataclass
class HandPosition:
    """
    The current 5-finger anchor position of the hand.

    `anchor_pitch` is the MIDI pitch of `anchor_finger` (usually thumb=1 for RH,
    pinky=5 for LH). All 5 fingers are then mapped to neighboring keys.

    Example (RH, anchor_finger=1=thumb, anchor_pitch=60=C4):
        finger 1 → C4 (60)
        finger 2 → D4 (62)
        finger 3 → E4 (64)
        finger 4 → F4 (65)
        finger 5 → G4 (67)
    """
    anchor_pitch: int    # MIDI pitch of the anchor finger
    anchor_finger: int   # Which finger is at the anchor (1=thumb for RH)
    hand: str = 'right'  # 'right' or 'left'

    @property
    def thumb_pitch(self) -> int:
        """The expected MIDI pitch of the thumb (finger 1)."""
        if self.hand == 'right':
            return self.anchor_pitch - (self.anchor_finger - 1) * 2
        else:
            return self.anchor_pitch + (self.anchor_finger - 1) * 2

    def semitone_shift(self, other: 'HandPosition') -> float:
        """Absolute semitone distance between two hand positions."""
        return abs(self.anchor_pitch - other.anchor_pitch)


class HandPositionTracker:
    """
    Infers hand position from (note, finger) assignments during DP and
    computes a penalty for large position shifts.

    Usage in _transition_cost():
        pos_prev = tracker.infer(note_prev, f_prev)
        pos_curr = tracker.infer(note_curr, f_curr)
        cost += tracker.shift_cost(pos_prev, pos_curr)
    """

    def infer(self, note: NoteEvent, finger: int) -> HandPosition:
        """
        Reconstruct the approximate hand position from a single (note, finger) pair.

        For RH: thumb (finger 1) is the leftmost anchor. Given finger N at pitch P,
        the thumb is approximately P - (N-1) * 2 semitones to the left (white keys).
        This is an approximation; the true position depends on the full scale.
        """
        hand = getattr(note, 'hand', 'right')
        pitch = note.pitch
        return HandPosition(
            anchor_pitch=pitch,
            anchor_finger=finger,
            hand=hand,
        )

    def shift_cost(self, pos_prev: HandPosition, pos_curr: HandPosition) -> float:
        """
        Return a cost for shifting hand position between two notes.

        Lazy First Principle: the hand should stay put unless forced to move.

        Small shifts (≤ threshold): forgiven — micro-adjustments are normal.
        Large shifts (> threshold): QUADRATIC penalty — makes large shifts
        very expensive, strongly discouraging unnecessary hand movement.
        """
        shift = pos_prev.semitone_shift(pos_curr)
        if shift <= _SHIFT_SEMITONE_THRESHOLD:
            return 0.0
        # Quadratic: 5-semitone shift → 0.5*4 = 2.0 cost
        #            10-semitone shift → 0.5*49 = 24.5 cost (was 2.1 linear!)
        excess = shift - _SHIFT_SEMITONE_THRESHOLD
        return excess ** 2 * 0.5


# ──────────────────────────────────────────────────────────────────────────
# Five-Finger Segment Detection
# ──────────────────────────────────────────────────────────────────────────

@dataclass
class FiveFingerSegment:
    """A segment where the hand doesn't need to cross or shift."""
    start_idx: int
    end_idx: int         # exclusive
    fingers: List[int]
    ascending: bool


def _is_white_key(pitch: int) -> bool:
    return (pitch % 12) in _WHITE_KEY_PCS


def _stepwise_and_restricted(notes: List[NoteEvent], start: int, end: int) -> bool:
    """
    Returns True if all notes in [start, end) form a purely stepwise run (1-2 semitones)
    in one direction with total span ≤ _FIVE_FINGER_MAX_SPAN.
    """
    if end - start < 2:
        return False
    ascending = notes[start + 1].pitch > notes[start].pitch
    for i in range(start, end - 1):
        diff = notes[i + 1].pitch - notes[i].pitch
        if ascending and diff not in (1, 2):
            return False
        if not ascending and diff not in (-1, -2):
            return False
    total_span = abs(notes[end - 1].pitch - notes[start].pitch)
    return total_span <= _FIVE_FINGER_MAX_SPAN


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

    Returns a list of FiveFingerSegment objects.
    """
    segments = []
    n = len(notes)
    i = 0

    while i < n - 1:
        # Check if a segment starts here
        ascending = notes[i + 1].pitch > notes[i].pitch

        # Grow the segment
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
                # Assign consecutive fingers (no crossing needed)
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

                # Pad if needed (shouldn't be but defensive)
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

    Args:
        skip_indices: Set of note indices to never constrain (e.g., climax note).
                      These positions will be left free for the DP to decide.

    Usage — call before DP:
        forced = apply_five_finger_constraints(notes, hand, forced, skip_indices={climax_idx})
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
