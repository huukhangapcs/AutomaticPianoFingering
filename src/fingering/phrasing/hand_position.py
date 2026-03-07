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

# ── Anatomical Finger Offset Table ───────────────────────────────────
# Physical offset (mm) from thumb (f1) to each finger in a natural,
# relaxed hand position (right hand). Based on adult hand anthropometry:
#   Thumb (f1)   : anchor = 0 mm
#   Index (f2)   : ~42 mm  (≈1.8 WK — wider gap due to thumb abduction)
#   Middle (f3)  : ~72 mm  (≈3.1 WK — longest finger, natural arch)
#   Ring (f4)    : ~97 mm  (≈4.1 WK — tendon-coupled to middle)
#   Pinky (f5)   : ~118 mm (≈5.0 WK — smallest, shortest reach)
# Left hand mirrors (negative offsets).
# NOTE: Previous model used linear (f-1)*WK = 0, 23.5, 47, 70.5, 94 mm —
#       significantly underestimating index reach and overestimating ring/pinky.
_FINGER_OFFSET_MM: dict[int, float] = {
    1:   0.0,   # Thumb   (anchor)
    2:  42.0,   # Index   (~1.8 WK)
    3:  72.0,   # Middle  (~3.1 WK)
    4:  97.0,   # Ring    (~4.1 WK)
    5: 118.0,   # Pinky   (~5.0 WK)
}

# ── Per-finger position tolerance (mm) ───────────────────────────────
# How far a note can be from the expected finger position and still be
# considered "in position" (no hand repositioning needed).
# Thumb is most flexible (abducts widely); pinky is least flexible.
_FINGER_TOLERANCE_MM: dict[int, float] = {
    1: 25.0,   # Thumb:  very flexible, wide abduction range
    2: 18.0,   # Index:  good independence
    3: 15.0,   # Middle: moderate
    4: 12.0,   # Ring:   limited by tendon coupling
    5: 10.0,   # Pinky:  least flexible
}

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
    Other fingers follow anatomical offsets (_FINGER_OFFSET_MM), NOT uniform
    white-key spacing. The offsets reflect real hand geometry:

        Right Hand:
            f1 = thumb_mm + 0 mm   (thumb)
            f2 = thumb_mm + 42 mm  (index — wider gap due to thumb abduction)
            f3 = thumb_mm + 72 mm  (middle)
            f4 = thumb_mm + 97 mm  (ring)
            f5 = thumb_mm + 118 mm (pinky)

        Left Hand (mirror — offsets are negated):
            f1 = thumb_mm - 0 mm
            f2 = thumb_mm - 42 mm
            ... etc.
    """
    thumb_mm: float   # mm position of thumb (f1)
    hand: str         # 'right' | 'left'

    def finger_mm(self, f: int) -> float:
        """Physical position (mm) of finger f using anatomical offsets."""
        offset = _FINGER_OFFSET_MM[f]
        if self.hand == 'left':
            offset = -offset
        return self.thumb_mm + offset

    def distance_to(self, other: 'HandState') -> float:
        """Physical distance (mm) the thumb must travel to reach `other`."""
        return abs(self.thumb_mm - other.thumb_mm)

    def is_in_position(self, note: NoteEvent, finger: int) -> bool:
        """
        True if `note` can be played with `finger` without moving the hand.

        Uses per-finger tolerance: thumb is most flexible (±25mm),
        pinky is least flexible (±10mm).
        """
        expected_mm = self.finger_mm(finger)
        actual_mm   = physical_key_position_mm(note.pitch)
        tolerance   = _FINGER_TOLERANCE_MM[finger]
        return abs(actual_mm - expected_mm) <= tolerance

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

        Uses anatomical offsets (not uniform WK spacing) to back-calculate
        where the thumb must be when finger `f` is on `note`.

        For RH: thumb is to the LEFT of finger f by _FINGER_OFFSET_MM[f].
            thumb_mm = note_mm - _FINGER_OFFSET_MM[finger]

        For LH: thumb is to the RIGHT of finger f (keyboard mirror).
            thumb_mm = note_mm + _FINGER_OFFSET_MM[finger]
        """
        hand    = getattr(note, 'hand', 'right')
        note_mm = physical_key_position_mm(note.pitch)
        offset  = _FINGER_OFFSET_MM[finger]
        if hand == 'right':
            thumb_mm = note_mm - offset
        else:
            thumb_mm = note_mm + offset
        return HandState(thumb_mm=thumb_mm, hand=hand)

    def shift_cost(
        self,
        prev: HandState,
        curr: HandState,
        f_prev: int | None = None,
        f_curr: int | None = None,
    ) -> float:
        """
        Cost of moving the hand from `prev` to `curr` position.

        Uses thumb_mm delta (actual physical distance) instead of MIDI pitch.

        Free zone  : ≤ _FREE_SHIFT_MM (~½ white key) — micro-adjustments.
        Quadratic  : excess² × 0.5 / WK²  — large shifts are very expensive.
        Twist bonus: when hand barely moves but finger index jumps ≥3 positions,
                     add a small penalty for the awkward wrist/hand rotation
                     required. E.g., 1→5 at the same keyboard position is
                     technically in-range but biomechanically awkward.

        Examples (WK = 23.5 mm):
            shift =  0 mm → cost 0.0   (perfectly in position)
            shift = 12 mm → cost 0.0   (within free zone)
            shift = 35 mm → cost 0.5×(23.5/23.5)² = 0.5  (1 WK excess)
            shift = 94 mm → cost 0.5×(82/23.5)² ≈ 6.1  (4 WK excess = octave shift)
        """
        shift_mm = prev.distance_to(curr)

        # ── Physical displacement cost (quadratic beyond free zone) ──
        if shift_mm <= _FREE_SHIFT_MM:
            base = 0.0
        else:
            excess_mm = shift_mm - _FREE_SHIFT_MM
            excess_wk = excess_mm / _WK
            base = excess_wk ** 2 * 0.5

        # ── Finger-twist penalty ──────────────────────────────────────
        # When thumb barely moves but finger index jumps ≥3 (e.g. 1→5 in same
        # position), the hand must supinate/pronate. Add a soft penalty.
        twist = 0.0
        if f_prev is not None and f_curr is not None:
            f_delta = abs(f_curr - f_prev)
            if shift_mm <= _FREE_SHIFT_MM and f_delta >= 3:
                # Scale: Δf=3 → 0.3, Δf=4 → 0.6  (kept light, just a tiebreaker)
                twist = (f_delta - 2) * 0.3

        return base + twist


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Hand Movement Taxonomy
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

from enum import Enum, auto


class HandMovementType(Enum):
    """
    The 5 fundamental types of hand movement in piano playing.

    Each transition between two consecutive (note, finger) pairs falls into
    exactly one of these categories. The DP uses this classification to apply
    the appropriate biomechanical cost instead of scattered ad-hoc rules.

                         Ascending RH  |  Descending RH
    ───────────────────────────────────────────────────────
    Hand stays in place  IN_FORM       |  IN_FORM
    Fingers reach out    STRETCHING    |  STRETCHING
    Thumb crosses under  THUMB_UNDER   |  (not applicable)
    Finger crosses over  (applies LH)  |  CROSS_OVER
    Full reposition      RESET         |  RESET
    """

    IN_FORM     = auto()   # Hand stays in position; finger within _FINGER_TOLERANCE_MM
    STRETCHING  = auto()   # Finger reaches beyond normal position; no crossing
    THUMB_UNDER = auto()   # Ascending: current f∈{2,3} → next f=1 (thumb crosses under)
    CROSS_OVER  = auto()   # Descending: current f=1 → next f∈{2,3} (finger crosses over)
    RESET       = auto()   # Hand lifts and repositions; thumb_mm shift > threshold


def classify_movement(
    note_prev: NoteEvent,
    f_prev: int,
    note_curr: NoteEvent,
    f_curr: int,
    tracker: Optional['HandPositionTracker'] = None,
) -> HandMovementType:
    """
    Classify the transition from (note_prev, f_prev) → (note_curr, f_curr).

    Classification rules (in priority order):

    1. THUMB_UNDER — ascending passage, current finger is 2 or 3, next is 1.
       The thumb physically crosses underneath the active finger.
       Right hand: note goes up in pitch, f_curr=1 < f_prev ∈ {2,3}.
       Left  hand: note goes DOWN in pitch (mirror), f_prev=1 → f_curr ∈ {2,3}.

    2. CROSS_OVER — descending passage, current finger is 1, next is 2 or 3.
       A mid-length finger arches over the thumb.
       Right hand: note goes down in pitch, f_prev=1 → f_curr ∈ {2,3}.
       Left  hand: note goes UP in pitch (mirror), f_prev ∈ {2,3} → f_curr=1.

    3. IN_FORM — hand stays in position (is_in_position check passes).
       Note falls within per-finger tolerance of expected position.

    4. RESET — thumb must travel far (> FREE_SHIFT_MM after free zone).
       A physical repositioning of the entire hand.

    5. STRETCHING — the 'else': note is out of position but closer than RESET
       threshold. Fingers extend or contract without a full hand move.

    Args:
        note_prev, f_prev: previous note + finger
        note_curr, f_curr: current note + finger
        tracker: HandPositionTracker instance (creates one if None)

    Returns:
        HandMovementType enum value
    """
    from fingering.phrasing.hand_position import HandPositionTracker
    if tracker is None:
        tracker = HandPositionTracker()

    hand      = getattr(note_curr, 'hand', 'right')
    ascending = note_curr.pitch > note_prev.pitch

    # ── 1. Thumb-Under ─────────────────────────────────────────────────────
    # RH ascending: finger  f_prev ∈ {2,3} → f_curr = 1  (thumb slides under)
    # LH descending (mirrored): f_prev ∈ {2,3} → f_curr = 1
    if hand == 'right':
        is_thumb_under = (ascending and f_curr == 1 and f_prev in (2, 3))
    else:
        is_thumb_under = (not ascending and f_curr == 1 and f_prev in (2, 3))

    if is_thumb_under:
        return HandMovementType.THUMB_UNDER

    # ── 2. Cross-Over ──────────────────────────────────────────────────────
    # RH descending: f_prev = 1 → f_curr ∈ {2,3}  (finger arches over thumb)
    # LH ascending (mirrored): f_prev = 1 → f_curr ∈ {2,3}
    if hand == 'right':
        is_cross_over = (not ascending and f_prev == 1 and f_curr in (2, 3))
    else:
        is_cross_over = (ascending and f_prev == 1 and f_curr in (2, 3))

    if is_cross_over:
        return HandMovementType.CROSS_OVER

    # ── 3. In-Form ─────────────────────────────────────────────────────────
    state_prev = tracker.infer(note_prev, f_prev)
    if state_prev.is_in_position(note_curr, f_curr):
        return HandMovementType.IN_FORM

    # ── 4. Reset ───────────────────────────────────────────────────────────
    # Hand must physically reposition (thumb travels beyond free zone)
    state_curr = tracker.infer(note_curr, f_curr)
    shift = tracker.shift_cost(state_prev, state_curr)
    if shift > 0.0:
        return HandMovementType.RESET

    # ── 5. Stretching ──────────────────────────────────────────────────────
    # Out of position but no significant thumb travel — fingers extend/contract
    return HandMovementType.STRETCHING


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
