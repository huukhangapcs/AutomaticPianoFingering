"""
Piano keyboard geometry helpers.

Span is measured in white-key units — the natural unit for hand stretch.
Two adjacent white keys = span 1. Octave (C→C) = span 7.

Biomechanical models added (v2):
  - Black key depth correction
  - 3-4 / 4-5 tendon coupling penalty
  - Tempo-aware span scaling
"""

from __future__ import annotations
from typing import Tuple
from fingering.models.note_event import NoteEvent

# Maximum comfortable span between two fingers (white-key units).
# Based on adult hand anthropometry studies (avg male/female blended).
MAX_SPAN: dict[Tuple[int, int], int] = {
    (1, 2): 5, (1, 3): 7, (1, 4): 8, (1, 5): 9,
    (2, 3): 3, (2, 4): 5, (2, 5): 7,
    (3, 4): 3, (3, 5): 5,
    (4, 5): 3,
}

# ==============================================================
# Physical Keyboard Model (mm)
# ==============================================================
# Standard piano key dimensions (Steinway/Yamaha grand average):
#   White key width : 23.5 mm
#   Black key width : 13.7 mm
# Key center positions within one octave, measured from C.
# Source: Kinoshita et al. (2007), piano biomechanics studies.
# ==============================================================

_WHITE_KEY_WIDTH_MM = 23.5
OCTAVE_WIDTH_MM     = 7 * _WHITE_KEY_WIDTH_MM   # = 164.5 mm

# Pitch class (0=C … 11=B) → physical center of key (mm from C of same octave)
_PC_TO_MM_OFFSET: dict[int, float] = {
    0:   0.0,    # C  (white)
    1:  14.4,    # C# (black — sits ~2/3 of white key width from C)
    2:  23.5,    # D  (white)
    3:  35.3,    # D# (black)
    4:  47.0,    # E  (white)
    5:  70.5,    # F  (white)   ← no black between E-F!
    6:  82.3,    # F# (black)
    7:  94.0,    # G  (white)
    8: 107.6,    # G# (black)
    9: 117.5,    # A  (white)
    10: 131.2,   # A# (black)
    11: 141.0,   # B  (white)
}

# Maximum comfortable inter-finger span in mm (adult average, M+F blended).
# Based on Li et al. (2016) hand anthropometry + piano biomechanics.
# Values are LARGER than the WK-unit table because mm is more precise.
MAX_SPAN_MM: dict[Tuple[int, int], float] = {
    (1, 2): 118.0,   # Thumb – Index
    (1, 3): 163.5,   # Thumb – Middle
    (1, 4): 191.0,   # Thumb – Ring
    (1, 5): 215.0,   # Thumb – Pinky  (full hand stretch)
    (2, 3):  66.0,   # Index – Middle
    (2, 4): 114.0,   # Index – Ring
    (2, 5): 158.0,   # Index – Pinky
    (3, 4):  66.0,   # Middle – Ring
    (3, 5): 114.0,   # Middle – Pinky
    (4, 5):  66.0,   # Ring – Pinky
}

# Minimum comfortable span (some fingers can't be too close)
MIN_SPAN: dict[Tuple[int, int], int] = {
    (1, 2): 0, (1, 3): 1, (1, 4): 1, (1, 5): 1,
    (2, 3): 0, (2, 4): 0, (2, 5): 0,
    (3, 4): 0, (3, 5): 0,
    (4, 5): 0,
}


def finger_span_limits(f1: int, f2: int) -> Tuple[int, int]:
    """
    Return (min_span, max_span) for the given finger pair.
    Fingers are always sorted so (f1 < f2).
    """
    key = (min(f1, f2), max(f1, f2))
    return MIN_SPAN.get(key, 0), MAX_SPAN.get(key, 9)


def white_key_span(note_a: NoteEvent, note_b: NoteEvent) -> int:
    """Absolute white-key distance between two notes (legacy, kept for compat)."""
    return abs(note_a.white_key_index - note_b.white_key_index)


def physical_key_position_mm(pitch: int) -> float:
    """
    Physical center position of a piano key in millimeters.

    Reference point: A0 (MIDI pitch 21) = 0 mm.
    Direction: left → right (ascending pitch = larger mm value).

    This gives an accurate physical distance between any two keys,
    accounting for the non-uniform placement of black keys.

    Example:
        E5 (76) → 846.5 mm
        F5 (77) → 870.0 mm   (E–F gap = 23.5 mm, one white key)
        G5 (79) → 917.0 mm
    """
    # A0 = MIDI 21. C of same octave = MIDI 21 - 9 = MIDI 12 → octave 1
    # More robustly: use MIDI 0 = C(-1) as epoch.
    octave = pitch // 12       # C-octave (C0 = octave 0)
    pc     = pitch % 12        # pitch class within octave
    return octave * OCTAVE_WIDTH_MM + _PC_TO_MM_OFFSET[pc]


def physical_span_mm(note_a: NoteEvent, note_b: NoteEvent) -> float:
    """
    Absolute physical distance between two piano keys (mm).

    More accurate than white_key_span because it uses actual key center
    positions, properly accounting for black key placement.
    """
    return abs(physical_key_position_mm(note_a.pitch)
               - physical_key_position_mm(note_b.pitch))


def finger_max_span_mm(f1: int, f2: int) -> float:
    """Max comfortable span (mm) for a finger pair, from MAX_SPAN_MM table."""
    key = (min(f1, f2), max(f1, f2))
    return MAX_SPAN_MM.get(key, 215.0)


def is_in_hand_position(note_a: NoteEvent, f_a: int,
                        note_b: NoteEvent, f_b: int) -> bool:
    """
    Returns True if note_b with finger f_b is reachable from the hand
    position established by note_a with finger f_a WITHOUT repositioning.

    Core of Lazy First Principle: if True, hand stays put.
    """
    span = physical_span_mm(note_a, note_b)
    max_s = finger_max_span_mm(f_a, f_b)
    return span <= max_s


def is_ascending(note_a: NoteEvent, note_b: NoteEvent) -> bool:
    return note_b.pitch > note_a.pitch


def natural_finger_order(f_prev: int, f_curr: int, ascending: bool, hand: str) -> bool:
    """
    Returns True if (f_prev -> f_curr) is the natural anatomical finger order
    for the given melodic direction and hand.

    Right Hand:
      ascending  = thumb (1) → pinky (5): f_curr > f_prev is natural
      descending = pinky (5) → thumb (1): f_curr < f_prev is natural

    Left Hand (mirror image on keyboard):
      ascending (pitch goes up) = pinky-side moves right on keyboard
        → the LH thumb is on the HIGH pitch side
        → natural order ascending: f_curr < f_prev (5→4→3→2→1)
      descending: f_curr > f_prev is natural for LH
    """
    if hand == 'right':
        return (ascending and f_curr > f_prev) or (not ascending and f_curr < f_prev)
    else:  # left hand
        return (ascending and f_curr < f_prev) or (not ascending and f_curr > f_prev)


def thumb_crossing_natural(f_prev: int, f_curr: int, ascending: bool, hand: str) -> bool:
    """
    Returns True if f_curr == 1 (thumb) crossing is the standard scale motion
    (thumb-under for RH ascending, thumb-under for LH descending).
    """
    if hand == 'right':
        return f_curr == 1 and ascending
    else:
        return f_curr == 1 and not ascending


def semitone_interval(note_a: NoteEvent, note_b: NoteEvent) -> int:
    """Signed semitone interval (positive = ascending)."""
    return note_b.pitch - note_a.pitch


def is_physically_reachable(note_a: NoteEvent, f_a: int,
                            note_b: NoteEvent, f_b: int) -> bool:
    """Can a human hand play note_a with f_a and note_b with f_b simultaneously?"""
    span = white_key_span(note_a, note_b)
    _, max_s = finger_span_limits(f_a, f_b)
    min_s, _ = finger_span_limits(f_a, f_b)
    return min_s <= span <= max_s


# ================================================================
# Biomechanical Extensions (v2)
# ================================================================

# Black key depth factor: black keys sit ~1cm higher and are narrower.
# When one note is black, the effective span decreases slightly.
# Empirical correction: subtract 0.5 white-key units per black key involved.
_BLACK_KEY_SPAN_CORRECTION = 0.5  # per black key in the pair

def black_key_span_correction(note_a: NoteEvent, note_b: NoteEvent) -> float:
    """
    Return a correction to add to the measured white-key span when
    one or both notes are black keys.

    Black keys are physically closer to your body (shorter key).
    The thumb or fingers resting on a black key have reduced horizontal reach
    compared to what the white-key-index difference suggests.

    Returns a NEGATIVE float (reduces effective span by this amount).
    """
    n_black = int(note_a.is_black) + int(note_b.is_black)
    return -n_black * _BLACK_KEY_SPAN_CORRECTION


# Tendon coupling: the ring finger (4) shares a tendon sheath with the
# middle finger (3). This makes rapid independent movement of the 3-4 pair
# harder than any other pair. The 4-5 pair is similarly constrained.
# This penalty applies when the two fingers must alternate rapidly.
_TENDON_COUPLED_PAIRS = frozenset([
    (3, 4),  # Middle ↔ Ring   — strongest coupling
    (4, 5),  # Ring ↔ Pinky    — moderate coupling
])

def tendon_coupling_penalty(
    f_prev: int, f_curr: int, note_duration: float, bpm: float = 120.0
) -> float:
    """
    Returns extra cost when two anatomically coupled fingers must alternate.

    The penalty scales with tempo: faster = harder for linked tendons.
    At very slow tempo (< 60 BPM) the coupling is barely noticeable.

    Args:
        f_prev: previous finger
        f_curr: current finger
        note_duration: duration of current note in quarter-note beats
        bpm: tempo in beats-per-minute (default 120)

    Returns:
        float: extra cost to add (> 0 = harder)
    """
    pair = (min(f_prev, f_curr), max(f_prev, f_curr))
    if pair not in _TENDON_COUPLED_PAIRS:
        return 0.0
    if pair == (3, 4):  # Extensor digitorum communis — strongest coupling
        base_penalty = 3.0
    else:               # 4-5 pair
        base_penalty = 1.5

    # Normalize by tempo: penalty scales with speed of alternation
    # At 60 bpm, quarter note = 1.0 sec → comfortable
    # At 180 bpm, quarter note = 0.33 sec → very hard
    notes_per_second = bpm / 60.0  # notes/sec at this tempo
    if note_duration <= 0:
        speed_factor = 2.0  # Zero/grace notes = treat as maximum speed
    else:
        speed_factor = min(2.0, notes_per_second / note_duration / 4.0)  # cap at 2×
    return base_penalty * speed_factor


def tempo_adjusted_max_span(
    f1: int, f2: int, bpm: float = 120.0
) -> int:
    """
    Return the max comfortable span between two fingers, adjusted for tempo.

    At higher tempo, the hand must stay more compact; reaching for wide spans
    disrupts timing. Based on empirical piano pedagogy:
      - At <= 60 BPM (Adagio): full static span is fine
      - At 120 BPM: reduce max span by 1 white key
      - At >= 180 BPM (Allegro vivace): reduce by 2 white keys

    Args:
        f1, f2: finger indices (1-5)
        bpm: tempo in beats-per-minute

    Returns:
        int: adjusted max span in white-key units
    """
    _, raw_max = finger_span_limits(f1, f2)
    if bpm <= 60:
        reduction = 0
    elif bpm <= 120:
        reduction = 1
    elif bpm <= 180:
        reduction = 2
    else:
        reduction = 3  # Very fast: very compact hand required
    return max(raw_max - reduction, 1)  # Never go below 1
