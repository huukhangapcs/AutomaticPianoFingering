"""
Scale Fingering Database — Tone-Specific Fingering for All 12 Keys.

Chứa ngón tay chuẩn cho từng gam trên cả 2 tay, theo từng tone cụ thể
(bao gồm C major, G major, F# major, ...).

Mỗi entry là xâu chuỗi ngón tay: [RH_ascending, RH_descending, LH_ascending, LH_descending]

Reference: ABRSM/Faber Piano Scales and Exercises (Grade 5-8) + Hanon.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple

# Tonic pitch class (C=0, C#=1, ... B=11)
# Each value: (RH_ascending, RH_descending, LH_ascending, LH_descending)
# Finger lists: one entry per note in the octave (8 notes inclusive)
_MAJOR_SCALE_FINGERINGS: Dict[int, Tuple[List[int], List[int], List[int], List[int]]] = {
    # C major — no black keys
    0: (
        [1, 2, 3, 1, 2, 3, 4, 5],   # RH ascending
        [5, 4, 3, 2, 1, 3, 2, 1],   # RH descending
        [5, 4, 3, 2, 1, 3, 2, 1],   # LH ascending
        [1, 2, 3, 1, 2, 3, 4, 5],   # LH descending
    ),
    # G major — one black key (F#)
    7: (
        [1, 2, 3, 1, 2, 3, 4, 5],
        [5, 4, 3, 2, 1, 3, 2, 1],
        [5, 4, 3, 2, 1, 3, 2, 1],
        [1, 2, 3, 1, 2, 3, 4, 5],
    ),
    # D major — F#, C#
    2: (
        [1, 2, 3, 1, 2, 3, 4, 5],
        [5, 4, 3, 2, 1, 3, 2, 1],
        [5, 4, 3, 2, 1, 3, 2, 1],
        [1, 2, 3, 1, 2, 3, 4, 5],
    ),
    # A major — F#, C#, G#
    9: (
        [1, 2, 3, 1, 2, 3, 4, 5],
        [5, 4, 3, 2, 1, 3, 2, 1],
        [5, 4, 3, 2, 1, 3, 2, 1],
        [1, 2, 3, 1, 2, 3, 4, 5],
    ),
    # E major — 4 sharps
    4: (
        [1, 2, 3, 1, 2, 3, 4, 5],
        [5, 4, 3, 2, 1, 3, 2, 1],
        [5, 4, 3, 2, 1, 3, 2, 1],
        [1, 2, 3, 1, 2, 3, 4, 5],
    ),
    # B major — 5 sharps (RH starts on 1, LH uses 4 on root B)
    11: (
        [1, 2, 3, 1, 2, 3, 4, 5],
        [5, 4, 3, 2, 1, 3, 2, 1],
        [4, 3, 2, 1, 4, 3, 2, 1],   # LH B: start on 4 (thumb on E)
        [1, 2, 3, 4, 1, 2, 3, 4],
    ),
    # F# major — 6 sharps, both hands restructured
    6: (
        [2, 3, 4, 1, 2, 3, 4, 5],   # RH: start on 2 (F# = black key)
        [5, 4, 3, 2, 1, 4, 3, 2],
        [4, 3, 2, 1, 4, 3, 2, 1],   # LH: 4 on F#, 1 on B (only white)
        [1, 2, 3, 4, 1, 2, 3, 4],
    ),
    # Db major (= C# major) — 5 flats. RH starts on 2 as Db is black
    1: (
        [2, 3, 1, 2, 3, 4, 1, 2],   # RH: thumb only on white keys F, C
        [2, 1, 4, 3, 2, 1, 3, 2],
        [3, 4, 1, 2, 3, 1, 2, 3],   # LH: thumb on Gb, Db (white neighbors)
        [3, 2, 1, 3, 2, 1, 4, 3],
    ),
    # Ab major — 4 flats
    8: (
        [3, 4, 1, 2, 3, 1, 2, 3],   # RH: starts on 3 (Ab = black key)
        [3, 2, 1, 3, 2, 1, 4, 3],
        [3, 2, 1, 4, 3, 2, 1, 3],   # LH
        [3, 1, 2, 3, 4, 1, 2, 3],
    ),
    # Eb major — 3 flats
    3: (
        [3, 1, 2, 3, 4, 1, 2, 3],   # RH starts on 3 (Eb is black)
        [3, 2, 1, 4, 3, 2, 1, 3],
        [3, 2, 1, 4, 3, 2, 1, 3],   # LH
        [3, 1, 2, 3, 4, 1, 2, 3],
    ),
    # Bb major — 2 flats
    10: (
        [4, 1, 2, 3, 1, 2, 3, 4],   # RH: start on 4 (Bb is black)
        [4, 3, 2, 1, 3, 2, 1, 4],
        [3, 2, 1, 4, 3, 2, 1, 3],   # LH
        [3, 1, 2, 3, 4, 1, 2, 3],
    ),
    # F major — 1 flat (Bb)
    5: (
        [1, 2, 3, 4, 1, 2, 3, 4],   # RH: thumb avoids Bb → 4 under
        [4, 3, 2, 1, 4, 3, 2, 1],
        [5, 4, 3, 2, 1, 3, 2, 1],   # LH same as C major
        [1, 2, 3, 1, 2, 3, 4, 5],
    ),
}

# Natural minor (same as relative major except tonic shifts)
# Use same structure: mapped by tonic pc (0=Am, 2=Bm, etc.)
_NATURAL_MINOR_FINGERINGS: Dict[int, Tuple[List[int], List[int], List[int], List[int]]] = {
    # A minor — same as C major
    9: (
        [1, 2, 3, 1, 2, 3, 4, 5],
        [5, 4, 3, 2, 1, 3, 2, 1],
        [5, 4, 3, 2, 1, 3, 2, 1],
        [1, 2, 3, 1, 2, 3, 4, 5],
    ),
    # D minor
    2: (
        [1, 2, 3, 1, 2, 3, 4, 5],
        [5, 4, 3, 2, 1, 3, 2, 1],
        [5, 4, 3, 2, 1, 3, 2, 1],
        [1, 2, 3, 1, 2, 3, 4, 5],
    ),
    # G minor — Bb present
    7: (
        [1, 2, 3, 1, 2, 3, 4, 5],
        [5, 4, 3, 2, 1, 3, 2, 1],
        [5, 4, 3, 2, 1, 3, 2, 1],
        [1, 2, 3, 1, 2, 3, 4, 5],
    ),
    # C minor — Bb, Eb, Ab
    0: (
        [1, 2, 3, 1, 2, 3, 4, 5],
        [5, 4, 3, 2, 1, 3, 2, 1],
        [5, 4, 3, 2, 1, 3, 2, 1],
        [1, 2, 3, 1, 2, 3, 4, 5],
    ),
}


# ──────────────────────────────────────────────────────────────
# Hanon 5-Finger Exercise Patterns
# Mỗi pattern là chuỗi intervals bán cung cho 1 unit (lên + xuống)
# ──────────────────────────────────────────────────────────────

# Hanon exercise: stepwise 1-2-3-4-5 then back
HANON_PATTERN_FINGERS_RH = [1, 2, 3, 4, 5, 4, 3, 2, 1]  # Complete up+down
HANON_PATTERN_FINGERS_LH = [5, 4, 3, 2, 1, 2, 3, 4, 5]  # Mirror

# Double-note Hanon (thirds): 1-3 + 2-4 + ...
HANON_THIRDS_RH = [(1, 3), (2, 4), (3, 5), (2, 4), (1, 3)]
HANON_THIRDS_LH = [(5, 3), (4, 2), (3, 1), (4, 2), (5, 3)]


def get_major_scale_fingering(
    tonic_pc: int,
    hand: str,
    ascending: bool,
) -> Optional[List[int]]:
    """
    Return the canonical fingering for a major scale starting on `tonic_pc`.

    Args:
        tonic_pc: Pitch class of tonic (0=C, 1=Db, 2=D, ..., 11=B)
        hand: 'right' or 'left'
        ascending: True for ascending, False for descending

    Returns:
        List of 8 finger assignments, or None if unknown key.
    """
    entry = _MAJOR_SCALE_FINGERINGS.get(tonic_pc)
    if entry is None:
        return None
    rh_asc, rh_desc, lh_asc, lh_desc = entry
    if hand == 'right':
        return rh_asc if ascending else rh_desc
    else:
        return lh_asc if ascending else lh_desc


def get_minor_scale_fingering(
    tonic_pc: int,
    hand: str,
    ascending: bool,
) -> Optional[List[int]]:
    """Return the canonical fingering for a natural minor scale."""
    entry = _NATURAL_MINOR_FINGERINGS.get(tonic_pc)
    if entry is None:
        # Fallback to C major pattern
        return get_major_scale_fingering(0, hand, ascending)
    rh_asc, rh_desc, lh_asc, lh_desc = entry
    if hand == 'right':
        return rh_asc if ascending else rh_desc
    else:
        return lh_asc if ascending else lh_desc


def detect_scale_tonic(notes, template_steps) -> Optional[int]:
    """
    Given notes that match a scale template, infer the tonic pitch class.
    Returns the tonic PC (0-11) of the first note.
    """
    if not notes:
        return None
    return notes[0].pitch % 12


def get_hanon_fingering(hand: str, ascending: bool) -> List[int]:
    """
    Return the 5-finger Hanon exercise fingering.
    9 notes total (5 up + 4 back) for use in stepwise passages.
    """
    if hand == 'right':
        return HANON_PATTERN_FINGERS_RH if ascending else list(reversed(HANON_PATTERN_FINGERS_RH))
    else:
        return HANON_PATTERN_FINGERS_LH if ascending else list(reversed(HANON_PATTERN_FINGERS_LH))
