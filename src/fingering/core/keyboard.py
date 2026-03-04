"""
Piano keyboard geometry helpers.

Span is measured in white-key units — the natural unit for hand stretch.
Two adjacent white keys = span 1. Octave (C→C) = span 7.
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
    """Absolute white-key distance between two notes."""
    return abs(note_a.white_key_index - note_b.white_key_index)


def is_ascending(note_a: NoteEvent, note_b: NoteEvent) -> bool:
    return note_b.pitch > note_a.pitch


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
