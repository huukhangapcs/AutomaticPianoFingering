"""
Hand Reset Detector — Physical repositioning classifier.

Determines whether the hand can reposition between two consecutive phrases,
based on the gap/held-note BEFORE the phrase starts. This is distinct from
musical phrase structure: a note can be long enough to allow repositioning
even if it's not a cadence or a musical boundary.

Three reset types (see HandResetType in phrase.py):
  FULL  — Long rest (>= ~0.4s): hand can jump anywhere, no stitch cost
  SOFT  — Short rest OR long held note, no slur: stitch is relaxed
  NONE  — Legato / no gap: hand must stay in position

Usage:
    from fingering.phrasing.hand_reset import classify_reset
    reset = classify_reset(notes, idx=last_note_of_prev_phrase, bpm=120.0)
"""

from __future__ import annotations
from typing import List

from fingering.models.note_event import NoteEvent
from fingering.phrasing.phrase import HandResetType

# ── Thresholds (absolute time, seconds) ──────────────────────────────────
# These are physical — a pianist can move their hand ~20cm in 0.3–0.4s.
# Calibrated for typical classical repertoire at quarter = 100–160 BPM.

# Rest gap >= this → FULL reset (hand can land anywhere)
_FULL_RESET_SEC  = 0.40   # ~half-note rest at 120 BPM

# Rest gap >= this → SOFT reset (fewer fingers blocked)
_SOFT_RESET_SEC  = 0.15   # ~8th-note rest at 120 BPM

# Note duration >= this → SOFT reset (finger held → other fingers free)
_HELD_NOTE_BEATS = 2.0    # half note or longer = enough time to reposition


def classify_reset(
    notes: List[NoteEvent],
    idx: int,
    bpm: float = 120.0,
) -> HandResetType:
    """
    Classify the hand reset opportunity between notes[idx] and notes[idx+1].

    Args:
        notes: Full note stream.
        idx:   Index of the LAST note of the preceding phrase.
        bpm:   Tempo in quarter notes per minute (used to convert beats → seconds).

    Returns:
        HandResetType.FULL, SOFT, or NONE.
    """
    if bpm <= 0:
        bpm = 120.0

    # No next note → trivially unconstrained (end of piece)
    if idx < 0 or idx + 1 >= len(notes):
        return HandResetType.FULL

    note_curr = notes[idx]
    note_next = notes[idx + 1]

    # ── Slur guard: cannot reposition while slurred ───────────────────────
    # If either the current note ends a slur or the next begins one,
    # the hand must maintain legato position.
    if note_curr.in_slur or note_next.in_slur:
        return HandResetType.NONE

    beats_per_sec = bpm / 60.0

    # ── Rest gap between notes ────────────────────────────────────────────
    gap_beats = note_next.onset - note_curr.offset
    gap_sec   = gap_beats / beats_per_sec if beats_per_sec > 0 else 0.0

    if gap_sec >= _FULL_RESET_SEC:
        return HandResetType.FULL

    if gap_sec >= _SOFT_RESET_SEC:
        return HandResetType.SOFT

    # ── Held note (no gap, but finger is down long enough) ────────────────
    # A long note lets the other fingers relax and reposition — especially
    # with sustain pedal. This doesn't give FULL freedom but does give SOFT.
    held_beats = note_curr.duration
    if held_beats >= _HELD_NOTE_BEATS and not note_curr.is_tied_stop:
        return HandResetType.SOFT

    return HandResetType.NONE
