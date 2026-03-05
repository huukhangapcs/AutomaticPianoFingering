"""
PianoFingering — Thin pipeline wrapper for SimpleFingering.

Usage:
    from fingering.simple import PianoFingering

    pf = PianoFingering(bpm=120.0)
    rh_fingers = pf.run(rh_notes)
    lh_fingers = pf.run(lh_notes)
"""

from __future__ import annotations
from typing import List, Tuple, Optional

from fingering.models.note_event import NoteEvent
from fingering.simple.fingering_dp import SimpleFingering


class PianoFingering:
    """
    Physics-first piano fingering pipeline.

    Single-hand usage:
        pf = PianoFingering()
        fingering = pf.run(notes)          # → List[int]

    Annotate in-place:
        pf.run_and_annotate(notes)         # writes note.finger
    """

    def __init__(self, bpm: float = 120.0):
        self.bpm = bpm
        self._solver = SimpleFingering(bpm=bpm)

    def run(
        self,
        notes: List[NoteEvent],
        bpm: Optional[float] = None,
    ) -> List[int]:
        """Assign fingers 1–5 to each note. Returns list same length as notes."""
        return self._solver.solve(notes, bpm=bpm)

    def run_and_annotate(
        self,
        notes: List[NoteEvent],
        bpm: Optional[float] = None,
    ) -> List[NoteEvent]:
        """Run and write finger assignments back into NoteEvent.finger."""
        fingers = self.run(notes, bpm=bpm)
        for note, f in zip(notes, fingers):
            note.finger = f
        return notes

    def run_grand_staff(
        self,
        rh_notes: List[NoteEvent],
        lh_notes: List[NoteEvent],
        bpm: Optional[float] = None,
    ) -> Tuple[List[int], List[int]]:
        """Run both hands independently and return (rh_fingers, lh_fingers)."""
        rh = self.run(rh_notes, bpm=bpm)
        lh = self.run(lh_notes, bpm=bpm)
        return rh, lh
