"""
NoteEvent — Core data model for a single piano note.

White Key Index encoding:
  C=0, D=1, E=2, F=3, G=4, A=5, B=6, then +7 per octave.
  This gives a linear scale of keyboard position for span calculation.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional

# MIDI pitch → whether it is a black key
_BLACK_KEYS = {1, 3, 6, 8, 10}  # semitone offsets within an octave

# MIDI pitch → white key index within an octave (C=0 .. B=6)
_WHITE_KEY_IN_OCTAVE = {0: 0, 2: 1, 4: 2, 5: 3, 7: 4, 9: 5, 11: 6}
# For black keys we map to the nearest lower white key (for span purposes)
_SEMITONE_TO_WHITE = {0: 0, 1: 0, 2: 1, 3: 1, 4: 2, 5: 3,
                      6: 3, 7: 4, 8: 4, 9: 5, 10: 5, 11: 6}


def midi_to_white_key_index(pitch: int) -> int:
    """Linear white-key position on the keyboard (C0 = 0, D0 = 1, …)."""
    octave = pitch // 12
    semitone = pitch % 12
    return octave * 7 + _SEMITONE_TO_WHITE[semitone]


def is_black_key(pitch: int) -> bool:
    return (pitch % 12) in _BLACK_KEYS


@dataclass
class NoteEvent:
    """A single note in the piano score, enriched with keyboard geometry."""

    # --- Musical identity ---
    pitch: int              # MIDI pitch (60 = C4)
    onset: float            # Beat position (quarter-note beats)
    offset: float           # End beat
    hand: str               # 'right' | 'left'
    voice: int = 0          # Voice index within the hand (polyphony)
    measure: int = 0        # Bar number (1-indexed)
    beat: float = 1.0       # Beat within measure (1-indexed)

    # --- Articulation / expression marks (from MusicXML) ---
    in_slur: bool = False           # Is this note covered by a slur?
    slur_start: bool = False        # Does a slur start on this note?
    slur_end: bool = False          # Does a slur end on this note?
    is_staccato: bool = False
    has_accent: bool = False
    has_tenuto: bool = False
    dynamic: str = "mf"             # 'pp','p','mp','mf','f','ff'

    # --- Tie (distinct from slur: same pitch held across barline) ---
    is_tied_start: bool = False     # <tie type="start">: note starts a tie
    is_tied_stop: bool = False      # <tie type="stop">: note is held from previous

    # --- Keyboard geometry (auto-computed) ---
    is_black: bool = field(init=False)
    white_key_index: int = field(init=False)

    # --- Assigned fingering (filled by solver) ---
    finger: Optional[int] = None    # 1=thumb … 5=pinky

    def __post_init__(self):
        self.is_black = is_black_key(self.pitch)
        self.white_key_index = midi_to_white_key_index(self.pitch)

    @property
    def duration(self) -> float:
        return self.offset - self.onset

    @property
    def pitch_class(self) -> int:
        return self.pitch % 12

    @property
    def octave(self) -> int:
        return self.pitch // 12

    def __repr__(self) -> str:
        note_names = ['C', 'C#', 'D', 'D#', 'E', 'F',
                      'F#', 'G', 'G#', 'A', 'A#', 'B']
        name = note_names[self.pitch_class]
        oct_ = self.octave - 1  # MIDI octave convention
        f = self.finger if self.finger else '?'
        return f"NoteEvent({name}{oct_} m{self.measure}b{self.beat:.1f} f={f})"
