"""
Phrase — core data structures for phrase-aware fingering.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional
from fingering.models.note_event import NoteEvent


class PhraseIntent(Enum):
    """Musical intent of a phrase — shapes fingering decisions."""
    CANTABILE  = auto()   # Singing, legato, expressive
    BRILLIANT  = auto()   # Fast, virtuosic, clear articulation
    LEGATO     = auto()   # Smooth connection, no breaks
    STACCATO   = auto()   # Detached, bouncy
    EXPRESSIVE = auto()   # Default: shape towards climax


class ArcType(Enum):
    """Shape of the melodic contour within a phrase."""
    ARCH       = auto()   # Rises then falls (most common)
    CLIMB      = auto()   # Continuously ascending (tension)
    FALL       = auto()   # Continuously descending (release)
    WAVE       = auto()   # Oscillating (neutral)
    FLAT       = auto()   # Relatively static


@dataclass
class PhraseBoundarySignal:
    """
    All signals collected at a potential phrase boundary position.
    Position = index of the LAST note of the candidate phrase.
    """
    position: int
    slur_end: bool = False
    rest_follows: bool = False
    rest_duration: float = 0.0       # In beats
    cadence_strength: float = 0.0    # 0.0 → 1.0
    large_interval: bool = False     # |interval| > 7 semitones
    # === 3 Improvements ===
    melodic_arc: ArcType = ArcType.FLAT
    next_note_is_downbeat: bool = False  # Phrase lands on beat 1 of new measure
    phrase_length_prior: float = 0.5    # Bayesian prior from standard phrase lengths
    dynamic_change: bool = False

    def boundary_score(self) -> float:
        """Fuse all signals into a single boundary strength [0..1]."""
        score = 0.0
        score += 0.30 * float(self.slur_end)
        score += 0.25 * float(self.rest_follows) * min(1.0, self.rest_duration)
        score += 0.20 * self.cadence_strength
        # === Improvement 2: metric position ===
        score += 0.12 * float(self.next_note_is_downbeat)
        # === Improvement 3: phrase length prior ===
        score += 0.08 * self.phrase_length_prior
        score += 0.03 * float(self.large_interval)
        score += 0.02 * float(self.dynamic_change)
        return min(score, 1.0)


@dataclass
class Phrase:
    """A detected phrase — the primary unit of pianist thinking."""
    id: int
    notes: List[NoteEvent]
    hand: str                           # 'right' | 'left'
    boundary_score: float = 0.0

    # --- Filled by IntentAnalyzer ---
    intent: PhraseIntent = PhraseIntent.EXPRESSIVE
    arc_type: ArcType = ArcType.ARCH
    climax_idx: int = 0
    tension_curve: List[float] = field(default_factory=list)

    # --- Structural info ---
    starts_after_rest: bool = False

    @property
    def start_beat(self) -> float:
        return self.notes[0].onset if self.notes else 0.0

    @property
    def end_beat(self) -> float:
        return self.notes[-1].offset if self.notes else 0.0

    @property
    def duration_beats(self) -> float:
        return self.end_beat - self.start_beat

    @property
    def num_measures(self) -> float:
        if len(self.notes) < 2:
            return 1.0
        return self.notes[-1].measure - self.notes[0].measure + 1

    def __len__(self) -> int:
        return len(self.notes)

    def __repr__(self) -> str:
        return (f"Phrase(id={self.id}, notes={len(self.notes)}, "
                f"intent={self.intent.name}, arc={self.arc_type.name})")
