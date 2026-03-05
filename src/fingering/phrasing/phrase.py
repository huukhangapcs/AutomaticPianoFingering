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

    Signals are fused in boundary_score() — weights calibrated so that:
    - A single weak signal (rest alone) is NOT enough  
    - Two medium signals (rest + agogic, or cadence + downbeat) ARE enough
    - Slur end alone is strong enough (it's the most reliable signal)
    """
    position: int
    slur_end: bool = False
    rest_follows: bool = False
    rest_duration: float = 0.0       # In beats
    cadence_strength: float = 0.0    # 0.0 → 1.0 (metric + interval cues)
    large_interval: bool = False     # |interval| > 7 semitones
    # === Improvement signals ===
    melodic_arc: ArcType = ArcType.FLAT
    next_note_is_downbeat: bool = False  # Phrase lands on beat 1 of new measure
    phrase_length_prior: float = 0.0    # Bayesian prior from standard phrase lengths
    dynamic_change: bool = False
    # === New signals (Fix 2, Fix 3) ===
    note_duration: float = 0.0       # Absolute duration of the current note
    agogic_accent: float = 0.0       # Current note is locally longest (0–1 strength)
    melodic_resolution: float = 0.0  # Stepwise descent into long note (0–1)
    harmonic_cadence: float = 0.0    # Bass movement suggesting V→I or similar (0–1)

    def boundary_score(self) -> float:
        """
        Fuse all signals → single boundary strength [0..1].

        Weight philosophy:
          rest_follows    = 0.20  (necessary but not sufficient alone)
          slur_end        = 0.35  (strongest single signal when present)
          agogic_accent   = 0.20  (primary signal in absence of slurs)
          cadence_strength= 0.15  (harmonic + metric cues combined)
          harmonic_cadence= 0.15  (bass V→I motion)
          melodic_resol.  = 0.10  (stepwise descent into breathpoint)
          downbeat        = 0.10  (next note is beat 1 of new measure)
          phrase_len_prior= 0.10  (classical phrase length norms)
          large_interval  = 0.03  (leap suggests new topic)
          dynamic_change  = 0.02  (dynamic shift)

        Note: rest_follows is CAPPED so that rest alone (0.20) < threshold (0.40).
        Need at least one more supporting signal to confirm a boundary.
        """
        score = 0.0
        score += 0.35 * float(self.slur_end)                         # strongest
        
        # Scaling rest score by duration: long rests should force a boundary
        if self.rest_follows:
            if self.rest_duration >= 1.0:   # >= Quarter rest
                score += 0.45               # Forces boundary (threshold is 0.40)
            elif self.rest_duration >= 0.5: # >= Eighth rest
                score += 0.30
            else:                           # Short rest
                score += 0.22
                
        # Scaling agogic accent: very long notes force a boundary
        if self.agogic_accent > 0:
            if self.agogic_accent >= 1.0 or self.note_duration >= 3.0:
                score += 0.45               # Extremely long note -> forces boundary
            elif self.agogic_accent >= 0.5 or self.note_duration >= 2.0:
                score += 0.35               # Strong signal, forces boundary if next note is on downbeat
            else:
                score += 0.15 * self.agogic_accent
                
        score += 0.15 * self.cadence_strength                        # metric + interval
        score += 0.18 * self.harmonic_cadence                        # bass V→I (now precise)
        score += 0.10 * self.melodic_resolution                      # stepwise descent
        score += 0.10 * float(self.next_note_is_downbeat)            # phrase lands on 1
        score += 0.10 * self.phrase_length_prior                     # 4/8 bar norm
        score += 0.03 * float(self.large_interval)                   # leap = topic change
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
