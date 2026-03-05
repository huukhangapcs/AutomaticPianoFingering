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
    melodic_leap_compensation: float = 0.0 # Leap >= P4 followed by stepwise resolution (0-1)
    # === Fix 5: Phrase-final lengthening (review.txt Signal 2) ===
    phrase_final_lengthening: float = 0.0  # Duration increases into this note (0–1)

    def boundary_score(self) -> float:
        """
        [Phase 3: Deep Learning Preparatory]
        Fuse all signals → single boundary probability [0..1] via Sigmoid.

        Instead of hard threshold logic, we treat these raw signals as features
        and compute a logit value, which is then squashed by a sigmoid.
        This provides a smooth probability surface for the downstream 
        DP/Viterbi segmentation engine to optimize globally.
        """
        import math
        
        # Base bias (negative to suppress noise, boundary is a rare event)
        logit = -2.5
        
        # --- Strong Explicit Signals ---
        if self.slur_end:
            logit += 1.8                               # Slur naturally signifies breath
            
        if self.rest_follows:
            if self.rest_duration >= 1.0:
                logit += 4.0                           # Quarter rest: Almost certainly
            elif self.rest_duration >= 0.5:
                logit += 2.0                           # Eighth rest: Likely
            else:
                logit += 0.8                           # Short rest: Requires support

        # --- Musical Context Signals ---
        if self.agogic_accent > 0:
            if self.agogic_accent >= 1.0 or self.note_duration >= 3.0:
                logit += 3.5                           # Extremely long note
            elif self.agogic_accent >= 0.5 or self.note_duration >= 2.0:
                logit += 2.0                           
            else:
                logit += 1.0 * self.agogic_accent

        logit += 1.5 * self.cadence_strength           # Metric + Interval cues
        logit += 2.0 * self.harmonic_cadence           # Bass V→I motion
        logit += 1.2 * self.melodic_leap_compensation  # Leap >= P4 then stepwise resolve
        logit += 1.0 * self.melodic_resolution         # Stepwise descent
        logit += 0.8 * float(self.next_note_is_downbeat) # Phrase lands on beat 1
        logit += 0.8 * float(self.large_interval)        # Leap = topic change
        logit += 0.3 * float(self.dynamic_change)        # Dynamic shift
        logit += 1.5 * self.phrase_final_lengthening     # Duration ramp-up before boundary
        
        # We purposely exclude `phrase_length_prior` from the low-level P(boundary) 
        # so that the Viterbi/DP solver can apply the prior as an edge transition cost!
        
        probability = 1.0 / (1.0 + math.exp(-logit))
        return probability


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
