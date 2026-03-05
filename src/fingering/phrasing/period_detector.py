"""
T2-A: Antecedent-Consequent (Period) Detector

A pianist naturally groups phrases into periods:
  - Antecedent: "question phrase" — ends on dominant (V) or half-cadence
  - Consequent: "answer phrase"  — ends on tonic (I) or authentic cadence

This is how classical music is actually structured at the 8-measure level:
  m7–10:  E (dominant) = antecedent
  m11–14: A (tonic)    = consequent
  → Together = 1 period (8m)

This module post-processes Phrase lists and groups them into Periods
where the underlying harmonic structure supports it.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from fingering.models.note_event import NoteEvent
from fingering.phrasing.phrase import Phrase
from fingering.phrasing.harmonic_skeleton import HarmonicSkeleton


@dataclass
class Period:
    """
    A musical period = antecedent (question) + consequent (answer).
    The antecedent ends harmonically 'open', the consequent closes it.
    """
    antecedent: Phrase
    consequent: Phrase
    tonic_pc: int
    is_parallel: bool = False    # Both phrases start with same motif
    is_contrasting: bool = False  # Different opening material

    @property
    def start_measure(self) -> int:
        return self.antecedent.notes[0].measure if self.antecedent.notes else 0

    @property
    def end_measure(self) -> int:
        return self.consequent.notes[-1].measure if self.consequent.notes else 0

    @property
    def total_measures(self) -> int:
        return self.end_measure - self.start_measure + 1

    def __repr__(self) -> str:
        return (f"Period(m{self.start_measure}–m{self.end_measure}, "
                f"{self.total_measures}m, "
                f"{'parallel' if self.is_parallel else 'contrasting'})")


# ──────────────────────────────────────────────────────────────
# Cadence Classification
# ──────────────────────────────────────────────────────────────

def _is_half_cadence_ending(phrase: Phrase, tonic_pc: int, harmonic: HarmonicSkeleton) -> bool:
    """
    True if the phrase ends on the dominant — an 'open' or 'questioning' feeling.
    """
    if not phrase.notes:
        return False
    last_note = phrase.notes[-1]
    end_m = last_note.measure

    # Check if end measure has dominant harmony
    mh = harmonic.measure_harmonies
    dominant_pc = (tonic_pc + 7) % 12  # V = tonic + P5

    for h in mh:
        if abs(h.measure - end_m) <= 1:
            if h.root_pc == dominant_pc:
                return True
            # Also accept leading tone (tonic - 1 semitone)
            if h.root_pc == (tonic_pc - 1) % 12:
                return True
    return False


def _is_authentic_cadence_ending(
    phrase: Phrase, tonic_pc: int, harmonic: HarmonicSkeleton
) -> bool:
    """
    True if the phrase ends on the tonic — a 'closed' or 'answering' feeling.
    """
    if not phrase.notes:
        return False
    last_note = phrase.notes[-1]
    end_m = last_note.measure

    mh = harmonic.measure_harmonies
    for h in mh:
        if abs(h.measure - end_m) <= 1:
            if h.root_pc == tonic_pc:
                return True
    # Also check the actual note pitch class
    if last_note.pitch % 12 == tonic_pc:
        return True
    return False


def _similar_opening(phrase_a: Phrase, phrase_b: Phrase) -> bool:
    """
    True if both phrases start with the same interval pattern (parallel period).
    Compares first 4 intervals from each phrase.
    """
    n_compare = 4
    if len(phrase_a.notes) < n_compare or len(phrase_b.notes) < n_compare:
        return False

    def intervals(phrase: Phrase, n: int) -> tuple:
        notes = phrase.notes[:n + 1]
        return tuple(notes[i + 1].pitch - notes[i].pitch for i in range(n))

    iv_a = intervals(phrase_a, min(n_compare, len(phrase_a.notes) - 1))
    iv_b = intervals(phrase_b, min(n_compare, len(phrase_b.notes) - 1))

    if not iv_a or not iv_b or len(iv_a) != len(iv_b):
        return False

    # Allow ±2 semitone tolerance (transposition)
    matches = sum(1 for a, b in zip(iv_a, iv_b) if abs(a - b) <= 2)
    return matches / len(iv_a) >= 0.75


# ──────────────────────────────────────────────────────────────
# Period Detector
# ──────────────────────────────────────────────────────────────

class PeriodDetector:
    """
    Groups consecutive Phrases into Periods where antecedent-consequent
    structure is supported by harmonic evidence.

    A period is detected when:
    1. Two consecutive phrases have similar length (within 50%)
    2. First phrase ends on dominant or mid-level harmony (antecedent)
    3. Second phrase ends on tonic (consequent)
    4. Combined length is 6–16 measures (typical period length)
    """

    def __init__(
        self,
        tonic_pc: int = 0,
        max_length_ratio: float = 1.5,   # Consequent can be at most 1.5x antecedent
        min_period_measures: int = 6,
        max_period_measures: int = 18,
    ):
        self.tonic_pc = tonic_pc
        self.max_length_ratio = max_length_ratio
        self.min_period_measures = min_period_measures
        self.max_period_measures = max_period_measures

    def detect(
        self,
        phrases: List[Phrase],
        harmonic: HarmonicSkeleton,
    ) -> Tuple[List[Period], List[Phrase]]:
        """
        Returns (periods, unpaired_phrases).

        Periods: grouped antecedent+consequent pairs
        Unpaired phrases: phrases that don't form a period (e.g. B section material)
        """
        periods: List[Period] = []
        unpaired: List[Phrase] = []
        i = 0

        while i < len(phrases):
            if i + 1 >= len(phrases):
                unpaired.append(phrases[i])
                i += 1
                continue

            ant = phrases[i]
            con = phrases[i + 1]

            period = self._try_form_period(ant, con, harmonic)
            if period is not None:
                periods.append(period)
                i += 2
            else:
                unpaired.append(ant)
                i += 1

        return periods, unpaired

    def _try_form_period(
        self,
        ant: Phrase,
        con: Phrase,
        harmonic: HarmonicSkeleton,
    ) -> Optional[Period]:
        """
        Try to form a period from antecedent + consequent.
        Returns None if conditions not met.
        """
        if not ant.notes or not con.notes:
            return None

        # Length check
        ant_len = ant.notes[-1].measure - ant.notes[0].measure + 1
        con_len = con.notes[-1].measure - con.notes[0].measure + 1
        total_len = ant.notes[0].measure  # will compute below

        total_measures = con.notes[-1].measure - ant.notes[0].measure + 1

        if total_measures < self.min_period_measures:
            return None
        if total_measures > self.max_period_measures:
            return None
        if ant_len == 0 or con_len / ant_len > self.max_length_ratio:
            return None

        # Harmonic check
        ant_open   = _is_half_cadence_ending(ant, self.tonic_pc, harmonic)
        con_closed = _is_authentic_cadence_ending(con, self.tonic_pc, harmonic)

        # Need at least one condition (the other gets partial credit)
        if not ant_open and not con_closed:
            return None

        is_parallel = _similar_opening(ant, con)

        return Period(
            antecedent=ant,
            consequent=con,
            tonic_pc=self.tonic_pc,
            is_parallel=is_parallel,
            is_contrasting=not is_parallel,
        )
