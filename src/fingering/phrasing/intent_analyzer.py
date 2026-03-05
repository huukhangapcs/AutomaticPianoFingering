"""
Phrase Intent Analyzer — Layer B of the Phrase-Aware Fingering module.

Determines the musical intent of each detected phrase and computes the
tension arc (climax detection), which guides phrase-scoped DP decisions.

T2-B: Tension curve now blends shape (ARCH/CLIMB/FALL) with interval
      dissonance (tritone = max tension, perfect 5th = stable).
"""

from __future__ import annotations
import math
from typing import List
from fingering.models.note_event import NoteEvent
from fingering.phrasing.phrase import Phrase, PhraseIntent, ArcType


# Dynamics ordered from soft to loud
_DYNAMIC_RANK = {'pppp': 0, 'ppp': 1, 'pp': 2, 'p': 3, 'mp': 4,
                 'mf': 5, 'f': 6, 'ff': 7, 'fff': 8, 'ffff': 9}

# Note density threshold (notes per beat) that triggers BRILLIANT intent
_BRILLIANT_DENSITY = 3.5


def _dynamic_rank(d: str) -> int:
    return _DYNAMIC_RANK.get(d, 5)  # default mf


# ──────────────────────────────────────────────────────────────
# T2-B: Interval dissonance table
# Interval class → dissonance 0 (consonant) … 1 (maximally dissonant)
# ──────────────────────────────────────────────────────────────
_INTERVAL_DISSONANCE = {
    0:  0.0,   # Unison / octave
    7:  0.1,   # Perfect 5th
    5:  0.1,   # Perfect 4th
    4:  0.2,   # Major 3rd
    3:  0.2,   # Minor 3rd
    9:  0.3,   # Major 6th
    8:  0.3,   # Minor 6th
    2:  0.7,   # Major 2nd
    10: 0.7,   # Minor 7th
    1:  0.9,   # Minor 2nd
    11: 0.9,   # Major 7th (leading tone)
    6:  1.0,   # Tritone
}


def _dissonance_of_interval(semitones: int) -> float:
    """Return dissonance 0–1 for a given interval in semitones."""
    ic = abs(semitones) % 12
    return _INTERVAL_DISSONANCE.get(ic, 0.5)


class PhraseIntentAnalyzer:
    """
    Analyzes each Phrase to assign:
      - PhraseIntent (CANTABILE / BRILLIANT / LEGATO / STACCATO / EXPRESSIVE)
      - Climax index (highest tension note)
      - Tension curve [0.0 → 1.0] per note (shape + dissonance blended)

    Modelled after a pianist's reading of phrase character from notation.
    """

    def analyze(self, phrase: Phrase) -> Phrase:
        """Mutate phrase in-place to add intent, climax_idx, tension_curve."""
        phrase.intent = self._detect_intent(phrase)
        phrase.climax_idx = self._find_climax(phrase)
        phrase.tension_curve = self._compute_tension(phrase)
        return phrase

    def analyze_all(self, phrases: List[Phrase]) -> List[Phrase]:
        return [self.analyze(p) for p in phrases]

    # ------------------------------------------------------------------

    def _detect_intent(self, phrase: Phrase) -> PhraseIntent:
        """
        Rule-based intent detection from score markings.
        Priority: BRILLIANT > STACCATO > CANTABILE > LEGATO > EXPRESSIVE
        """
        notes = phrase.notes
        if not notes:
            return PhraseIntent.EXPRESSIVE

        duration = max(phrase.duration_beats, 0.01)
        density = len(notes) / duration
        dyn_rank = _dynamic_rank(notes[0].dynamic)
        staccato_ratio = sum(1 for n in notes if n.is_staccato) / len(notes)
        slur_ratio = sum(1 for n in notes if n.in_slur) / len(notes)

        if density >= _BRILLIANT_DENSITY and dyn_rank >= 6:
            return PhraseIntent.BRILLIANT
        if staccato_ratio >= 0.5:
            return PhraseIntent.STACCATO
        if slur_ratio >= 0.6 and dyn_rank <= 5:
            return PhraseIntent.CANTABILE
        if slur_ratio >= 0.4:
            return PhraseIntent.LEGATO
        return PhraseIntent.EXPRESSIVE

    def _find_climax(self, phrase: Phrase) -> int:
        """
        Find the climax note index — peak of musical tension.
        Primary: highest pitch. Tiebreaker: accent > later position.
        """
        notes = phrase.notes
        if not notes:
            return 0
        max_pitch = max(n.pitch for n in notes)
        candidates = [i for i, n in enumerate(notes) if n.pitch == max_pitch]
        accented = [i for i in candidates if notes[i].has_accent]
        return accented[0] if accented else candidates[-1]

    def _compute_tension(self, phrase: Phrase) -> List[float]:
        """
        Compute per-note tension curve [0.0 → 1.0].

        Blends:
          70% shape curve  (ARCH / CLIMB / FALL / WAVE / FLAT)
          30% dissonance   (T2-B: tritone=1.0, perfect 5th=0.1, etc.)

        This makes fingering stronger at genuinely dissonant moments,
        not just at pitch peaks — matching how a pianist feels tension.
        """
        n = len(phrase.notes)
        if n == 0:
            return []
        if n == 1:
            return [1.0]

        arc = phrase.arc_type
        climax = phrase.climax_idx
        notes = phrase.notes

        # Shape curve
        shape = []
        for i in range(n):
            if arc == ArcType.ARCH:
                if climax == 0:
                    t = 1.0 - i / max(n - 1, 1)
                elif i <= climax:
                    t = i / climax
                else:
                    t = 1.0 - (i - climax) / max(n - 1 - climax, 1)
            elif arc == ArcType.CLIMB:
                t = i / (n - 1)
            elif arc == ArcType.FALL:
                t = 1.0 - i / (n - 1)
            elif arc == ArcType.WAVE:
                t = 0.5 + 0.5 * math.sin(2 * math.pi * i / max(n - 1, 1))
            else:
                t = 0.5
            shape.append(t)

        # T2-B: Dissonance curve
        diss = [0.5]  # first note has no prior interval
        for i in range(1, n):
            diss.append(_dissonance_of_interval(notes[i].pitch - notes[i - 1].pitch))

        return [round(0.70 * s + 0.30 * d, 3) for s, d in zip(shape, diss)]
