"""
Phrase Intent Analyzer — Layer B of the Phrase-Aware Fingering module.

Determines the musical intent of each detected phrase and computes the
tension arc (climax detection), which guides phrase-scoped DP decisions.
"""

from __future__ import annotations
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


class PhraseIntentAnalyzer:
    """
    Analyzes each Phrase to assign:
      - PhraseIntent (CANTABILE / BRILLIANT / LEGATO / STACCATO / EXPRESSIVE)
      - Climax index (highest tension note)
      - Tension curve [0.0 → 1.0] per note

    Modelled after a pianist's reading of phrase character from notation.
    """

    def analyze(self, phrase: Phrase) -> Phrase:
        """
        Mutates phrase in-place to add intent, climax_idx, tension_curve.
        Returns the enriched phrase for chaining.
        """
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

        Priority order (highest specificity first):
          BRILLIANT > STACCATO > CANTABILE > LEGATO > EXPRESSIVE
        """
        notes = phrase.notes
        if not notes:
            return PhraseIntent.EXPRESSIVE

        # Note density (notes per beat)
        duration = max(phrase.duration_beats, 0.01)
        density = len(notes) / duration

        # Representative dynamic (use first note's as proxy)
        dyn = notes[0].dynamic
        dyn_rank = _dynamic_rank(dyn)

        # Check staccato
        staccato_count = sum(1 for n in notes if n.is_staccato)
        staccato_ratio = staccato_count / len(notes)

        # Check slur coverage
        slur_count = sum(1 for n in notes if n.in_slur)
        slur_ratio = slur_count / len(notes)

        # BRILLIANT: fast + loud
        if density >= _BRILLIANT_DENSITY and dyn_rank >= 6:  # f or louder
            return PhraseIntent.BRILLIANT

        # STACCATO: majority of notes are staccato
        if staccato_ratio >= 0.5:
            return PhraseIntent.STACCATO

        # CANTABILE: slurred + soft dynamic (singing quality)
        if slur_ratio >= 0.6 and dyn_rank <= 5:  # up to mf
            return PhraseIntent.CANTABILE

        # LEGATO: slurred but not specifically soft
        if slur_ratio >= 0.4:
            return PhraseIntent.LEGATO

        return PhraseIntent.EXPRESSIVE

    def _find_climax(self, phrase: Phrase) -> int:
        """
        Find the climax note index — the peak of musical tension.

        Primary criterion: highest pitch.
        Tiebreaker: accent mark, then later position (climax tends late).
        """
        notes = phrase.notes
        if not notes:
            return 0

        # Score each note for "climax-ness"
        max_pitch = max(n.pitch for n in notes)
        candidates = [
            i for i, n in enumerate(notes) if n.pitch == max_pitch
        ]

        # Prefer accented note among ties
        accented = [i for i in candidates if notes[i].has_accent]
        if accented:
            return accented[0]

        # Prefer the later occurrence (climax tends toward phrase end)
        return candidates[-1]

    def _compute_tension(self, phrase: Phrase) -> List[float]:
        """
        Compute a per-note tension curve [0.0 → 1.0].

        Shape depends on ArcType:
          ARCH  → rises to climax, then falls
          CLIMB → monotonically rises
          FALL  → monotonically falls
          WAVE  → sinusoidal approximation
          FLAT  → constant 0.5

        This curve is used by the phrase-scoped DP to:
          - Anticipate the climax (pre-climax: build tension)
          - Align finger order with melodic direction
          - Assign strong fingers near the climax
        """
        n = len(phrase.notes)
        if n == 0:
            return []
        if n == 1:
            return [1.0]

        arc = phrase.arc_type
        climax = phrase.climax_idx
        curve = []

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
                import math
                t = 0.5 + 0.5 * math.sin(2 * math.pi * i / max(n - 1, 1))

            else:  # FLAT
                t = 0.5

            curve.append(round(t, 3))

        return curve
