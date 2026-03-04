"""
Phrase-Scoped Viterbi DP — Layer C of the Phrase-Aware Fingering module.

Extends the standard ergonomic DP with phrase-level musical intent:
  - LEGATO: rewards finger substitution, penalizes repeated fingers
  - BRILLIANT: amplifies speed/span penalties
  - Climax note: rewards strong fingers (2, 3)
  - Tension arc: aligns finger order with melodic direction

The solver operates on a single Phrase at a time.
"""

from __future__ import annotations
import math
from typing import List, Optional, Dict

import numpy as np

from fingering.models.note_event import NoteEvent
from fingering.core.keyboard import (
    white_key_span, finger_span_limits, is_ascending
)
from fingering.phrasing.phrase import Phrase, PhraseIntent

# ---- Finger classification ----
STRONG_FINGERS = {2, 3}       # Most reliable, good touch control
MEDIUM_FINGERS = {1, 4}
WEAK_FINGERS   = {5}

# ---- Base ergonomic cost weights ----
STRETCH_WEIGHT      = 2.0    # Per white-key unit over max span
WEAK_PAIR_PENALTY   = 3.0    # Extra penalty for 3-4-5 finger combos
THUMB_ON_BLACK      = 5.0    # Thumb on black key
PINKY_ON_BLACK      = 2.5    # Pinky on black key
CROSSING_STANDARD   = 2.5    # Finger crossing (non-scale contexts)
CROSSING_THUMB_UNDER = 1.0   # Thumb under (standard in scales) — low
OFOK_PENALTY        = 12.0   # Same finger, different pitch (legato break)
DIRECTION_REWARD    = -1.0   # Natural finger order alignment

# ---- Intent modifiers ----
LEGATO_SUBST_REWARD  = -3.0   # Finger substitution in legato context
LEGATO_BREAK_PENALTY = 8.0    # Additional OFOK in legato
BRILLIANT_SPAN_MULT  = 1.8    # Amplify span penalties in fast passages
CLIMAX_WEAK_PENALTY  = 6.0    # Weak finger on climax note
CLIMAX_STRONG_REWARD = -3.0   # Strong finger on climax note
ARC_MISALIGN_PENALTY = 1.5    # Finger order fights melodic direction

LARGE_SPAN_THRESHOLD = 5      # White keys — above this = expensive

N_FINGERS = 5
INF = float('inf')


class PhraseScopedDP:
    """
    Viterbi DP solver that is aware of phrase-level musical intent.

    Usage:
        solver = PhraseScopedDP()
        fingering = solver.solve(phrase, stitch_constraint=None)
    """

    def solve(
        self,
        phrase: Phrase,
        stitch_constraint: Optional[Dict] = None,
    ) -> List[int]:
        """
        Return a list of finger assignments (1–5) for each note in phrase.

        stitch_constraint: {'allowed_first_fingers': [1,2,3,…]}
        """
        notes = phrase.notes
        n = len(notes)
        if n == 0:
            return []
        if n == 1:
            return [self._best_single_finger(notes[0], phrase)]

        dp   = np.full((n, N_FINGERS + 1), INF)
        prev = np.zeros((n, N_FINGERS + 1), dtype=int)

        # --- Initialise first note ---
        allowed = (
            stitch_constraint.get('allowed_first_fingers', list(range(1, 6)))
            if stitch_constraint else list(range(1, 6))
        )
        for f in allowed:
            dp[0, f] = self._init_cost(notes[0], f, phrase)

        # --- Fill DP table ---
        for i in range(1, n):
            tension = phrase.tension_curve[i] if phrase.tension_curve else 0.5
            is_climax = (i == phrase.climax_idx)

            for f_curr in range(1, 6):
                for f_prev in range(1, 6):
                    if dp[i - 1, f_prev] == INF:
                        continue
                    cost = self._transition_cost(
                        notes[i - 1], f_prev,
                        notes[i],    f_curr,
                        phrase.intent, tension, is_climax,
                    )
                    total = dp[i - 1, f_prev] + cost
                    if total < dp[i, f_curr]:
                        dp[i, f_curr] = total
                        prev[i, f_curr] = f_prev

        # --- Backtrack ---
        return self._backtrack(dp, prev, n)

    # ------------------------------------------------------------------
    # Cost components
    # ------------------------------------------------------------------

    def _init_cost(self, note: NoteEvent, finger: int, phrase: Phrase) -> float:
        cost = 0.0
        if note.is_black and finger == 1:
            cost += THUMB_ON_BLACK
        if note.is_black and finger == 5:
            cost += PINKY_ON_BLACK
        # On first note of phrase, prefer thumb/index for ascending phrases
        if phrase.arc_type.name in ('CLIMB', 'ARCH') and finger > 3:
            cost += 1.5
        return cost

    def _transition_cost(
        self,
        note_prev: NoteEvent, f_prev: int,
        note_curr: NoteEvent, f_curr: int,
        intent: PhraseIntent,
        tension: float,
        is_climax: bool,
    ) -> float:
        cost = 0.0
        span = white_key_span(note_prev, note_curr)
        ascending = is_ascending(note_prev, note_curr)
        _, max_span = finger_span_limits(f_prev, f_curr)

        # --- Ergonomic: stretch ---
        over = max(0, span - max_span)
        cost += over ** 2 * STRETCH_WEIGHT

        # --- Ergonomic: weak finger pair ---
        pair = (min(f_prev, f_curr), max(f_prev, f_curr))
        if pair in {(3, 4), (4, 5), (3, 5)}:
            cost += WEAK_PAIR_PENALTY

        # --- Ergonomic: black key penalties ---
        if note_curr.is_black:
            if f_curr == 1:
                cost += THUMB_ON_BLACK
            if f_curr == 5:
                cost += PINKY_ON_BLACK

        # --- Ergonomic: crossing ---
        crossing = False
        if ascending and f_curr < f_prev and f_curr != 1:
            crossing = True
        if not ascending and f_curr > f_prev and f_prev != 1:
            crossing = True
        if crossing:
            if f_curr == 1 and note_curr.is_black:
                cost += CROSSING_STANDARD + 4.0
            elif f_curr == 1:
                cost += CROSSING_THUMB_UNDER  # Normal scale thumb under
            else:
                cost += CROSSING_STANDARD

        # --- Ergonomic: same finger on different pitch (OFOK) ---
        if f_curr == f_prev and note_prev.pitch != note_curr.pitch:
            cost += OFOK_PENALTY
            if intent == PhraseIntent.LEGATO:
                cost += LEGATO_BREAK_PENALTY

        # --- Ergonomic: direction alignment reward ---
        natural_asc  = ascending  and f_curr > f_prev
        natural_desc = (not ascending) and f_curr < f_prev
        if natural_asc or natural_desc:
            cost += DIRECTION_REWARD

        # -----------------------------------------------------------
        # Intent-specific costs
        # -----------------------------------------------------------

        if intent == PhraseIntent.BRILLIANT:
            # Amplify span penalties (need compact, fast hand)
            if span > LARGE_SPAN_THRESHOLD:
                cost += (span - LARGE_SPAN_THRESHOLD) * BRILLIANT_SPAN_MULT

        if intent in (PhraseIntent.CANTABILE, PhraseIntent.LEGATO):
            # Reward finger substitution (holding note with new finger)
            if note_prev.pitch == note_curr.pitch and f_curr != f_prev:
                cost += LEGATO_SUBST_REWARD  # negative = reward

        # -----------------------------------------------------------
        # Climax note: prefer strong fingers
        # -----------------------------------------------------------
        if is_climax:
            if f_curr in WEAK_FINGERS:
                cost += CLIMAX_WEAK_PENALTY
            elif f_curr in STRONG_FINGERS:
                cost += CLIMAX_STRONG_REWARD

        # -----------------------------------------------------------
        # Tension arc alignment
        # When tension is high (approaching climax), reward natural
        # finger order that matches the melodic direction.
        # -----------------------------------------------------------
        if tension > 0.5:
            misaligned = (ascending and f_curr < f_prev and f_curr != 1)
            if misaligned:
                cost += ARC_MISALIGN_PENALTY * tension

        return max(cost, 0.0)

    def _best_single_finger(self, note: NoteEvent, phrase: Phrase) -> int:
        """Heuristic for single-note phrases."""
        if note.is_black:
            return 2  # Index finger safest on black key
        if phrase.intent == PhraseIntent.BRILLIANT:
            return 2
        return 1  # Thumb is default anchor

    def _backtrack(
        self,
        dp: np.ndarray,
        prev: np.ndarray,
        n: int,
    ) -> List[int]:
        # Find best last finger
        last_costs = dp[n - 1, 1:]  # fingers 1..5
        best_last = int(np.argmin(last_costs)) + 1

        fingers = [0] * n
        fingers[n - 1] = best_last
        for i in range(n - 2, -1, -1):
            fingers[i] = prev[i + 1, fingers[i + 1]]

        return fingers
