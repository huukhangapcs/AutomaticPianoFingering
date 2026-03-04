"""
Cross-Phrase Stitch — Layer D of the Phrase-Aware Fingering module.

Ensures natural finger continuity across phrase boundaries:
  - Computes which first fingers are viable for phrase N+1
    given the last finger used in phrase N.
  - Detects "junction crash" (ergonomically impossible transitions).
  - Optionally guides the preceding phrase's DP to choose a
    phrase-end finger that sets up the next phrase well.

Pianist insight: a skilled performer plans the END of phrase N
to make the START of phrase N+1 easier.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Set

from fingering.models.note_event import NoteEvent
from fingering.core.keyboard import (
    white_key_span, finger_span_limits, is_ascending
)
from fingering.phrasing.phrase import Phrase, ArcType

# If junction score exceeds this, it's a "crash" — hard constraint
CRASH_THRESHOLD = 15.0

# Maximum stitch cost allowed for a valid transition
STITCH_COST_THRESHOLD = 8.0


class CrossPhraseStitch:
    """
    Computes constraints linking consecutive phrases.

    Typical usage:
        stitch = CrossPhraseStitch()
        constraint = stitch.compute_constraint(phrase_a, phrase_b, fingering_a)
        # Pass constraint to PhraseScopedDP.solve(phrase_b, stitch_constraint=constraint)
    """

    def compute_constraint(
        self,
        phrase_a: Phrase,
        phrase_b: Phrase,
        fingering_a: List[int],
    ) -> Dict:
        """
        Given fingering for phrase_a, return a dict describing which
        first fingers are viable for phrase_b.

        Returns:
          {
            'allowed_first_fingers': [1, 2, 3, ...],
            'preferred_first_finger': int | None,
            'is_junction_crash': bool,
          }
        """
        if not fingering_a or not phrase_a.notes or not phrase_b.notes:
            return self._unconstrained()

        # If phrase_b starts after a rest → full freedom
        if phrase_b.starts_after_rest:
            return self._unconstrained()

        last_finger = fingering_a[-1]
        last_note   = phrase_a.notes[-1]
        first_note  = phrase_b.notes[0]

        allowed: List[int] = []
        costs:   Dict[int, float] = {}

        for f_start in range(1, 6):
            cost = self._junction_cost(last_note, last_finger,
                                       first_note, f_start)
            costs[f_start] = cost
            if cost < STITCH_COST_THRESHOLD:
                allowed.append(f_start)

        is_crash = len(allowed) == 0

        # If crash, relax to 3 best options regardless
        if is_crash:
            allowed = sorted(costs, key=costs.get)[:3]

        # Preferred = minimum cost
        preferred = min(allowed, key=lambda f: costs[f]) if allowed else None

        return {
            'allowed_first_fingers': allowed,
            'preferred_first_finger': preferred,
            'is_junction_crash': is_crash,
            'junction_costs': costs,
        }

    def preferred_end_finger(
        self,
        phrase_a: Phrase,
        phrase_b: Optional[Phrase],
    ) -> Optional[Set[int]]:
        """
        Suggest which end fingers are "good" for phrase_a
        given the upcoming phrase_b.

        Used optionally to bias the phrase_a DP *before* solving it,
        when phrase_b context is known (look-ahead).

        Returns a set of preferred finger values, or None if no preference.
        """
        if phrase_b is None:
            return None

        first_note_b = phrase_b.notes[0] if phrase_b.notes else None
        if first_note_b is None:
            return None

        last_note_a = phrase_a.notes[-1] if phrase_a.notes else None
        if last_note_a is None:
            return None

        # Determine direction of the jump between phrases
        interval = first_note_b.pitch - last_note_a.pitch
        ascending_jump = interval > 0

        # Pianist plans phrase-end finger based on direction of next phrase
        if phrase_b.starts_after_rest:
            return None  # No preference — rest allows any position
        elif abs(interval) > 12:
            # Large leap → end on thumb or index to free the hand
            return {1, 2}
        elif ascending_jump and phrase_b.arc_type in (ArcType.CLIMB, ArcType.ARCH):
            # Next phrase goes up → end low (thumb/index)
            return {1, 2}
        elif not ascending_jump and phrase_b.arc_type in (ArcType.FALL, ArcType.ARCH):
            # Next phrase goes down → end high (ring/pinky)
            return {4, 5}
        else:
            return {2, 3}  # Middle fingers are versatile

    # ------------------------------------------------------------------

    def _junction_cost(
        self,
        note_a: NoteEvent, f_a: int,
        note_b: NoteEvent, f_b: int,
    ) -> float:
        """
        Ergonomic cost of the transition FROM last note of phrase_a (f_a)
        TO first note of phrase_b (f_b).
        """
        cost = 0.0
        span = white_key_span(note_a, note_b)
        _, max_span = finger_span_limits(f_a, f_b)

        # Stretch penalty
        over = max(0, span - max_span)
        cost += over ** 2 * 2.5

        # Same finger on different pitch
        if f_a == f_b and note_a.pitch != note_b.pitch:
            cost += 10.0

        # Thumb on black key at start of phrase
        if note_b.is_black and f_b == 1:
            cost += 5.0

        return cost

    def _unconstrained(self) -> Dict:
        return {
            'allowed_first_fingers': [1, 2, 3, 4, 5],
            'preferred_first_finger': None,
            'is_junction_crash': False,
            'junction_costs': {f: 0.0 for f in range(1, 6)},
        }
