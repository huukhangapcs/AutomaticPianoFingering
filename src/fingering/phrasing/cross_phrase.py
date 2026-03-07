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
from fingering.phrasing.phrase import Phrase, ArcType, HandResetType
from fingering.phrasing.hand_position import HandPositionTracker, _FINGER_OFFSET_MM

# If junction score exceeds this, it's a "crash" — hard constraint
CRASH_THRESHOLD = 15.0

# Maximum stitch cost allowed for a valid transition
STITCH_COST_THRESHOLD = 8.0

# ── Lazy Reset Constants ────────────────────────────────────────────
# If the cheapest continuation cost is within this, skip reset entirely.
LAZY_CONTINUE_THRESHOLD = 6.0

# Only probe lazy continue when rest is not too long.
# Beyond this, the hand has been idle long enough that full reset is appropriate.
LAZY_PROBE_MAX_REST_SEC = 0.60   # ~3 eighth notes at 120 BPM

_tracker = HandPositionTracker()


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
        rest_sec: float = 0.0,
    ) -> Dict:
        """
        Given fingering for phrase_a, return a dict describing which
        first fingers are viable for phrase_b.

        Args:
            rest_sec: Duration of rest/gap between phrases in seconds.
                      Used to decide whether to attempt lazy continuation.

        Returns:
          {
            'allowed_first_fingers': [1, 2, 3, ...],
            'preferred_first_finger': int | None,
            'is_junction_crash': bool,
            'lazy_continued': bool,   # NEW: True if we skipped reset
          }
        """
        if not fingering_a or not phrase_a.notes or not phrase_b.notes:
            return self._unconstrained()

        # ── FULL reset check FIRST ─────────────────────────────────────────
        # FULL = long rest → hand is fully free. Lazy probe must NOT override this.
        if phrase_b.reset_type == HandResetType.FULL:
            return self._unconstrained()

        # ── Lazy Probe: Try-Continue-First (short non-zero rest only) ─────────
        # rest_sec=0 means legato → skip to normal NONE path.
        if 0.0 < rest_sec < LAZY_PROBE_MAX_REST_SEC:
            lazy_result = self._try_lazy_continue(phrase_a, phrase_b, fingering_a)
            if lazy_result is not None:
                return lazy_result

        # ── SOFT dispatch ────────────────────────────────────────────────────
        if phrase_b.reset_type == HandResetType.SOFT:
            return self._soft_constrained(phrase_a, phrase_b, fingering_a)

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

        if is_crash:
            allowed = sorted(costs, key=costs.get)[:3]

        lookahead_costs: dict[int, float] = {}
        for f_start in allowed:
            la = self._lookahead_cost(phrase_b, f_start, n_notes=3)
            lookahead_costs[f_start] = costs[f_start] + la

        preferred = min(allowed, key=lambda f: lookahead_costs[f]) if allowed else None

        return {
            'allowed_first_fingers': allowed,
            'preferred_first_finger': preferred,
            'is_junction_crash': is_crash,
            'junction_costs': costs,
            'lazy_continued': False,
        }

    def _try_lazy_continue(
        self,
        phrase_a: Phrase,
        phrase_b: Phrase,
        fingering_a: List[int],
    ) -> Optional[Dict]:
        """
        Lazy Reset Probe: can we skip the reset and continue naturally?

        Strategy:
          1. Infer the hand state from the last note of phrase_a.
          2. For each possible first finger of phrase_b, compute junction cost
             using HandState mm-based shift cost (anatomically accurate).
          3. If the minimum junction cost ≤ LAZY_CONTINUE_THRESHOLD, the hand
             can flow naturally into phrase_b without repositioning.
             Return a tight constraint favouring those cheap continuations.
          4. If stuck (all costs > threshold), return None → caller falls
             back to FULL/SOFT reset logic.

        Pianist analogy: after an 8th-note rest, a pianist doesn't necessarily
        lift and replant the hand — if the next note is close to the current
        hand position, they simply reach for it. Only when it's far away do
        they physically reposition.
        """
        if not fingering_a or not phrase_a.notes or not phrase_b.notes:
            return None

        last_f     = fingering_a[-1]
        last_note  = phrase_a.notes[-1]
        first_note = phrase_b.notes[0]

        # Infer current hand state from last note + last finger
        prev_state = _tracker.infer(last_note, last_f)

        costs: Dict[int, float] = {}
        for f_start in range(1, 6):
            # Physical cost: how far does thumb need to travel?
            next_state = _tracker.infer(first_note, f_start)
            shift = _tracker.shift_cost(prev_state, next_state,
                                        f_prev=last_f, f_curr=f_start)

            # Black-key penalty on first note with thumb
            bk = 3.0 if (first_note.is_black and f_start == 1) else 0.0

            # OFOK: same finger on different pitch
            ofok = 8.0 if (f_start == last_f and first_note.pitch != last_note.pitch) else 0.0

            costs[f_start] = shift + bk + ofok

        min_cost = min(costs.values())

        if min_cost > LAZY_CONTINUE_THRESHOLD:
            # Too expensive to continue — let normal reset logic handle it
            return None

        # Continuation is cheap — prefer fingers within the threshold
        allowed = [f for f, c in costs.items() if c <= LAZY_CONTINUE_THRESHOLD]

        # Lookahead score to pick the best among cheap continuations
        la_costs = {f: costs[f] + self._lookahead_cost(phrase_b, f, n_notes=3)
                    for f in allowed}
        preferred = min(allowed, key=la_costs.get) if allowed else None

        return {
            'allowed_first_fingers': allowed,
            'preferred_first_finger': preferred,
            'is_junction_crash': False,
            'junction_costs': costs,
            'lazy_continued': True,   # Flag: no physical reset was performed
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

    def _lookahead_cost(self, phrase_b: Phrase, first_finger: int, n_notes: int = 3) -> float:
        """
        Estimate the ergonomic cost of starting phrase_b with `first_finger`
        by simulating a simple greedy run over the first `n_notes` notes.

        Pianist insight: the first finger of a phrase determines hand position
        for the next few notes. A bad start compounds across the phrase.

        Returns: cumulative ergonomic cost (lower = better setup).
        """
        notes = phrase_b.notes
        if not notes or n_notes < 2:
            return 0.0

        look = notes[:min(n_notes, len(notes))]
        cost = 0.0
        f_prev = first_finger
        _, max_span_init = finger_span_limits(first_finger, first_finger)

        # Black key penalty on the very first note
        if look[0].is_black and first_finger == 1:
            cost += 5.0

        for k in range(1, len(look)):
            span = white_key_span(look[k - 1], look[k])
            ascending = is_ascending(look[k - 1], look[k])

            # Greedy: pick the best finger for look[k] given f_prev
            best_f, best_c = f_prev, float('inf')
            for f_curr in range(1, 6):
                _, max_sp = finger_span_limits(f_prev, f_curr)
                over = max(0.0, span - max_sp)  # stretch penalty
                c = over ** 2 * 2.0
                # Prefer natural finger direction
                if ascending and f_curr < f_prev and f_curr != 1:
                    c += 2.0
                if not ascending and f_curr > f_prev and f_prev != 1:
                    c += 2.0
                if f_curr < best_c or c < best_c:
                    best_f, best_c = f_curr, c
            cost += best_c
            f_prev = best_f

        return cost

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

    def _soft_constrained(
        self,
        phrase_a: Phrase,
        phrase_b: Phrase,
        fingering_a: List[int],
    ) -> Dict:
        """
        Soft reset: stitch threshold doubled — more fingers are allowed,
        but ergonomically extreme transitions are still penalized.

        Pianist analogy: a short rest lets you reposition, but not wildly—
        e.g. an 8th-note rest gives time to move, but not a full arm-swing.
        """
        last_finger = fingering_a[-1]
        last_note   = phrase_a.notes[-1]
        first_note  = phrase_b.notes[0]

        soft_threshold = STITCH_COST_THRESHOLD * 2.0   # Relaxed

        allowed: List[int] = []
        costs:   Dict[int, float] = {}

        for f_start in range(1, 6):
            cost = self._junction_cost(last_note, last_finger,
                                       first_note, f_start)
            costs[f_start] = cost
            if cost < soft_threshold:
                allowed.append(f_start)

        # At minimum allow the 3 cheapest options
        if len(allowed) < 3:
            allowed = sorted(costs, key=costs.get)[:3]

        preferred = min(allowed, key=lambda f: costs[f]) if allowed else None

        return {
            'allowed_first_fingers': allowed,
            'preferred_first_finger': preferred,
            'is_junction_crash': False,   # Soft reset never counts as crash
            'junction_costs': costs,
            'lazy_continued': False,
        }

    def _unconstrained(self) -> Dict:
        return {
            'allowed_first_fingers': [1, 2, 3, 4, 5],
            'preferred_first_finger': None,
            'is_junction_crash': False,
            'junction_costs': {f: 0.0 for f in range(1, 6)},
            'lazy_continued': False,
        }
