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
    white_key_span, finger_span_limits, is_ascending,
    natural_finger_order, thumb_crossing_natural,
    black_key_span_correction, tendon_coupling_penalty, tempo_adjusted_max_span,
)
from fingering.phrasing.phrase import Phrase, PhraseIntent
from fingering.phrasing.pattern_library import apply_pattern_constraints
from fingering.phrasing.chord_heuristic import build_forced_constraints
from fingering.phrasing.thumb_placement_planner import apply_thumb_constraints
from fingering.phrasing.hand_position import HandPositionTracker, apply_five_finger_constraints

_hand_tracker = HandPositionTracker()  # Stateless, shared instance

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
CROSSING_THUMB_UNDER = 2.5   # Thumb-under (raised from 1.0 — prevents 1-2-1-2 spam)
THUMB_UNDER_REWARD  = -2.5   # Extra reward when thumb-under is scale-correct (stronger to compensate raise)
FINGER_OVER_REWARD  = -1.8   # Reward for correct finger-over (descending RH or ascending LH)
FINGER_OVER_PENALTY = 2.0    # Penalty for wrong-direction finger-over
OFOK_PENALTY        = 12.0   # Same finger, different pitch (legato break)
DIRECTION_REWARD    = -1.8   # Natural finger order alignment (strengthened from -1.0)
REPEATED_THUMB_PENALTY = 6.0 # Rapid thumb reuse: thumb appearing after only 1-2 intervening fingers

# ── Phase 4: OFF_BY_ONE fix ──────────────────────────────────────────────
# Bias correction: DP systematically under-estimates finger number.
# 67.8% of OFF_BY_ONE cases are GT > Pred (predicted too LOW).
# Two phenomena:
# 1. Ascending passages: finger order should INCREASE but DP sometimes keeps f=1-2
# 2. Mid-register notes: finger 1-2 on high notes is ergonomically worse than 3-4
ASCENDING_FINGER_BIAS  = 0.6  # Light penalty: ascending + finger goes DOWN unexpectedly
DESCENDING_FINGER_BIAS = 0.4  # Light penalty: descending + finger goes UP unexpectedly
REGISTER_MISMATCH_COST = 0.5  # Thumb on mid-high note in non-forced context

# ── Nguyên tắc 5: Tránh ngón 4 nếu có thể ───────────────────────────────
# Ring finger (4) là ngón yếu nhất và ít độc lập nhất — gân nó chia sẻ với
# ngón 3 và 5. Pianist thực tế tránh dùng ngón 4 ở nốt đơn lẻ khi có lựa
# chọn tốt hơn (1-2-3 hoặc 5). Penalty nhẹ để không override các trường hợp
# thực sự cần ngón 4 (scale Bb, chord rải, v.v.).
FINGER4_SOLO_PENALTY   = 0.4  # Soft penalty: ngón 4 ở vị trí không cần thiết (tuned from 0.8)

# ── Nguyên tắc 11: Ngón dài cho phím đen ─────────────────────────────────
# Phím đen nằm sâu hơn ~10mm so với phím trắng. Pianist tự nhiên dùng ngón
# dài (2, 3, 4) để nhấn phím đen vì chúng với tới đúng độ sâu đó mà không
# cần cổ tay điều chỉnh. Ngón 1 và 5 trên phím đen đã bị penalize (THUMB_ON_BLACK,
# PINKY_ON_BLACK). Complement bằng reward ngón 2/3/4:
BLACK_KEY_LONG_FINGER_REWARD = -1.2  # Reward ngón 2/3/4 trên phím đen

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

        Priority order for forced assignments:
          1. Chord heuristic (Fix 3) — chord notes must be assigned together
          2. Pattern Library (Fix 2) — scale/arpeggio patterns get standard fingering
          3. Stitch constraint       — first finger constrained by cross-phrase stitch
          4. Viterbi DP              — remaining degrees of freedom

        stitch_constraint: {'allowed_first_fingers': [1,2,3,…]}
        """
        notes = phrase.notes
        n = len(notes)
        hand  = phrase.hand
        if n == 0:
            return []
        if n == 1:
            return [self._best_single_finger(notes[0], phrase)]

        # -----------------------------------------------------------
        # Build forced constraints dict: note_idx -> fixed_finger
        # Fix 2: Pattern Library (scale/arpeggio)
        # Fix 3: Chord Heuristic (simultaneous notes)
        # Chord heuristic applied LAST so it overrides pattern library
        # for chord notes.
        # -----------------------------------------------------------
        forced: Dict[int, int] = {}
        forced = apply_pattern_constraints(notes, hand, forced)       # Fix 2
        forced = build_forced_constraints(notes, hand, forced)        # Fix 3
        forced = apply_thumb_constraints(notes, hand, forced)         # Phase 3A: thumb planner
        # Phase 3B: five-finger bypass — skip climax note so DP can pick strong finger there
        climax_skip = {phrase.climax_idx} if phrase.climax_idx is not None else set()
        forced = apply_five_finger_constraints(notes, hand, forced, skip_indices=climax_skip)

        dp   = np.full((n, N_FINGERS + 1), INF)
        prev = np.zeros((n, N_FINGERS + 1), dtype=int)

        # --- Initialise first note ---
        allowed_first = (
            stitch_constraint.get('allowed_first_fingers', list(range(1, 6)))
            if stitch_constraint else list(range(1, 6))
        )
        # Stitch constraint (cross-phrase) takes PRIORITY over pattern library
        # for the first note. Only use pattern library forced[0] when there
        # is no cross-phrase constraint restricting the first finger.
        if 0 in forced and not stitch_constraint:
            allowed_first = [forced[0]]

        for f in allowed_first:
            dp[0, f] = self._init_cost(notes[0], f, phrase)

        # --- Fill DP table ---
        for i in range(1, n):
            tension   = phrase.tension_curve[i] if phrase.tension_curve else 0.5
            is_climax = (i == phrase.climax_idx)

            # Forced constraint: only allow pinned finger at this position
            forced_f = forced.get(i, None)
            f_curr_range = [forced_f] if forced_f is not None else range(1, 6)

            for f_curr in f_curr_range:
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
        # Nguyên tắc 11: reward ngón dài trên phím đen (init note)
        if note.is_black and finger in (2, 3, 4):
            cost += BLACK_KEY_LONG_FINGER_REWARD  # negative = reward
        # On first note of phrase, prefer thumb/index for ascending phrases
        if phrase.arc_type.name in ('CLIMB', 'ARCH') and finger > 3:
            cost += 1.5

        # Nguyên tắc 5: soft penalty cho ngón 4 ở đầu phrase
        if finger == 4:
            cost += FINGER4_SOLO_PENALTY

        # ── Phase 4: Register mismatch ─────────────────────────────────────
        # A pianist naturally uses higher fingers (3-4) for mid-range melody notes,
        # reserving thumb for pivot/crossing situations.
        # MIDI 65+ (F4 and above) is "mid-high" register for RH.
        # Penalise using thumb or index when not in a scale/forced context.
        # This gently pushes the DP away from the systematic "too-low" bias.
        is_mid_high = note.pitch >= 65    # F4 and above = mid-high register
        if is_mid_high and finger == 1:
            # Thumb on F4+ = likely part of a crossing, but if not forced:
            cost += REGISTER_MISMATCH_COST
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

        # ── Phase 3A: Repeated-thumb penalty ─────────────────────────────
        # Penalize thumb appearing immediately after thumb (1→X→1 within 2 notes).
        # This prevents the DP from choosing 1-2-1-2 patterns over correct
        # 3-4-5/2-3-4 runs. Only fires when prev was also thumb and curr is thumb.
        # Exception: allowed in legato context (finger substitution involves thumb).
        if f_curr == 1 and f_prev == 1 and note_prev.pitch != note_curr.pitch:
            if intent not in (PhraseIntent.LEGATO, PhraseIntent.CANTABILE):
                cost += REPEATED_THUMB_PENALTY

        # ── Biomechanics: black key geometry correction ────────────────────
        # Black keys are physically closer → effective span is smaller
        bk_correction = black_key_span_correction(note_prev, note_curr)
        effective_span = span + bk_correction  # bk_correction is negative

        # ── Biomechanics: tempo-aware max span ────────────────────────
        bpm = getattr(note_curr, 'tempo', 120.0) or 120.0
        adjusted_max = tempo_adjusted_max_span(f_prev, f_curr, bpm)

        # ── Ergonomic: stretch ────────────────────────────────────
        # Use effective_span vs tempo-adjusted max
        over = max(0.0, effective_span - adjusted_max)
        cost += over ** 2 * STRETCH_WEIGHT

        # ── Phase 3B: Hand position shift cost ───────────────────────────
        # Penalises large hand repositioning (pianist avoids unnecessary shifts).
        pos_prev = _hand_tracker.infer(note_prev, f_prev)
        pos_curr = _hand_tracker.infer(note_curr, f_curr)
        cost += _hand_tracker.shift_cost(pos_prev, pos_curr)

        # ── Phase 4: Melodic-direction / finger-direction alignment ──────────
        # In ascending passages, increasing finger number is natural (thumb-side to pinky).
        # When RH ascends but finger number DECREASES (except thumb-under crossing),
        # it implies an awkward position or over-reliance on thumb.
        # Mirror logic applies for descending.
        is_thumb_under  = f_curr < f_prev and ascending  and note_curr.hand == 'right'
        is_finger_over  = f_curr > f_prev and not ascending and note_curr.hand == 'right'
        # Penalise anti-direction moves that are NOT thumb-under/finger-over crossings
        if ascending and (f_curr < f_prev) and not is_thumb_under:
            # Finger going down while melody goes up — adds to low-finger bias
            cost += ASCENDING_FINGER_BIAS
        if not ascending and (f_curr > f_prev) and not is_finger_over:
            cost += DESCENDING_FINGER_BIAS
        pair = (min(f_prev, f_curr), max(f_prev, f_curr))
        if pair in {(3, 4), (4, 5), (3, 5)}:
            cost += WEAK_PAIR_PENALTY

        # Nguyên tắc 5: soft penalty khi landing trên ngón 4
        # (không áp dụng khi ngón 4 là một phần của weak pair đã penalize)
        if f_curr == 4 and pair not in {(3, 4), (4, 5)}:
            cost += FINGER4_SOLO_PENALTY

        # ── Biomechanics: tendon coupling penalty ────────────────────
        # Ring finger (4) shares a tendon with middle (3).
        # At speed, alternating 3-4 or 4-5 is much harder than 1-2 or 2-3.
        cost += tendon_coupling_penalty(
            f_prev, f_curr,
            note_duration=note_curr.duration,
            bpm=bpm,
        )

        # --- Ergonomic: black key penalties + Nguyên tắc 11 (ngón dài preference) ---
        if note_curr.is_black:
            if f_curr == 1:
                cost += THUMB_ON_BLACK
            elif f_curr == 5:
                cost += PINKY_ON_BLACK
            elif f_curr in (2, 3, 4):
                # Nguyên tắc 11: ngón dài tự nhiên với phím đen
                cost += BLACK_KEY_LONG_FINGER_REWARD  # negative = reward

        # --- Ergonomic: crossing ---
        hand = note_curr.hand
        crossing = False
        if ascending and f_curr < f_prev and f_curr != 1:
            crossing = True
        if not ascending and f_curr > f_prev and f_prev != 1:
            crossing = True
        # For LH: mirror the crossing logic
        if hand == 'left':
            crossing = False
            if not ascending and f_curr < f_prev and f_curr != 1:
                crossing = True
            if ascending and f_curr > f_prev and f_prev != 1:
                crossing = True
                
        if crossing:
            # --- Thumb-Under (Luyền Ngón Luồn) ---
            # Occurs when the THUMB (f=1) crosses UNDER a higher finger.
            # Standard technique for ascending scale in RH (or descending LH).
            # More natural when the thumb will land on a WHITE key.
            is_thumb_under = (
                (hand == 'right' and ascending and f_curr == 1)
                or (hand == 'left' and not ascending and f_curr == 1)
            )
            # --- Finger-Over (Vắt Ngón) ---
            # The opposite: a higher finger (2,3,4) crosses OVER the thumb.
            # Standard for descending RH (or ascending LH).
            # Example: RH descending, f_prev=1, f_curr=3 — finger 3 vaults over thumb.
            is_finger_over = (
                (hand == 'right' and not ascending and f_prev == 1 and f_curr in (2, 3, 4))
                or (hand == 'left' and ascending and f_prev == 1 and f_curr in (2, 3, 4))
            )

            if note_curr.is_black and f_curr == 1:
                # Absolute worst case: thumb-under landing on a black key
                cost += CROSSING_STANDARD + 4.0
            elif is_thumb_under:
                cost += CROSSING_THUMB_UNDER
                # Bonus reward when this thumb-under aligns with scale motion
                # (Note lands on white key and is part of a stepwise passage)
                semitone = abs(note_curr.pitch - note_prev.pitch)
                if not note_curr.is_black and semitone <= 4:
                    cost += THUMB_UNDER_REWARD   # negative = reward!  
            elif is_finger_over:
                cost += CROSSING_THUMB_UNDER
                # Bonus reward for correct finger-over in descending/ascending context
                semitone = abs(note_curr.pitch - note_prev.pitch)
                if not note_curr.is_black and semitone <= 4:
                    cost += FINGER_OVER_REWARD   # negative = reward!
            elif thumb_crossing_natural(f_prev, f_curr, ascending, hand):
                cost += CROSSING_THUMB_UNDER  # Standard scale thumb motion
            else:
                cost += CROSSING_STANDARD       # Non-standard crossing

        # --- Ergonomic: same finger on different pitch (OFOK) ---
        if f_curr == f_prev and note_prev.pitch != note_curr.pitch:
            cost += OFOK_PENALTY
            if intent == PhraseIntent.LEGATO:
                cost += LEGATO_BREAK_PENALTY

        # --- Ergonomic: direction alignment reward ---
        if natural_finger_order(f_prev, f_curr, ascending, note_curr.hand):
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
        # Tension arc alignment — hand-aware
        # -----------------------------------------------------------
        if tension > 0.5:
            misaligned = not natural_finger_order(f_prev, f_curr, ascending, note_curr.hand)
            # Don't penalize thumb-under (it's a special move, not misalignment)
            if misaligned and not thumb_crossing_natural(f_prev, f_curr, ascending, note_curr.hand):
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
