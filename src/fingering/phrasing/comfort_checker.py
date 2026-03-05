"""
Comfort Checker — Post-DP ergonomic validation.

After PhraseScopedDP produces a fingering, ComfortChecker evaluates
whether the sequence is physically comfortable for a human pianist.
If the difficulty score exceeds HARD_THRESHOLD, the pipeline re-solves
the phrase in STRICT mode with stronger ergonomic penalties.

Design:
  difficulty_per_transition() — per-transition score using span_cost + pair
  phrase_difficulty()         — overall score (mean of top-K transitions)
  is_too_hard()               — threshold check
  hardest_indices()           — which transitions are worst (for logging)

Difficulty is scored in [0, ∞):
  0.0  — completely comfortable (in-position, good pair)
  1.0  — minor tension (zone 2 stretch or weak pair once)
  3.0  — quite hard (zone 3 stretch or several weak pairs)
  8.0+ — very difficult / borderline impossible
"""

from __future__ import annotations
from typing import List, Tuple

from fingering.models.note_event import NoteEvent
from fingering.core.keyboard import (
    physical_span_mm, span_cost,
    _WHITE_KEY_WIDTH_MM as _WK,
    physical_key_position_mm,
)

# ──────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────

# Pair difficulty add-on (on top of span_cost) for tendon-coupled pairs
_PAIR_DIFFICULTY: dict[tuple, float] = {
    (3, 4): 1.5,   # Middle–Ring: shared tendon, very hard at speed
    (4, 5): 1.5,   # Ring–Pinky
    (3, 5): 1.0,   # Middle–Pinky: large skip
}

# Mean of top-K hardest transitions that triggers re-solve
HARD_THRESHOLD = 6.0   # only fire for genuinely extreme difficulty

# How many hardest transitions to average (top-K)
_TOP_K = 3

# ──────────────────────────────────────────────────────────────────
# Core functions
# ──────────────────────────────────────────────────────────────────

def transition_difficulty(
    note_prev: NoteEvent, f_prev: int,
    note_curr: NoteEvent, f_curr: int,
) -> float:
    """
    Difficulty score for a single (prev → curr) finger transition.

    Components:
      1. span_cost      — 3-zone physical stretch cost
      2. pair penalty   — tendon-coupled pair surcharge
      3. in_position    — subtract 1.0 if notes are in same hand position
    """
    span_mm = physical_span_mm(note_prev, note_curr)

    # Span zone cost
    sc = span_cost(span_mm, f_prev, f_curr)

    # Weak pair surcharge
    pair = (min(f_prev, f_curr), max(f_prev, f_curr))
    pair_extra = _PAIR_DIFFICULTY.get(pair, 0.0)

    # In-position bonus (same thumb anchor = 0 hand movement)
    hand = note_curr.hand
    off_prev = (f_prev - 1) * _WK
    off_curr = (f_curr - 1) * _WK
    thumb_prev = (physical_key_position_mm(note_prev.pitch) - off_prev
                  if hand == 'right'
                  else physical_key_position_mm(note_prev.pitch) + off_prev)
    thumb_curr = (physical_key_position_mm(note_curr.pitch) - off_curr
                  if hand == 'right'
                  else physical_key_position_mm(note_curr.pitch) + off_curr)
    in_pos_bonus = 1.0 if abs(thumb_prev - thumb_curr) < 12.0 else 0.0

    return max(0.0, sc + pair_extra - in_pos_bonus)


def phrase_difficulty(
    notes: List[NoteEvent],
    fingering: List[int],
) -> float:
    """
    Overall difficulty score for a phrase.

    Returns the average of the top-K hardest transition scores.
    Higher = harder.  0.0 = trivially easy.
    """
    if len(notes) < 2:
        return 0.0

    scores: List[float] = []
    for i in range(1, len(notes)):
        d = transition_difficulty(notes[i - 1], fingering[i - 1],
                                  notes[i],     fingering[i])
        scores.append(d)

    if not scores:
        return 0.0

    scores.sort(reverse=True)
    top_k = scores[:_TOP_K]
    return sum(top_k) / len(top_k)


def is_too_hard(
    notes: List[NoteEvent],
    fingering: List[int],
    threshold: float = HARD_THRESHOLD,
) -> bool:
    """Return True if the phrase difficulty exceeds threshold."""
    return phrase_difficulty(notes, fingering) > threshold


def hardest_indices(
    notes: List[NoteEvent],
    fingering: List[int],
    top_k: int = 3,
) -> List[Tuple[int, float]]:
    """
    Return (index, score) of the top_k hardest transitions.
    Index i means transition from notes[i-1]→notes[i].
    """
    scored = []
    for i in range(1, len(notes)):
        d = transition_difficulty(notes[i - 1], fingering[i - 1],
                                  notes[i],     fingering[i])
        scored.append((i, d))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]
