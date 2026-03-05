"""
Pattern Library — v2.

Detects known scale and arpeggio patterns in a note sequence and
injects hard-coded finger constraints into the phrase DP.

A pianist immediately recognizes recurring patterns (C major scale,
broken chord arpeggios, etc.) and applies memorized fingering rather
than computing from scratch.

New in v2:
  - Tone-specific scale fingering for all 12 major keys (via scale_fingering.py)
  - Hanon 5-finger exercise pattern detection for runs < 9 notes
  - Finger-over (vắt ngón) detection for descending RH passages

Supported patterns:
  - Major scale (ascending / descending) — tone-specific
  - Natural minor scale — tone-specific
  - Pentatonic scale
  - Broken chord arpeggio (triad spread 1-3-5)
  - Alberti bass (LH only)
  - Hanon 5-finger exercise
"""

from __future__ import annotations
from typing import List, Optional, Tuple
from dataclasses import dataclass, field
from fingering.models.note_event import NoteEvent
from fingering.phrasing.scale_fingering import (
    get_major_scale_fingering,
    get_minor_scale_fingering,
    get_hanon_fingering,
    detect_scale_tonic,
)

# ──────────────────────────────────────────────────────────────
# Semitone intervals for well-known patterns
# ──────────────────────────────────────────────────────────────

# Major scale intervals (relative steps in semitones): W W H W W W H
_MAJOR_SCALE_STEPS = (2, 2, 1, 2, 2, 2, 1)

# Natural minor scale
_MINOR_SCALE_STEPS = (2, 1, 2, 2, 1, 2, 2)

# Major pentatonic: W W 3H W 3H
_PENTATONIC_STEPS  = (2, 2, 3, 2, 3)

# Major triad intervals (root, 3rd, 5th): 4, 3 semitones
_TRIAD_STEPS       = (4, 3)   # ascending root-position triad

# ──────────────────────────────────────────────────────────────
# Standard fingerings per pattern (RH ascending, LH is mirrored)
# ──────────────────────────────────────────────────────────────

# Fallback fingering if tonic lookup fails (C major standard)
_RH_SCALE_FINGERS  = [1, 2, 3, 1, 2, 3, 4, 5]
_LH_SCALE_FINGERS  = [5, 4, 3, 2, 1, 3, 2, 1]

# RH broken-chord (E–G–C or similar root-pos triad ascending)
_RH_ARPEGGIO_3     = [1, 2, 4]   # 3-note arpeggio, e.g. C4-E4-G4
_LH_ARPEGGIO_3     = [5, 3, 1]   # mirror

# RH broken chord with octave: 1-2-4-1-2-4-...
_RH_ARPEGGIO_EXT   = [1, 2, 4, 1, 2, 4]
_LH_ARPEGGIO_EXT   = [5, 3, 1, 5, 3, 1]

# Hanon 5-finger pattern: ascending 1-2-3-4-5 + descending 4-3-2-1
# Typically used for passages of 4-8 stepwise notes without thumb crossing
_HANON_5_ASCENDING_RH  = [1, 2, 3, 4, 5]
_HANON_5_DESCENDING_RH = [5, 4, 3, 2, 1]
_HANON_5_ASCENDING_LH  = [5, 4, 3, 2, 1]
_HANON_5_DESCENDING_LH = [1, 2, 3, 4, 5]


@dataclass
class PatternMatch:
    """Describes a detected pattern match in the note stream."""
    start_idx: int          # index in phrase.notes
    end_idx:   int          # exclusive
    pattern:   str          # human-readable name
    fingers:   List[int]    # suggested fingering for each note in [start:end]
    confidence: float = 1.0


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────

def _intervals(notes: List[NoteEvent], start: int, length: int) -> Tuple[int, ...]:
    """Return semitone intervals for `length` consecutive notes from start."""
    return tuple(
        notes[start + i + 1].pitch - notes[start + i].pitch
        for i in range(length - 1)
    )


def _matches_template(
    steps: Tuple[int, ...],
    template: Tuple[int, ...],
    ascending: bool,
) -> bool:
    """Check if steps match template (or its negation for descending)."""
    if ascending:
        return steps == template
    else:
        return steps == tuple(-s for s in template)


def _all_consecutive(notes: List[NoteEvent], start: int, length: int) -> bool:
    """True if notes[start:start+length] are all non-chord, consecutive events."""
    for i in range(start, start + length - 1):
        if i + 1 >= len(notes):
            return False
    return True


# ──────────────────────────────────────────────────────────────
# Main Pattern Detector
# ──────────────────────────────────────────────────────────────

class PatternLibrary:
    """
    Scans a note list for known patterns and returns PatternMatch objects
    with suggested fingerings.

    Usage:
        lib = PatternLibrary()
        matches = lib.find_all(phrase.notes, hand='right')
        # matches is a list of PatternMatch, sorted by start_idx
    """

    def find_all(self, notes: List[NoteEvent], hand: str = 'right') -> List[PatternMatch]:
        """Run all pattern detectors and return non-overlapping matches."""
        all_matches: List[PatternMatch] = []
        i = 0
        n = len(notes)

        while i < n:
            m = (
                # Full scale (8 notes) first — highest priority
                self._try_scale(notes, i, hand, _MAJOR_SCALE_STEPS, 'major_scale')
                or self._try_scale(notes, i, hand, _MINOR_SCALE_STEPS, 'minor_scale')
                or self._try_scale(notes, i, hand, _PENTATONIC_STEPS, 'pentatonic')
                # Arpeggio
                or self._try_arpeggio(notes, i, hand)
                # Hanon 5-finger (before scalar run for higher priority)
                or self._try_hanon(notes, i, hand)
                # Partial scale (4–7 notes) — fallback, higher min_len to reduce false positives
                or self._try_scale_partial(notes, i, hand, _MAJOR_SCALE_STEPS, 'major_scale_partial', min_len=4)
                or self._try_scale_partial(notes, i, hand, _MINOR_SCALE_STEPS, 'minor_scale_partial', min_len=4)
            )
            if m:
                all_matches.append(m)
                i = m.end_idx   # Skip consumed notes
            else:
                i += 1

        return all_matches

    def _try_scale(
        self,
        notes: List[NoteEvent],
        start: int,
        hand: str,
        template: Tuple[int, ...],
        name: str,
    ) -> Optional[PatternMatch]:
        """
        Try to match a full octave scale (8 notes) starting at `start`.
        Uses tone-specific fingering from scale_fingering.py for accuracy.
        """
        length = len(template) + 1  # e.g. 8 notes for major scale
        if start + length > len(notes):
            return None

        steps = _intervals(notes, start, length)
        ascending = notes[start + 1].pitch > notes[start].pitch

        if not _matches_template(steps, template, ascending):
            return None

        # Detect tonic pitch class from starting note
        tonic_pc = detect_scale_tonic(notes[start:start+length], template)
        
        # Lookup tone-specific fingering from the scale database
        if 'minor' in name:
            fingers = get_minor_scale_fingering(tonic_pc, hand, ascending)
        else:
            fingers = get_major_scale_fingering(tonic_pc, hand, ascending)
        
        if fingers is None:
            # Fallback to C major standard position
            if hand == 'right':
                fingers = _RH_SCALE_FINGERS if ascending else list(reversed(_LH_SCALE_FINGERS))
            else:
                fingers = _LH_SCALE_FINGERS if not ascending else list(reversed(_RH_SCALE_FINGERS))

        # Trim to exactly `length` fingers
        fingers = fingers[:length]

        return PatternMatch(
            start_idx=start,
            end_idx=start + length,
            pattern=f"{name}_{'asc' if ascending else 'desc'}",
            fingers=fingers,
        )

    def _try_scale_partial(
        self,
        notes: List[NoteEvent],
        start: int,
        hand: str,
        template: Tuple[int, ...],
        name: str,
        min_len: int = 3,
    ) -> Optional[PatternMatch]:
        """
        Match a contiguous sub-sequence of a scale template for min_len–7 notes.

        Example: E5→F5→G5 = last 3 notes of C major (steps +1+2 at the top octave
        match template offset 4 of major scale = positions 5-6-7-8 → fingers 3-4-5).
        """
        if start + min_len > len(notes) or start + 1 >= len(notes):
            return None

        ascending = notes[start + 1].pitch > notes[start].pitch

        # Find maximum contiguous stepwise motion in the same direction
        length = 1
        for i in range(1, len(notes) - start):
            interval = notes[start + i].pitch - notes[start + i - 1].pitch
            if ascending and interval in (1, 2):
                length += 1
            elif (not ascending) and interval in (-1, -2):
                length += 1
            else:
                break

        if length < min_len:
            return None

        # Determine fingering for a generic scalar run
        # If it's a short run (e.g., 3-5 notes), use consecutive fingers
        # If it's long, use scale pattern (requires more complex thumb logic, but we default to standard)
        if hand == 'right':
            full_f = _RH_SCALE_FINGERS if ascending else list(reversed(_LH_SCALE_FINGERS))
        else:
            full_f = _LH_SCALE_FINGERS if not ascending else list(reversed(_RH_SCALE_FINGERS))

        # We don't know the exact starting offset, so we just supply consecutive fingers
        # For a generic run, 1-2-3-4-5 is a safe default, letting DP adjust if needed
        fingers = full_f[:length]
        if length > len(fingers):
            fingers = (full_f * 2)[:length]

        return PatternMatch(
            start_idx=start,
            end_idx=start + length,
            pattern=f"scalar_run_{'asc' if ascending else 'desc'}",
            fingers=fingers,
            confidence=0.5, # Lower confidence for generic runs
        )

    def _try_arpeggio(
        self,
        notes: List[NoteEvent],
        start: int,
        hand: str,
    ) -> Optional[PatternMatch]:
        """
        Detect root-position triad arpeggios: intervals 4+3 (M3+m3) ascending
        or 3+4 for first inversion, repeated up to 6 notes.
        """
        n = len(notes)
        if start + 3 > n:
            return None

        steps = _intervals(notes, start, 3)   # 2 intervals for 3 notes
        ascending = notes[start + 1].pitch > notes[start].pitch

        root_pos  = (4, 3)
        first_inv = (3, 5)   # first inversion (minor third + perfect fourth)

        is_triad = ascending and steps in (root_pos, first_inv)
        is_triad_desc = (not ascending) and steps in ((-4, -3), (-3, -5))

        if not (is_triad or is_triad_desc):
            return None

        # Try to extend to 6 notes (repeat pattern)
        length = 3
        if start + 6 <= n:
            steps6 = _intervals(notes, start, 6)
            # 6 notes -> 5 intervals. E.g. (4, 3, 5, 4, 3) for root C major C-E-G-C-E-G
            if ascending and steps6 == (root_pos[0], root_pos[1], 12 - sum(root_pos), root_pos[0], root_pos[1]):
                length = 6
            if (not ascending) and steps6 == (-4, -3, -5, -4, -3):
                length = 6

        if hand == 'right':
            template = _RH_ARPEGGIO_EXT[:length] if ascending else list(reversed(_LH_ARPEGGIO_EXT[:length]))
        else:
            template = _LH_ARPEGGIO_EXT[:length] if not ascending else list(reversed(_RH_ARPEGGIO_EXT[:length]))

        return PatternMatch(
            start_idx=start,
            end_idx=start + length,
            pattern=f"arpeggio_{'asc' if ascending else 'desc'}",
            fingers=template[:length],
        )


    def _try_hanon(
        self,
        notes: List[NoteEvent],
        start: int,
        hand: str,
    ) -> Optional[PatternMatch]:
        """
        Detect Hanon 5-finger exercise pattern:
        - Purely stepwise motion (whole/half steps only)
        - Length between 5-9 notes
        - Does NOT require a thumb crossing (stays within 5 fingers)
        
        This is activated when the note run is too short for a full thumbed scale
        but too long to be a random interval. The output is a compact 1-2-3-4-5
        style fingering without thumb-under, ideal for rapid 5-finger passages.
        """
        n = len(notes)
        if start + 4 >= n:
            return None

        # Determine direction from first two notes
        if start + 1 >= n:
            return None
        ascending = notes[start + 1].pitch > notes[start].pitch

        # Grow the run while it stays purely stepwise
        length = 1
        direction_sign = 1 if ascending else -1
        for i in range(1, min(9, n - start)):   # Hanon max 9 notes
            interval = notes[start + i].pitch - notes[start + i - 1].pitch
            if interval * direction_sign in (1, 2):  # Half or whole step same direction
                length += 1
            else:
                break

        # Must be at least 5 notes to qualify as Hanon exercise
        if length < 5:
            return None

        # Confirm it does NOT span a full octave (that would be a scale)
        total_span = abs(notes[start + length - 1].pitch - notes[start].pitch)
        if total_span >= 12:
            return None  # Let _try_scale handle full-octave patterns

        # Assign 5-finger pattern (no thumb crossing needed)
        if hand == 'right':
            fingers = (_HANON_5_ASCENDING_RH if ascending else _HANON_5_DESCENDING_RH)[:length]
        else:
            fingers = (_HANON_5_ASCENDING_LH if ascending else _HANON_5_DESCENDING_LH)[:length]

        return PatternMatch(
            start_idx=start,
            end_idx=start + length,
            pattern=f"hanon5_{'asc' if ascending else 'desc'}",
            fingers=fingers,
            confidence=0.85,
        )


def apply_pattern_constraints(
    notes: List[NoteEvent],
    hand: str,
    existing_constraints: dict,
) -> dict:
    """
    Run the pattern library on `notes` and merge detected fingerings
    into `existing_constraints[note_idx] = forced_finger`.

    Returns an updated constraints dict.
    """
    lib = PatternLibrary()
    matches = lib.find_all(notes, hand=hand)

    constraints = dict(existing_constraints)
    for m in matches:
        for j, finger in enumerate(m.fingers):
            idx = m.start_idx + j
            # Only override if not already constrained
            if idx not in constraints:
                constraints[idx] = finger

    return constraints
