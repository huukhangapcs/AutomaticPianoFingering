"""
Pattern Library — Fix 2.

Detects known scale and arpeggio patterns in a note sequence and
injects hard-coded finger constraints into the phrase DP.

A pianist immediately recognizes recurring patterns (C major scale,
broken chord arpeggios, etc.) and applies memorized fingering rather
than computing from scratch.

Supported patterns:
  - Major scale (ascending / descending)
  - Pentatonic scale
  - Broken chord arpeggio (triad spread 1-3-5)
  - Alberti bass (LH only)
"""

from __future__ import annotations
from typing import List, Optional, Tuple
from dataclasses import dataclass, field
from fingering.models.note_event import NoteEvent

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

# RH major scale: 1–2–3–1–2–3–4–5 (C position, no black keys)
_RH_SCALE_FINGERS  = [1, 2, 3, 1, 2, 3, 4, 5]
# LH major scale descending reads the same fingering for descending pitch:
_LH_SCALE_FINGERS  = [5, 4, 3, 2, 1, 3, 2, 1]

# RH broken-chord (E–G–C or similar root-pos triad ascending)
_RH_ARPEGGIO_3     = [1, 2, 4]   # 3-note arpeggio, e.g. C4-E4-G4
_LH_ARPEGGIO_3     = [5, 3, 1]   # mirror

# RH broken chord with octave: 1-2-4-1-2-4-...
_RH_ARPEGGIO_EXT   = [1, 2, 4, 1, 2, 4]
_LH_ARPEGGIO_EXT   = [5, 3, 1, 5, 3, 1]


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
        """
        length = len(template) + 1  # e.g. 8 notes for major scale
        if start + length > len(notes):
            return None

        steps = _intervals(notes, start, length)
        ascending = notes[start + 1].pitch > notes[start].pitch

        if not _matches_template(steps, template, ascending):
            return None

        # Assign fingering based on hand + direction
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
        full_template = template if ascending else tuple(-s for s in template)

        n_avail = min(len(notes) - start, len(template))  # max notes to consume

        # Try all starting offsets and lengths within the template
        for offset in range(len(template) - min_len + 1):
            for length in range(min_len, n_avail + 1):
                if offset + length - 1 > len(full_template):
                    continue
                sub = full_template[offset: offset + length - 1]
                if len(sub) == 0:
                    continue
                got = _intervals(notes, start, length)
                if len(got) != len(sub):
                    continue
                if got == sub:
                    # Match — slice the correct fingering segment
                    if hand == 'right':
                        full_f = _RH_SCALE_FINGERS if ascending else list(reversed(_LH_SCALE_FINGERS))
                    else:
                        full_f = _LH_SCALE_FINGERS if not ascending else list(reversed(_RH_SCALE_FINGERS))

                    finger_slice = full_f[offset: offset + length]
                    if len(finger_slice) < length:
                        # Extend by wrapping (cross-thumb patterns)
                        finger_slice = (full_f * 2)[offset: offset + length]

                    return PatternMatch(
                        start_idx=start,
                        end_idx=start + length,
                        pattern=f"{name}_{'asc' if ascending else 'desc'}_off{offset}",
                        fingers=finger_slice[:length],
                        confidence=0.75,
                    )
        return None

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
            if ascending and steps6 == root_pos + root_pos:
                length = 6
            if (not ascending) and steps6 == (-4, -3, -4, -3, -4, -3):
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
