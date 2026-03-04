"""
Chord Fingering Heuristic — Fix 3.

When multiple notes sound simultaneously (chords), strict Viterbi DP
cannot model them: all notes in a chord must be assigned to *different*
fingers simultaneously. This module applies rule-based chord fingering
before the DP runs.

Rules (pianist conventions):
  Right Hand chord (lowest→highest pitch):
    2 notes  → [thumb, index] i.e. [1, 2] or stretch [1, 3]
    3 notes  → [1, 2, 3] or [1, 3, 5] depending on span
    4 notes  → [1, 2, 3, 5]
    5 notes  → [1, 2, 3, 4, 5]

  Left Hand chord (lowest→highest pitch):
    2 notes  → [5, 1]   (pinky on bottom, thumb on top)
    3 notes  → [5, 3, 1]
    4 notes  → [5, 4, 2, 1]  or [5, 3, 2, 1]
    5 notes  → [5, 4, 3, 2, 1]

  Adjustments:
    - If span > 10 semitones, use wider fingers (skip weak fingers like 2/4)
    - Black key note → avoid thumb where possible
"""

from __future__ import annotations
from typing import List, Dict, Tuple
from fingering.models.note_event import NoteEvent

# Standard chord fingerings indexed by (hand, n_notes)
# Notes sorted lowest→highest pitch
_CHORD_FINGERINGS: Dict[Tuple[str, int], List[int]] = {
    ('right', 1): [1],
    ('right', 2): [1, 2],
    ('right', 3): [1, 3, 5],
    ('right', 4): [1, 2, 4, 5],
    ('right', 5): [1, 2, 3, 4, 5],

    ('left',  1): [1],
    ('left',  2): [5, 1],
    ('left',  3): [5, 3, 1],
    ('left',  4): [5, 3, 2, 1],
    ('left',  5): [5, 4, 3, 2, 1],
}


def detect_chords(notes: List[NoteEvent], tolerance: float = 0.05) -> List[List[int]]:
    """
    Group notes into simultaneous chords.

    Returns a list of groups — each group is a list of indices into `notes`
    that sound at the same time (onset within `tolerance` beats).

    Single notes are returned as groups of size 1.
    """
    groups: List[List[int]] = []
    used = [False] * len(notes)

    for i, note in enumerate(notes):
        if used[i]:
            continue
        group = [i]
        used[i] = True
        for j in range(i + 1, len(notes)):
            if not used[j] and abs(notes[j].onset - note.onset) <= tolerance:
                group.append(j)
                used[j] = True
        groups.append(group)

    return groups


def assign_chord_fingers(
    notes: List[NoteEvent],
    hand: str,
) -> Dict[int, int]:
    """
    Assign fingers to all chord notes in `notes`.

    Returns a dict mapping note_index → finger (1–5).
    Single-note "chords" (groups of size 1) are left unassigned (caller's DP handles them).
    """
    groups = detect_chords(notes)
    assignments: Dict[int, int] = {}

    for group in groups:
        if len(group) == 1:
            continue   # Let DP handle single notes

        # Sort group by pitch
        group_sorted = sorted(group, key=lambda idx: notes[idx].pitch)
        n = min(len(group_sorted), 5)
        template = list(_CHORD_FINGERINGS.get((hand, n), range(1, n + 1)))

        # Adjust for black keys: avoid thumb on black key
        # For RH: if lowest note is black, shift up (1→2 on lowest)
        if hand == 'right':
            if notes[group_sorted[0]].is_black and template[0] == 1 and len(template) > 1:
                template = [2] + template[1:]
        else:
            if notes[group_sorted[-1]].is_black and template[-1] == 1 and len(template) > 1:
                template = template[:-1] + [2]

        for rank, note_idx in enumerate(group_sorted):
            if rank < len(template):
                assignments[note_idx] = template[rank]

    return assignments


def build_forced_constraints(
    notes: List[NoteEvent],
    hand: str,
    existing: Dict[int, int] | None = None,
) -> Dict[int, int]:
    """
    Merge chord heuristic results into an existing constraints dict.

    For indices in chords, the chord heuristic takes priority (overrides).
    For single-note positions, keep existing constraints.
    """
    result = dict(existing or {})
    chord_assignments = assign_chord_fingers(notes, hand)

    # Chord heuristic overrides pattern library for chord notes
    result.update(chord_assignments)
    return result
