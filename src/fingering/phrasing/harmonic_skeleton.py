"""
Layer 2: Harmonic Skeleton

Estimates the chord root per measure and detects cadential motion.
A pianist uses harmonic rhythm (how often chords change) and cadence
type (V→I, IV→I) to locate phrase endings.

This does NOT require explicit harmonic annotation — it infers
harmony from the bass (lowest notes) in each measure, cross-checked
against the RH pitch content.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from fingering.models.note_event import NoteEvent


# ──────────────────────────────────────────────────────────────
# Data Structures
# ──────────────────────────────────────────────────────────────

@dataclass
class MeasureHarmony:
    """Harmonic snapshot for a single measure."""
    measure: int
    bass_pc: int            # Pitch class of bass note (0–11)
    rh_pitch_classes: set   # Set of pitch classes in RH
    root_pc: int            # Estimated chord root
    chord_type: str         # 'major' | 'minor' | 'dom7' | 'dim' | 'unknown'


@dataclass
class Cadence:
    """A detected harmonic cadence between two consecutive measures."""
    measure: int            # The measure WHERE cadence resolves
    type: str               # 'authentic' | 'half' | 'plagal' | 'deceptive'
    strength: float         # 0–1


@dataclass
class HarmonicSkeleton:
    """Complete harmonic map of the piece."""
    measure_harmonies: List[MeasureHarmony]
    cadences: List[Cadence]
    harmonic_rhythm: float  # avg measures between chord changes

    def cadence_at(self, measure: int, tolerance: int = 1) -> Optional[Cadence]:
        """Find cadence near a given measure (within tolerance)."""
        for c in self.cadences:
            if abs(c.measure - measure) <= tolerance:
                return c
        return None

    def chord_changes(self) -> List[int]:
        """Returns measures where the chord root changes."""
        changes = []
        prev = None
        for mh in self.measure_harmonies:
            if prev is not None and mh.root_pc != prev:
                changes.append(mh.measure)
            prev = mh.root_pc
        return changes


# ──────────────────────────────────────────────────────────────
# Chord Type Templates (pitch class sets)
# ──────────────────────────────────────────────────────────────

# Relative pitch classes for common chord types (from root = 0)
_CHORD_TEMPLATES = {
    'major': frozenset([0, 4, 7]),
    'minor': frozenset([0, 3, 7]),
    'dom7':  frozenset([0, 4, 7, 10]),
    'dim':   frozenset([0, 3, 6]),
    'maj7':  frozenset([0, 4, 7, 11]),
    'min7':  frozenset([0, 3, 7, 10]),
}


def _match_chord_type(pitch_classes: set, root_pc: int) -> str:
    """Match pitch classes against templates to determine chord type."""
    if not pitch_classes:
        return 'unknown'
    relative = frozenset((pc - root_pc) % 12 for pc in pitch_classes)
    best, best_overlap = 'unknown', 0
    for ctype, template in _CHORD_TEMPLATES.items():
        overlap = len(relative & template)
        if overlap > best_overlap:
            best, best_overlap = ctype, overlap
    return best if best_overlap >= 2 else 'unknown'


# ──────────────────────────────────────────────────────────────
# Cadence Detection Rules
# ──────────────────────────────────────────────────────────────

# Bass root movement (in semitones mod 12) → (cadence type, strength)
# Perfect 4th up (5 semitones) = V→I (most common authentic cadence)
_CADENCE_RULES: Dict[int, Tuple[str, float]] = {
    5:  ('authentic', 0.90),   # P4 up  = V→I (dominant → tonic)
    -7: ('authentic', 0.90),   # P5 down = same interval, enharmonic
    7:  ('authentic', 0.75),   # P5 up  = IV→I (plagal variant)
    -5: ('plagal',   0.65),    # P4 down = plagal cadence IV→I
    -4: ('half',     0.50),    # M3 down = I→V (half cadence ending)
    4:  ('deceptive', 0.40),   # M3 up  = deceptive motion
}


# ──────────────────────────────────────────────────────────────
# Builder
# ──────────────────────────────────────────────────────────────

def build_harmonic_skeleton(
    rh: List[NoteEvent],
    lh: List[NoteEvent],
) -> HarmonicSkeleton:
    """
    Build a HarmonicSkeleton from RH + LH note streams.
    Uses LH bass notes as primary harmonic indicator.
    """
    all_notes = rh + lh
    if not all_notes:
        return HarmonicSkeleton([], [], 0.0)

    total_measures = max(n.measure for n in all_notes)

    # Group notes by measure
    rh_by_m: Dict[int, List[NoteEvent]] = {}
    lh_by_m: Dict[int, List[NoteEvent]] = {}
    for n in rh:
        rh_by_m.setdefault(n.measure, []).append(n)
    for n in lh:
        lh_by_m.setdefault(n.measure, []).append(n)

    # Build per-measure harmony
    harmonies: List[MeasureHarmony] = []
    for m in range(1, total_measures + 1):
        rh_notes = rh_by_m.get(m, [])
        lh_notes = lh_by_m.get(m, [])

        # Bass = lowest pitch in LH (or RH if no LH)
        bass_source = lh_notes or rh_notes
        if not bass_source:
            continue
        bass_note = min(bass_source, key=lambda n: n.pitch)
        bass_pc = bass_note.pitch % 12

        # RH pitch classes
        rh_pcs = {n.pitch % 12 for n in rh_notes}

        # Root estimation: bass note is primary; check if it forms a chord
        # with RH content. If not, use most common RH pitch class.
        root_pc = bass_pc
        chord_type = _match_chord_type(rh_pcs | {bass_pc}, root_pc)

        harmonies.append(MeasureHarmony(
            measure=m,
            bass_pc=bass_pc,
            rh_pitch_classes=rh_pcs,
            root_pc=root_pc,
            chord_type=chord_type,
        ))

    # Detect cadences from consecutive chord root changes
    cadences: List[Cadence] = []
    for i in range(1, len(harmonies)):
        prev = harmonies[i - 1]
        curr = harmonies[i]
        interval = (curr.root_pc - prev.root_pc) % 12
        # Normalize to -6..+6 range
        if interval > 6:
            interval -= 12

        rule = _CADENCE_RULES.get(interval)
        if rule:
            ctype, strength = rule
            # Bonus if current chord is major (tonic arrival)
            if curr.chord_type == 'major':
                strength = min(1.0, strength + 0.10)
            cadences.append(Cadence(
                measure=curr.measure,
                type=ctype,
                strength=strength,
            ))

    # Compute harmonic rhythm: avg measures between chord changes
    prev_root = None
    changes = 0
    for mh in harmonies:
        if prev_root is not None and mh.root_pc != prev_root:
            changes += 1
        prev_root = mh.root_pc
    harmonic_rhythm = len(harmonies) / max(1, changes) if changes > 0 else float(len(harmonies))

    return HarmonicSkeleton(
        measure_harmonies=harmonies,
        cadences=cadences,
        harmonic_rhythm=harmonic_rhythm,
    )
