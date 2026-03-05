"""
Layer 0: Score Scanner — ScoreProfile

Pre-reads the entire note stream to extract global structure:
  - Key / mode
  - Texture type (melody+bass, chordal, two-voice)
  - Metric regularity

This mirrors how a pianist first scans a piece before playing.
The texture determines which segmentation strategy to apply downstream.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple
from fingering.models.note_event import NoteEvent


@dataclass
class ScoreProfile:
    """Global score properties extracted from the full note stream."""
    # Pitch statistics
    rh_pitch_range: Tuple[int, int] = (60, 84)   # (min, max) MIDI
    lh_pitch_range: Tuple[int, int] = (36, 60)

    # Rhythmic texture
    texture: str = 'melody+bass'   # see TEXTURE_* constants below
    avg_rh_notes_per_measure: float = 4.0
    avg_lh_notes_per_measure: float = 1.0

    # Has explicit markings?
    has_slurs: bool = False
    has_dynamics: bool = False

    # Metric
    total_measures: int = 0

    # Estimated key (most common pitch class in RH)
    tonic_pc: int = 0    # 0=C, 1=C#, ..., 11=B

    # Texture constants (set as class attrs for external use)
    TEXTURE_MELODY_BASS    = 'melody+bass'    # RH single line, LH chords
    TEXTURE_TWO_VOICE      = 'two_voice'      # Both hands melodic
    TEXTURE_CHORDAL        = 'chordal'        # Both hands chords
    TEXTURE_SINGLE_STAFF   = 'single_staff'   # Only one hand present


# ──────────────────────────────────────────────────────
# Builder
# ──────────────────────────────────────────────────────

def build_score_profile(
    rh: List[NoteEvent],
    lh: List[NoteEvent],
    tonic_pc_override: Optional[int] = None,  # Fix 1: from MusicXML key sig
) -> ScoreProfile:
    """
    Analyse the full note streams and return a ScoreProfile.
    Called once at the start of the pipeline.
    """
    all_notes = rh + lh
    if not all_notes:
        return ScoreProfile()

    total_measures = max(n.measure for n in all_notes)

    # Pitch ranges
    rh_pitches = [n.pitch for n in rh] if rh else [60]
    lh_pitches = [n.pitch for n in lh] if lh else [48]
    rh_range = (min(rh_pitches), max(rh_pitches))
    lh_range = (min(lh_pitches), max(lh_pitches))

    # Explicit markings
    has_slurs    = any(n.slur_start or n.slur_end or n.in_slur for n in all_notes)
    has_dynamics = len({n.dynamic for n in all_notes if n.dynamic}) > 1

    # Notes per measure averages
    def notes_per_measure(notes: List[NoteEvent]) -> float:
        if not notes:
            return 0.0
        per_m: dict[int, int] = {}
        for n in notes:
            per_m[n.measure] = per_m.get(n.measure, 0) + 1
        return sum(per_m.values()) / max(1, len(per_m))

    avg_rh = notes_per_measure(rh)
    avg_lh = notes_per_measure(lh)

    # Texture classification
    if not lh:
        texture = ScoreProfile.TEXTURE_SINGLE_STAFF
    elif avg_lh >= 3.0 and avg_rh >= 3.0:
        # Both hands dense → check if LH is melodic (widely spaced intervals)
        lh_intervals = _melodic_variety(lh)
        if lh_intervals > 3.0:
            texture = ScoreProfile.TEXTURE_TWO_VOICE
        else:
            texture = ScoreProfile.TEXTURE_CHORDAL
    else:
        texture = ScoreProfile.TEXTURE_MELODY_BASS

    # Fix 1: Use MusicXML key signature if provided — much more reliable
    if tonic_pc_override is not None:
        tonic_pc = tonic_pc_override
    else:
        # Fallback: use LH bass notes
        tonic_source = lh if lh else rh
        if tonic_source:
            by_measure: dict[int, list] = {}
            for n in tonic_source:
                by_measure.setdefault(n.measure, []).append(n)
            bass_pcs: dict[int, int] = {}
            for m_notes in by_measure.values():
                bass_note = min(m_notes, key=lambda n: n.pitch)
                pc = bass_note.pitch % 12
                bass_pcs[pc] = bass_pcs.get(pc, 0) + 1
            tonic_pc = max(bass_pcs, key=bass_pcs.get)
        else:
            tonic_pc = 0

    return ScoreProfile(
        rh_pitch_range=rh_range,
        lh_pitch_range=lh_range,
        texture=texture,
        avg_rh_notes_per_measure=avg_rh,
        avg_lh_notes_per_measure=avg_lh,
        has_slurs=has_slurs,
        has_dynamics=has_dynamics,
        total_measures=total_measures,
        tonic_pc=tonic_pc,
    )


def _melodic_variety(notes: List[NoteEvent], n_sample: int = 50) -> float:
    """
    Return the average absolute interval between consecutive notes.
    High variety (>3 semitones avg) → melodic line.
    Low variety (<2 semitones avg) → chord block.
    """
    sample = notes[:n_sample]
    if len(sample) < 2:
        return 0.0
    intervals = [abs(sample[i+1].pitch - sample[i].pitch)
                 for i in range(len(sample) - 1)]
    return sum(intervals) / len(intervals)
