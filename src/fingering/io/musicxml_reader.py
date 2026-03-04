"""
Lightweight MusicXML reader — parses pitch/duration/fingering/slur/dynamics
from a MusicXML file without requiring music21.

Uses only stdlib xml.etree.ElementTree.
"""

from __future__ import annotations
import xml.etree.ElementTree as ET
from typing import List, Optional, Tuple
from fingering.models.note_event import NoteEvent

# Step-name → semitone offset from C within octave
_STEP_TO_SEMITONE = {'C': 0, 'D': 2, 'E': 4, 'F': 5,
                     'G': 7, 'A': 9, 'B': 11}

_DYNAMIC_WORDS = ['pppp','ppp','pp','p','mp','mf','f','ff','fff','ffff']


def _step_to_midi(step: str, alter: int, octave: int) -> int:
    """Convert MusicXML pitch description → MIDI pitch number."""
    return (octave + 1) * 12 + _STEP_TO_SEMITONE[step] + alter


class MusicXMLReader:
    """
    Parses a MusicXML file and returns a list of NoteEvents.

    Handles:
      - Single-staff scores (treble clef)
      - Rests (skipped)
      - Slur start/stop/continue
      - Staccato, accent, tenuto notations
      - Fingering (existing annotations stored in note.finger as ground truth)
      - Key/time signature, tempo, divisions
      - Dynamic words (pp, mf, f, etc.)
    """

    def parse(self, path: str, hand: str = 'right') -> List[NoteEvent]:
        tree = ET.parse(path)
        root = tree.getroot()

        # Strip namespace if present
        ns = ''
        if root.tag.startswith('{'):
            ns = root.tag.split('}')[0] + '}'

        def tag(name: str) -> str:
            return f'{ns}{name}'

        notes: List[NoteEvent] = []
        divisions = 1          # MusicXML duration units per quarter note
        current_beat_time = 0.0   # Running time in quarter-note beats
        current_measure = 0
        current_dynamic = 'mf'
        open_slurs: dict[str, bool] = {}  # slur_number → currently open

        for measure_el in root.iter(tag('measure')):
            current_measure += 1
            measure_beat_start = current_beat_time
            beat_in_measure = 1.0

            for child in measure_el:
                local = child.tag.replace(ns, '')

                # --- Attributes: divisions, time sig ---
                if local == 'attributes':
                    div_el = child.find(tag('divisions'))
                    if div_el is not None:
                        divisions = int(div_el.text)

                # --- Dynamic direction words ---
                if local == 'direction':
                    for dyn_el in child.iter(tag('dynamics')):
                        for d in _DYNAMIC_WORDS:
                            if dyn_el.find(tag(d)) is not None:
                                current_dynamic = d
                                break

                # --- Note element ---
                if local == 'note':
                    # Skip rests
                    if child.find(tag('rest')) is not None:
                        dur_el = child.find(tag('duration'))
                        if dur_el is not None:
                            beats = int(dur_el.text) / divisions
                            current_beat_time += beats
                        continue

                    # Pitch
                    pitch_el = child.find(tag('pitch'))
                    if pitch_el is None:
                        continue
                    step = pitch_el.find(tag('step')).text
                    octave = int(pitch_el.find(tag('octave')).text)
                    alter_el = pitch_el.find(tag('alter'))
                    alter = int(float(alter_el.text)) if alter_el is not None else 0
                    midi_pitch = _step_to_midi(step, alter, octave)

                    # Duration in quarter-note beats
                    dur_el = child.find(tag('duration'))
                    dur_beats = int(dur_el.text) / divisions if dur_el is not None else 0.5

                    # Chord: starts at same time as previous note
                    is_chord = child.find(tag('chord')) is not None
                    note_onset = current_beat_time if not is_chord else (
                        current_beat_time - (notes[-1].duration if notes else 0)
                    )

                    # Beat within measure
                    beat_in_meas = note_onset - measure_beat_start + 1.0

                    # Notations
                    slur_start = slur_end = False
                    is_staccato = has_accent = has_tenuto = False
                    gt_finger: Optional[int] = None

                    notations_el = child.find(tag('notations'))
                    if notations_el is not None:
                        # Slurs
                        for slur_el in notations_el.iter(tag('slur')):
                            slur_type = slur_el.get('type', '')
                            slur_num = slur_el.get('number', '1')
                            if slur_type == 'start':
                                open_slurs[slur_num] = True
                                slur_start = True
                            elif slur_type == 'stop':
                                open_slurs.pop(slur_num, None)
                                slur_end = True

                        # Articulations
                        art_el = notations_el.find(tag('articulations'))
                        if art_el is not None:
                            is_staccato = art_el.find(tag('staccato')) is not None
                            has_accent  = art_el.find(tag('accent'))   is not None
                            has_tenuto  = art_el.find(tag('tenuto'))   is not None

                        # Fingering (ground truth)
                        tech_el = notations_el.find(tag('technical'))
                        if tech_el is not None:
                            f_el = tech_el.find(tag('fingering'))
                            if f_el is not None and f_el.text:
                                try:
                                    gt_finger = int(f_el.text.strip())
                                except ValueError:
                                    pass

                    in_slur = bool(open_slurs)

                    note = NoteEvent(
                        pitch=midi_pitch,
                        onset=note_onset,
                        offset=note_onset + dur_beats,
                        hand=hand,
                        measure=current_measure,
                        beat=round(beat_in_meas, 3),
                        in_slur=in_slur,
                        slur_start=slur_start,
                        slur_end=slur_end,
                        is_staccato=is_staccato,
                        has_accent=has_accent,
                        has_tenuto=has_tenuto,
                        dynamic=current_dynamic,
                        finger=gt_finger,   # ground truth stored here
                    )
                    notes.append(note)

                    if not is_chord:
                        current_beat_time += dur_beats

        return notes
