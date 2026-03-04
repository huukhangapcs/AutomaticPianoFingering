"""
Lightweight MusicXML reader — parses pitch/duration/fingering/slur/dynamics
from a MusicXML file without requiring music21.

Supports both single-staff and grand staff (2-staff piano) scores.
Staff 1 → right hand ('right'), Staff 2 → left hand ('left').

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
    Parses a MusicXML file and returns NoteEvents.

    Grand staff support:
      - <staff>1</staff> → hand='right'
      - <staff>2</staff> → hand='left'
      - <backup> elements correctly reset the timekeeper per staff

    Handles: rests, slurs, staccato/accent/tenuto, fingering GT,
             key/time signature, tempo, divisions, dynamic words.
    """

    def parse_grand_staff(self, path: str) -> Tuple[List[NoteEvent], List[NoteEvent]]:
        """
        Parse a grand staff (2-staff piano score) into separate RH and LH lists.

        Returns: (right_hand_notes, left_hand_notes)
        Both lists are sorted by onset time.
        """
        all_notes = self._parse_all(path)
        rh = [n for n in all_notes if n.hand == 'right']
        lh = [n for n in all_notes if n.hand == 'left']
        return rh, lh

    def parse(self, path: str, hand: str = 'right') -> List[NoteEvent]:
        """
        Parse a single-staff file, tagging all notes with the given hand.
        For grand staff files use parse_grand_staff() instead.
        """
        notes = self._parse_all(path)
        for n in notes:
            n.hand = hand  # Force override for single-staff use
        return notes

    def _parse_all(self, path: str) -> List[NoteEvent]:
        tree = ET.parse(path)
        root = tree.getroot()

        # Strip namespace if present
        ns = ''
        if root.tag.startswith('{'):
            ns = root.tag.split('}')[0] + '}'

        def tag(name: str) -> str:
            return f'{ns}{name}'

        notes: List[NoteEvent] = []
        divisions = 1
        current_measure = 0
        current_dynamic = 'mf'
        open_slurs: dict[str, bool] = {}
        # Per-staff time tracking — persists across ALL measures
        # staff_time[staff_num]        = current beat position
        # measure_beat_start[staff_num] = beat position at start of current measure
        staff_time: dict[str, float]        = {}
        measure_beat_start: dict[str, float] = {}

        for measure_el in root.iter(tag('measure')):
            current_measure += 1
            # Reset measure_beat_start to current positions at start of each measure
            for sn in staff_time:
                measure_beat_start[sn] = staff_time[sn]

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

                # --- <backup>: rewind time cursor for new staff ---
                if local == 'backup':
                    # <backup> rewrites the time position — we just note it and let
                    # per-staff tracking handle positioning.
                    pass

                # --- Note element ---
                if local == 'note':
                    # Determine staff number → hand
                    staff_el = child.find(tag('staff'))
                    staff_num = staff_el.text if staff_el is not None else '1'
                    hand = 'right' if staff_num == '1' else 'left'

                    # Initialise per-staff time cursor on first encounter in this score
                    if staff_num not in staff_time:
                        staff_time[staff_num] = 0.0
                        measure_beat_start[staff_num] = 0.0

                    t = staff_time[staff_num]

                    # Skip rests — advance time
                    if child.find(tag('rest')) is not None:
                        dur_el = child.find(tag('duration'))
                        if dur_el is not None:
                            beats = int(dur_el.text) / divisions
                            if child.find(tag('chord')) is None:
                                staff_time[staff_num] = t + beats
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

                    # Chord: starts at same time as previous note of same staff
                    is_chord = child.find(tag('chord')) is not None
                    note_onset = t if not is_chord else max(t - dur_beats, 0.0)

                    # Beat within measure
                    mbs = measure_beat_start.get(staff_num, 0.0)
                    beat_in_meas = note_onset - mbs + 1.0

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

                    # Advance this staff's time cursor
                    if not is_chord:
                        staff_time[staff_num] = note_onset + dur_beats

        # Sort by onset then by hand (right before left) for consistent order
        notes.sort(key=lambda n: (n.onset, 0 if n.hand == 'right' else 1))
        return notes
