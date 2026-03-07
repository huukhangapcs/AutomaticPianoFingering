"""
musicxml_parser.py — Parse MusicXML và trích xuất notes tay phải (staff=1).

Chiến lược:
    - Chỉ lấy <note> thuộc <staff>1</staff> (Treble clef, tay phải)
    - Bỏ qua <rest>
    - Group các <chord> thành ChordGroup (nốt cùng onset)
    - Tính tọa độ x = pitch_to_coord(step, octave, alter)
    - Đọc ground-truth <fingering> nếu có (để evaluate)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional
import xml.etree.ElementTree as ET

from src.physics_model import pitch_to_coord, is_black_key


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass
class NoteEvent:
    """Một nốt nhạc tay phải đã được enrich với tọa độ vật lý."""
    note_id: str                   # "m{measure}_n{idx}"
    measure: int
    step: str                      # 'C'..'B'
    octave: int
    alter: float                   # -1=flat, 0=natural, +1=sharp
    x: float                       # keyboard coordinate
    is_black: bool
    duration: int                  # in divisions
    onset_division: int            # tích lũy từ đầu bài (để sort)
    voice: int
    is_chord_member: bool          # True nếu là nốt thứ 2+ cùng onset
    chord_rank: int                # 0=highest pitch, 1=next lower, ...
    gt_finger: Optional[int]       # ground-truth fingering (từ file)
    xml_element: ET.Element        # tham chiếu để writer inject lại


@dataclass
class ChordGroup:
    """Nhóm các nốt cùng onset trong cùng voice."""
    onset_division: int
    notes: List[NoteEvent] = field(default_factory=list)

    @property
    def primary(self) -> NoteEvent:
        """Nốt cao nhất (highest x) — là nốt chính solver sẽ gán finger."""
        return self.notes[0]

    def assign_chord_ranks(self):
        """Sort note theo x giảm dần, gán chord_rank."""
        self.notes.sort(key=lambda n: -n.x)
        for rank, note in enumerate(self.notes):
            note.chord_rank = rank
            note.is_chord_member = (rank > 0)


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def parse_rh_notes(musicxml_path: str) -> tuple[List[NoteEvent], int, int]:
    """Parse MusicXML, trả về danh sách NoteEvent tay phải.

    Returns:
        (notes, divisions, tempo)
        - notes: danh sách NoteEvent đã sort theo onset, chord_rank=0 trước
        - divisions: số divisions per quarter note
        - tempo: BPM (từ <sound tempo="...">)
    """
    tree = ET.parse(musicxml_path)
    root = tree.getroot()

    divisions = 1
    tempo = 120
    notes: List[NoteEvent] = []

    # Accumulate onset (absolute division position)
    current_division = 0
    measure_num = 0

    for measure in root.iter('measure'):
        measure_num = int(measure.get('number', measure_num + 1))
        note_idx = 0
        measure_division = current_division  # start of this measure
        local_cursor = 0   # position within measure (in divisions)

        for elem in measure:
            tag = elem.tag

            # Update divisions from attributes
            if tag == 'attributes':
                div_elem = elem.find('divisions')
                if div_elem is not None:
                    divisions = int(div_elem.text)

            # Read tempo
            elif tag == 'direction':
                sound = elem.find('.//sound')
                if sound is not None and sound.get('tempo'):
                    tempo = int(float(sound.get('tempo')))

            # Backup: move cursor backward
            elif tag == 'backup':
                dur = elem.find('duration')
                if dur is not None:
                    local_cursor -= int(dur.text)

            # Forward: move cursor forward (skip)
            elif tag == 'forward':
                dur = elem.find('duration')
                if dur is not None:
                    local_cursor += int(dur.text)

            # Note element
            elif tag == 'note':
                # Skip rests
                if elem.find('rest') is not None:
                    dur = elem.find('duration')
                    if elem.find('chord') is None:
                        local_cursor += int(dur.text) if dur is not None else 0
                    continue

                # Only right hand: staff=1
                staff_elem = elem.find('staff')
                if staff_elem is not None and staff_elem.text != '1':
                    # still advance cursor for non-chord notes
                    dur_elem = elem.find('duration')
                    if elem.find('chord') is None and dur_elem is not None:
                        local_cursor += int(dur_elem.text)
                    continue

                # Pitch
                pitch = elem.find('pitch')
                if pitch is None:
                    continue

                step = pitch.find('step').text.upper()
                octave = int(pitch.find('octave').text)
                alter_elem = pitch.find('alter')
                alter = float(alter_elem.text) if alter_elem is not None else 0.0

                # Duration
                dur_elem = elem.find('duration')
                duration = int(dur_elem.text) if dur_elem is not None else divisions

                # Voice
                voice_elem = elem.find('voice')
                voice = int(voice_elem.text) if voice_elem is not None else 1

                # Chord flag: nốt này có cùng onset với nốt trước không?
                is_chord = elem.find('chord') is not None
                if is_chord:
                    onset = measure_division + local_cursor
                else:
                    onset = measure_division + local_cursor
                    local_cursor += duration

                # Coordinate
                x = pitch_to_coord(step, octave, alter)
                black = is_black_key(step, alter)

                # Ground-truth fingering
                gt = None
                fingering_elem = elem.find('.//fingering')
                if fingering_elem is not None:
                    try:
                        gt = int(fingering_elem.text)
                    except (ValueError, TypeError):
                        gt = None

                note = NoteEvent(
                    note_id=f"m{measure_num}_n{note_idx}",
                    measure=measure_num,
                    step=step,
                    octave=octave,
                    alter=alter,
                    x=x,
                    is_black=black,
                    duration=duration,
                    onset_division=onset,
                    voice=voice,
                    is_chord_member=False,  # updated later
                    chord_rank=0,
                    gt_finger=gt,
                    xml_element=elem,
                )
                notes.append(note)
                note_idx += 1

        current_division = measure_division + local_cursor

    # Group chords và assign ranks
    notes = _group_and_rank_chords(notes)
    return notes, divisions, tempo


def _group_and_rank_chords(notes: List[NoteEvent]) -> List[NoteEvent]:
    """Group các nốt cùng onset, sort chord theo x giảm dần, gán rank."""
    # Group by (onset_division, voice)
    groups: dict[tuple[int, int], list[NoteEvent]] = {}
    for note in notes:
        key = (note.onset_division, note.voice)
        groups.setdefault(key, []).append(note)

    result: List[NoteEvent] = []
    for key in sorted(groups.keys()):
        group = groups[key]
        # Sort: highest x first
        group.sort(key=lambda n: -n.x)
        for rank, note in enumerate(group):
            note.chord_rank = rank
            note.is_chord_member = (rank > 0)
        result.extend(group)

    return result


def get_primary_notes(notes: List[NoteEvent]) -> List[NoteEvent]:
    """Chỉ lấy nốt chính (chord_rank=0) — input cho solver."""
    return [n for n in notes if n.chord_rank == 0]


def get_chord_secondaries(notes: List[NoteEvent], primary: NoteEvent) -> List[NoteEvent]:
    """Lấy các nốt phụ trong cùng chord group với primary note."""
    return [
        n for n in notes
        if n.onset_division == primary.onset_division
           and n.voice == primary.voice
           and n.chord_rank > 0
    ]
