"""
musicxml_writer.py — Ghi fingering kết quả trở lại MusicXML.

Chiến lược:
    - Với mỗi NoteEvent đã được gán finger, tìm xml_element tương ứng
    - Tìm hoặc tạo chuỗi <notations><technical><fingering>
    - Set text = str(finger)
    - Ghi ra file output (không overwrite input)
"""

from __future__ import annotations
from typing import List, Tuple
import xml.etree.ElementTree as ET

from src.musicxml_parser import NoteEvent


def inject_fingering(
    input_path: str,
    assignments: List[Tuple[NoteEvent, int]],
    output_path: str,
) -> None:
    """Inject computed fingerings vào MusicXML và ghi ra file.

    Args:
        input_path:   Đường dẫn file MusicXML gốc
        assignments:  List[(NoteEvent, finger)] từ solver
        output_path:  Đường dẫn file output
    """
    # Parse lại file gốc để giữ nguyên format
    ET.register_namespace('', 'http://www.musicxml.org/schema/mxml')
    tree = ET.parse(input_path)

    for note, finger in assignments:
        if finger is None or finger < 1 or finger > 5:
            continue

        elem = note.xml_element

        # Tìm hoặc tạo <notations>
        notations = elem.find('notations')
        if notations is None:
            notations = ET.SubElement(elem, 'notations')

        # Tìm hoặc tạo <technical>
        technical = notations.find('technical')
        if technical is None:
            technical = ET.SubElement(notations, 'technical')

        # Tìm hoặc tạo <fingering>
        fingering_elem = technical.find('fingering')
        if fingering_elem is None:
            fingering_elem = ET.SubElement(technical, 'fingering')

        fingering_elem.text = str(finger)

    # Ghi ra file
    _write_pretty(tree, output_path)
    print(f"✅ Wrote {len(assignments)} fingerings → {output_path}")


def _write_pretty(tree: ET.ElementTree, path: str) -> None:
    """Ghi XML với declaration và indent."""
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None

    # Python 3.9+: ET.indent()
    try:
        ET.indent(tree.getroot(), space='  ')
    except AttributeError:
        pass  # Python < 3.9: bỏ qua indent

    tree.write(
        path,
        encoding='UTF-8',
        xml_declaration=True,
    )
