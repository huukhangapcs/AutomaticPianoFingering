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


def sparsify_assignments(
    assignments: List[Tuple[NoteEvent, int]],
    is_lh: bool = False
) -> List[Tuple[NoteEvent, int]]:
    """Lọc bớt các số ngón tay hiển nhiên để tránh rác bản nhạc (Visual Clutter).
    
    Rule:
    - Luôn hiện số ở nốt đầu tiên của tay (hoặc sau đoạn nghỉ/câu mới).
    - Với tay trái (LH): Ưu tiên hiện ở đầu mỗi ô nhịp (Measure) hoặc khi phải nhảy xa (HAND_SHIFT / RESET / Phá form).
    - Với tay phải (RH): Ẩn các đoạn Scale chạy ngón liền bậc (IN_FORM). Chỉ hiện khi vắt ngón (CROSS_OVER, THUMB_UNDER), dãn tay (STRETCH) hoặc nhảy.
    - Hợp âm: Nếu nốt cao nhất bị ẩn, các nốt bè dưới cũng ẩn theo. Nếu nốt cao nhất hiện, hiện toàn bộ hợp âm.
    """
    from src.physics_model import MoveType, classify_move
    
    filtered = []
    last_primary_note = None
    last_primary_finger = None
    last_primary_kept = True
    last_measure = -1
    
    for note, finger in assignments:
        if note.chord_rank == 0:
            kept = False
            
            if last_primary_note is None:
                kept = True
            elif note.onset_division > last_primary_note.onset_division + last_primary_note.duration:
                # Có khoảng nghỉ (rest) -> Câu mới
                kept = True
            elif is_lh and note.measure != last_measure:
                # Tay trái: Luôn đánh dấu nốt đầu tiên của ô nhịp
                kept = True
            else:
                # Cùng chung logic tay phải/trái nhờ việc lật trục X ở Parser
                move_type = classify_move(last_primary_finger, finger, last_primary_note.x, note.x)
                
                if not is_lh:
                    # Tay phải: Giữ lại mọi thứ KHÁC IN_FORM (vắt ngón, dãn tay, nhảy quãng)
                    if move_type != MoveType.IN_FORM:
                        kept = True
                else:
                    # Tay trái: Ít đánh số hơn nữa. Chỉ hiện khi chuyển ngón phức tạp hoặc nhảy
                    if move_type in (MoveType.HAND_SHIFT, MoveType.RESET, MoveType.CROSS_OVER, MoveType.THUMB_UNDER):
                        kept = True
                        
            if kept:
                filtered.append((note, finger))
                last_primary_kept = True
            else:
                last_primary_kept = False
                
            last_primary_note = note
            last_primary_finger = finger
            last_measure = note.measure
        else:
            # Chord member xử lý ăn theo nốt Primary
            if last_primary_kept:
                filtered.append((note, finger))

    return filtered


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
