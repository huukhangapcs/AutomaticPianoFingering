#!/usr/bin/env python3
"""
evaluate_dataset.py — So sánh Predicted Fingering với tập dữ liệu PianoFingeringDataset_v1.2
Đọc format dữ liệu txt của dataset:
Cấu trúc mỗi cột của file .txt:
0: note_id
1: onset_time (seconds)
2: offset_time (seconds)
3: pitch_name (e.g. C4)
4: onset_velocity
5: offset_velocity
6: channel (0 = right hand, 1 = left hand)
7: fingering (1-5 for right hand, -1 to -5 for left hand)
"""

import os
import sys
import glob
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))

from src.musicxml_parser import NoteEvent
from src.fingering_solver import solve
from src.physics_model import pitch_to_coord, is_black_key


def parse_dataset_txt(filepath: str) -> list[NoteEvent]:
    """Parse format .txt của dataset thành danh sách NoteEvent (chỉ lấy tay phải)."""
    notes = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('//'):
                continue
            
            parts = line.split('\t')
            if len(parts) < 8:
                continue
            
            channel = int(parts[6])
            if channel != 0:
                continue  # Bỏ qua tay trái
                
            pitch_name = parts[3]
            step = pitch_name[0]
            
            # Xử lý alter (sharp/flat)
            alter = 0.0
            if len(pitch_name) > 2 and pitch_name[1] == '#':
                alter = 1.0
                octave = int(pitch_name[2:])
            elif len(pitch_name) > 2 and pitch_name[1] == 'b':
                alter = -1.0
                octave = int(pitch_name[2:])
            else:
                octave = int(pitch_name[1:])
                
            onset_sec = float(parts[1])
            onset_div = int(onset_sec * 1000) # Giả lập division bằng millisecond
            
            dur_sec = float(parts[2]) - onset_sec
            dur_div = int(dur_sec * 1000)
            
            x, y, z = pitch_to_coord(step, octave, alter)
            is_black = is_black_key(step, alter)
            
            raw_finger = parts[7]
            # Handle substitutions like "4_1" -> we just take the first one "4" as main finger
            if '_' in raw_finger:
                raw_finger = raw_finger.split('_')[0]
                
            gt_finger = int(raw_finger)
            
            note = NoteEvent(
                note_id=parts[0],
                measure=1,  # Not relevant for txt dataset
                step=step,
                octave=octave,
                alter=alter,
                x=x,
                y=y,
                z=z,
                is_black=is_black,
                duration=dur_div,
                onset_division=onset_div,
                voice=1,
                is_chord_member=False,
                chord_rank=0,
                gt_finger=gt_finger,
                xml_element=None
            )
            notes.append(note)
            
    # Group chords
    from src.musicxml_parser import _group_and_rank_chords
    notes = _group_and_rank_chords(notes)
    
    return notes


def evaluate_dataset(dataset_dir: str):
    """Quét tất cả file txt trong dataset và chạy đánh giá."""
    txt_files = glob.glob(os.path.join(dataset_dir, "*_fingering.txt"))
    if not txt_files:
        print(f"❌ Không tìm thấy file txt nào trong {dataset_dir}")
        return
        
    print(f"🔍 Bắt đầu đánh giá trên {len(txt_files)} files...")
    
    total_notes = 0
    total_exact = 0
    total_obo = 0
    
    for i, filepath in enumerate(txt_files):
        filename = os.path.basename(filepath)
        print(f"  [{i+1}/{len(txt_files)}] {filename}...", end='', flush=True)
        
        try:
            notes = parse_dataset_txt(filepath)
            if not notes:
                print(" (Bỏ qua: không có nốt tay phải)")
                continue
                
            # Simulate divisions (using 1000 as we converted seconds to ms)
            assignments = solve(notes, divisions=1000, is_lh=False)
            
            results = [(note, finger) for note, finger in assignments
                       if note.chord_rank == 0 and note.gt_finger is not None]
                       
            if not results:
                print(" (Bỏ qua: không có kết quả hợp lệ)")
                continue
                
            file_total = len(results)
            file_exact = sum(1 for note, f in results if note.gt_finger == f)
            file_obo = sum(1 for note, f in results 
                           if note.gt_finger is not None and abs(note.gt_finger - f) == 1)
                           
            total_notes += file_total
            total_exact += file_exact
            total_obo += file_obo
            
            acc = file_exact / file_total * 100
            print(f" {file_exact}/{file_total} ({acc:.1f}%)")
            
        except Exception as e:
            print(f" (Lỗi: {str(e)})")

    if total_notes == 0:
        print("Không có nốt nào được đánh giá.")
        return
        
    exact_rate = total_exact / total_notes * 100
    obo_rate = total_obo / total_notes * 100
    
    print("\n" + "=" * 55)
    print("  DATASET EVALUATION REPORT")
    print("=" * 55)
    print(f"  Total files evaluated : {len(txt_files)}")
    print(f"  Total notes evaluated : {total_notes}")
    print(f"  Exact match           : {total_exact} ({exact_rate:.2f}%)")
    print(f"  Off-by-one            : {total_obo} ({obo_rate:.2f}%)")
    print(f"  Combined (≤1 error)   : {total_exact + total_obo} ({(exact_rate + obo_rate):.2f}%)")
    print("=" * 55 + "\n")


if __name__ == '__main__':
    evaluate_dataset('PianoFingeringDataset_v1.2/FingeringFiles')
