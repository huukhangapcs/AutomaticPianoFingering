#!/usr/bin/env python3
"""
main.py — Entry point: đọc MusicXML → solve → ghi ouput.

Usage:
    python main.py --input test_file/FN7ALfpGxiI.musicxml --output output/result.musicxml
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from src.fingering_solver import solve
from src.musicxml_writer import inject_fingering


def main():
    parser = argparse.ArgumentParser(description='Automatic Piano Fingering — Right Hand')
    parser.add_argument('--input', required=True, help='Input MusicXML file')
    parser.add_argument('--output', default='output/result.musicxml', help='Output MusicXML file')
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"❌ File không tồn tại: {args.input}")
        sys.exit(1)

    print(f"📄 Parsing: {args.input}")
    from src.musicxml_parser import parse_hand_notes
    from src.musicxml_writer import sparsify_assignments
    
    # Process Right Hand (staff=1)
    notes_rh, div_rh, tempo_rh = parse_hand_notes(args.input, staff_id=1)
    print(f"   → RH: {len(notes_rh)} notes, divisions={div_rh}, tempo={tempo_rh}")
    
    # Process Left Hand (staff=2)
    notes_lh, div_lh, tempo_lh = parse_hand_notes(args.input, staff_id=2)
    print(f"   → LH: {len(notes_lh)} notes, divisions={div_lh}, tempo={tempo_lh}")

    print("🎹 Solving fingering (Viterbi DP)...")
    assign_rh = solve(notes_rh, div_rh, tempo_rh, is_lh=False) if notes_rh else []
    assign_lh = solve(notes_lh, div_lh, tempo_lh, is_lh=True) if notes_lh else []
    
    print(f"   → RH raw assignments: {len(assign_rh)}")
    print(f"   → LH raw assignments: {len(assign_lh)}")

    # Sparsify (Filter obvious fingerings)
    sparse_rh = sparsify_assignments(assign_rh, is_lh=False)
    sparse_lh = sparsify_assignments(assign_lh, is_lh=True)
    
    print(f"   → RH sparse markers : {len(sparse_rh)}")
    print(f"   → LH sparse markers : {len(sparse_lh)}")

    # Combine
    combined_assignments = sparse_rh + sparse_lh

    inject_fingering(args.input, combined_assignments, args.output)


if __name__ == '__main__':
    main()
