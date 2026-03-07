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

from src.musicxml_parser import parse_rh_notes
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
    notes, divisions, tempo = parse_rh_notes(args.input)
    print(f"   → {len(notes)} notes (RH), divisions={divisions}, tempo={tempo}")

    print("🎹 Solving fingering (Viterbi DP)...")
    assignments = solve(notes, divisions)
    print(f"   → {len(assignments)} assignments")

    # Print preview
    from src.musicxml_parser import get_primary_notes
    primaries = get_primary_notes(notes)
    print(f"\n{'Measure':>7} {'Note':>6} {'x':>6} {'Predict':>8} {'GT':>4}")
    print("-" * 40)
    for note, finger in assignments:
        if note.chord_rank == 0:
            gt_str = str(note.gt_finger) if note.gt_finger else ' -'
            match = '✓' if note.gt_finger == finger else '✗'
            print(f"{note.measure:>7} {note.step}{note.octave:>5} {note.x:>6.1f} "
                  f"{finger:>8} {gt_str:>4} {match}")

    inject_fingering(args.input, assignments, args.output)


if __name__ == '__main__':
    main()
