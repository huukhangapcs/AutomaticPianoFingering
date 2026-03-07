#!/usr/bin/env python3
"""
evaluate.py — So sánh predicted fingering với ground-truth trong MusicXML.

Usage:
    python evaluate.py --input test_file/FN7ALfpGxiI.musicxml
"""

import argparse
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))

from src.musicxml_parser import parse_hand_notes, get_primary_notes
from src.fingering_solver import solve


def evaluate(input_path: str) -> dict:
    """Chạy solver và so sánh với ground-truth.

    Returns:
        dict chứa các metrics: match_rate, off_by_one_rate, per_measure
    """
    notes, divisions, tempo = parse_hand_notes(input_path, staff_id=1)
    assignments = solve(notes, divisions, tempo, is_lh=False)

    # Chỉ evaluate primary notes có ground-truth
    results = [(note, finger) for note, finger in assignments
               if note.chord_rank == 0 and note.gt_finger is not None]

    if not results:
        print("⚠️  Không có ground-truth fingering trong file.")
        return {}

    total = len(results)
    exact = sum(1 for note, f in results if note.gt_finger == f)
    off_by_one = sum(1 for note, f in results
                     if note.gt_finger is not None and abs(note.gt_finger - f) == 1)

    # Per-measure breakdown
    per_measure: dict[int, dict] = defaultdict(lambda: {'total': 0, 'correct': 0, 'errors': []})
    for note, f in results:
        m = note.measure
        per_measure[m]['total'] += 1
        if note.gt_finger == f:
            per_measure[m]['correct'] += 1
        else:
            per_measure[m]['errors'].append(
                f"{note.step}{note.octave}: GT={note.gt_finger} Pred={f}"
            )

    # Error distribution
    error_dist: dict[int, int] = defaultdict(int)
    for note, f in results:
        if note.gt_finger is not None:
            diff = abs(note.gt_finger - f)
            error_dist[diff] += 1

    return {
        'total': total,
        'exact': exact,
        'match_rate': exact / total,
        'off_by_one': off_by_one,
        'off_by_one_rate': off_by_one / total,
        'per_measure': dict(per_measure),
        'error_dist': dict(error_dist),
        'assignments': results,
    }


def print_report(metrics: dict):
    """In báo cáo đẹp ra terminal."""
    if not metrics:
        return

    total = metrics['total']
    exact = metrics['exact']
    obo = metrics['off_by_one']
    mr = metrics['match_rate'] * 100
    obo_r = metrics['off_by_one_rate'] * 100

    print("\n" + "=" * 55)
    print("  AUTOMATIC PIANO FINGERING — EVALUATION REPORT")
    print("=" * 55)
    print(f"  Total notes evaluated : {total}")
    print(f"  Exact match           : {exact} ({mr:.1f}%)")
    print(f"  Off-by-one            : {obo} ({obo_r:.1f}%)")
    print(f"  Combined (≤1 error)   : {exact + obo} ({(mr + obo_r):.1f}%)")
    print("=" * 55)

    # Error distribution
    print("\n  Error Distribution:")
    print(f"  {'|diff|':>8}  {'count':>6}  {'%':>6}")
    print("  " + "-" * 24)
    for diff in sorted(metrics['error_dist']):
        count = metrics['error_dist'][diff]
        pct = count / total * 100
        bar = '█' * int(pct / 2)
        print(f"  {diff:>8}  {count:>6}  {pct:>5.1f}%  {bar}")

    # Per-measure (show only measures with notes)
    print("\n  Per-measure Accuracy:")
    print(f"  {'Meas':>6}  {'Acc':>6}  {'Details'}")
    print("  " + "-" * 50)
    for m in sorted(metrics['per_measure']):
        info = metrics['per_measure'][m]
        t = info['total']
        c = info['correct']
        acc = c / t * 100 if t > 0 else 0
        status = '✓' if acc == 100 else ('~' if acc >= 50 else '✗')
        errors = ', '.join(info['errors'][:3])
        ellipsis = '...' if len(info['errors']) > 3 else ''
        print(f"  {m:>6}  {acc:>5.0f}%  {status}  {errors}{ellipsis}")

    print("=" * 55 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Evaluate piano fingering against ground-truth')
    parser.add_argument('--input', required=True, help='Input MusicXML file with GT fingering')
    parser.add_argument('--verbose', action='store_true', help='Print per-note comparison')
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"❌ File không tồn tại: {args.input}")
        sys.exit(1)

    print(f"📄 Evaluating: {args.input}")
    metrics = evaluate(args.input)

    if args.verbose and metrics.get('assignments'):
        print(f"\n{'Meas':>5} {'Note':>6} {'x':>6} {'Pred':>5} {'GT':>4} {'✓?':>3}")
        print("-" * 35)
        for note, f in metrics['assignments']:
            match = '✓' if note.gt_finger == f else '✗'
            print(f"{note.measure:>5} {note.step}{note.octave:>5} "
                  f"{note.x:>6.1f} {f:>5} {note.gt_finger:>4} {match:>3}")

    print_report(metrics)


if __name__ == '__main__':
    main()
