#!/usr/bin/env python3
"""
Grand Staff Demo: Run Phrase-Aware Fingering on a two-staff piano score.

Parses RH (staff 1) and LH (staff 2) separately, runs independent
phrase-aware pipelines with hand-correct ergonomics, then prints
per-note prediction vs. ground truth for both hands.

Usage:
    python scripts/demo_grand_staff.py test_file/FN7ALfpGxiI.musicxml
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from fingering.io.musicxml_reader import MusicXMLReader
from fingering.phrasing.pipeline import PhraseAwareFingering

NOTE_NAMES = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']

def note_name(pitch: int) -> str:
    return f"{NOTE_NAMES[pitch % 12]}{pitch // 12 - 1}"

def run_hand(notes, label: str, paf: PhraseAwareFingering):
    print(f"\n{'─'*62}")
    print(f"  {label}  ({len(notes)} notes)")
    print(f"{'─'*62}")
    if not notes:
        print("  (no notes)")
        return

    phrases   = paf.get_phrases(notes)
    fingering = paf.run(notes)

    # Phrase summary
    print(f"  Detected {len(phrases)} phrases:")
    for p in phrases:
        if not p.notes:
            continue
        climax = note_name(p.notes[p.climax_idx].pitch)
        print(f"    P{p.id+1:02d} m{p.notes[0].measure}→{p.notes[-1].measure:>2} "
              f"{p.intent.name:<12s} arc={p.arc_type.name:<6s} climax={climax}")

    # Per-note table (only GT-annotated notes, to keep output compact)
    gt_notes = [(i, n, fingering[i]) for i, n in enumerate(notes) if n.finger is not None]
    if not gt_notes:
        print(f"\n  (no ground-truth annotations in this hand)")
    else:
        print(f"\n  {'Meas':<5}{'Note':<6}{'Beat':<6}{'GT':>3}{'Pred':>5}  Match")
        matches = 0
        for i, note, pred in gt_notes:
            gt = note.finger
            ok = '✅' if gt == pred else '❌'
            if gt == pred: matches += 1
            print(f"  m{note.measure:<4d}{note_name(note.pitch):<6s}b{note.beat:<4.1f}"
                  f"  {gt:>1}  →  {pred:>1}   {ok}")
        mr = matches / len(gt_notes) * 100
        print(f"\n  Match rate: {matches}/{len(gt_notes)} = {mr:.1f}%")

def main(path: str):
    print(f"\n{'='*62}")
    print(f"  Automatic Piano Fingering — Grand Staff Demo")
    print(f"  File: {os.path.basename(path)}")
    print(f"{'='*62}")

    reader = MusicXMLReader()
    rh, lh = reader.parse_grand_staff(path)
    print(f"\n✓ Parsed: RH={len(rh)} notes, LH={len(lh)} notes")

    paf = PhraseAwareFingering(boundary_threshold=0.30)

    run_hand(rh, "RIGHT HAND (Staff 1 — Treble)", paf)
    run_hand(lh, "LEFT HAND  (Staff 2 — Bass)",   paf)

    print(f"\n{'='*62}\n")

if __name__ == '__main__':
    path = sys.argv[1] if len(sys.argv) > 1 else 'test_file/FN7ALfpGxiI.musicxml'
    main(path)
