#!/usr/bin/env python3
"""
Demo: Run Phrase-Aware Fingering on a real MusicXML file.

Reads the file, runs the pipeline, then prints:
  - Detected phrases with their intent and arc
  - Per-note comparison: our prediction vs. ground truth (if available)
  - Match rate summary

Usage:
    python scripts/demo_musicxml.py test_file/cW8VLC9nnTo.musicxml
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from fingering.io.musicxml_reader import MusicXMLReader
from fingering.phrasing.pipeline import PhraseAwareFingering

NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F',
              'F#', 'G', 'G#', 'A', 'A#', 'B']

def note_name(pitch: int) -> str:
    pc  = pitch % 12
    oct_ = pitch // 12 - 1
    return f"{NOTE_NAMES[pc]}{oct_}"

def main(path: str):
    print(f"\n{'='*60}")
    print(f"  Automatic Piano Fingering — Demo")
    print(f"  File: {os.path.basename(path)}")
    print(f"{'='*60}\n")

    # 1. Parse
    reader = MusicXMLReader()
    notes  = reader.parse(path, hand='right')
    print(f"✓ Parsed {len(notes)} notes\n")

    # 2. Run pipeline
    paf = PhraseAwareFingering(boundary_threshold=0.30)

    # Get phrases for analysis
    phrases = paf.get_phrases(notes)
    print(f"✓ Detected {len(phrases)} phrases\n")

    for p in phrases:
        climax_note = note_name(p.notes[p.climax_idx].pitch) if p.notes else '?'
        print(f"  Phrase {p.id+1:2d} | m{p.notes[0].measure:2d}→m{p.notes[-1].measure:2d} "
              f"| {len(p.notes):2d} notes "
              f"| {p.intent.name:<12s}"
              f"| Arc: {p.arc_type.name:<6s}"
              f"| Climax: {climax_note}")

    # 3. Get fingering
    fingering = paf.run(notes)

    # 4. Compare with ground truth
    print(f"\n{'─'*60}")
    print(f"  {'Measure':<8} {'Note':<6} {'Beat':<6} {'GT':>4} {'Pred':>5} {'✓?':>4}")
    print(f"{'─'*60}")

    gt_available = [n for n in notes if n.finger is not None]
    matches = 0
    total_gt = 0

    for note, f_pred in zip(notes, fingering):
        gt = note.finger  # ground truth (from file annotation)
        match_icon = ''
        if gt is not None:
            total_gt += 1
            if gt == f_pred:
                matches += 1
                match_icon = '✅'
            else:
                match_icon = '❌'

        gt_str  = str(gt) if gt is not None else '—'
        print(f"  m{note.measure:<7d} {note_name(note.pitch):<6s} b{note.beat:<4.1f} "
              f"  {gt_str:>2}  →  {f_pred:>1}   {match_icon}")

    # 5. Summary
    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    print(f"  Total notes    : {len(notes)}")
    print(f"  Notes with GT  : {total_gt}")
    if total_gt > 0:
        match_rate = matches / total_gt * 100
        print(f"  Matches        : {matches}/{total_gt}")
        print(f"  Match Rate     : {match_rate:.1f}%")
    print(f"  Phrases        : {len(phrases)}")

    # 6. Per-phrase breakdown
    print(f"\n{'─'*60}")
    print(f"  Per-phrase match analysis:")
    note_idx = 0
    for p in phrases:
        n = len(p.notes)
        ph_matches = 0
        ph_gt = 0
        for i in range(n):
            real_note = p.notes[i]
            pred = fingering[note_idx + i]
            if real_note.finger is not None:
                ph_gt += 1
                if real_note.finger == pred:
                    ph_matches += 1
        mr = ph_matches / ph_gt * 100 if ph_gt > 0 else float('nan')
        note_idx += n
        mr_str = f"{mr:.0f}%" if ph_gt > 0 else "  N/A "
        print(f"  Phrase {p.id+1:2d} [{p.intent.name:<12s}] "
              f"match={mr_str:>5}  ({ph_matches}/{ph_gt} GT)")
    print()

if __name__ == '__main__':
    path = sys.argv[1] if len(sys.argv) > 1 else 'test_file/cW8VLC9nnTo.musicxml'
    main(path)
