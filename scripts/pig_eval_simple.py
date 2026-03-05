"""
PIG Dataset Benchmark — Simplified Physics-First Fingering Engine.

Compares SimpleFingering (simple/) vs PhraseAwareFingering (phrasing/).
"""

import os
import re
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from fingering.models.note_event import NoteEvent
from fingering.simple.pipeline import PianoFingering


# ── reuse parser from pig_eval.py ────────────────────────────────────────────

def pitch_name_to_midi(name: str) -> int:
    note_basemap = {'C': 0, 'C#': 1, 'Db': 1, 'D': 2, 'D#': 3, 'Eb': 3,
                    'E': 4, 'F': 5, 'F#': 6, 'Gb': 6, 'G': 7, 'G#': 8,
                    'Ab': 8, 'A': 9, 'A#': 10, 'Bb': 10, 'B': 11}
    match = re.match(r"([A-G][#b]?)(-?\d+)", name)
    if not match:
        return 60
    return note_basemap[match.group(1)] + (int(match.group(2)) + 1) * 12


def parse_pig_file(filepath):
    rh_notes, lh_notes = [], []
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    for line in lines:
        line = line.strip()
        if not line or line.startswith('//'):
            continue
        parts = line.split('\t')
        if len(parts) < 8:
            continue
        onset     = float(parts[1])
        offset    = float(parts[2])
        pitch_name = parts[3]
        channel   = int(parts[6])
        finger_str = parts[7]

        finger = None
        if finger_str and finger_str != 'N':
            try:
                finger = abs(int(finger_str.split('_')[0]))
            except ValueError:
                pass

        pitch_midi = pitch_name_to_midi(pitch_name)
        beat = onset / 0.5
        measure = int(onset / 2.0) + 1

        note = NoteEvent(
            pitch=pitch_midi,
            onset=beat,
            offset=beat + (offset - onset) / 0.5,
            hand='right' if channel == 0 else 'left',
            measure=measure, beat=beat, finger=finger,
        )
        if channel == 0:
            rh_notes.append(note)
        else:
            lh_notes.append(note)

    rh_notes.sort(key=lambda x: x.onset)
    lh_notes.sort(key=lambda x: x.onset)
    return rh_notes, lh_notes


# ── benchmark ────────────────────────────────────────────────────────────────

def run_benchmark(pig_dir: str, n_files: int = 20):
    fingering_dir = os.path.join(pig_dir, "FingeringFiles")
    if not os.path.exists(fingering_dir):
        print(f"Error: {fingering_dir} not found.")
        return

    txt_files = sorted(f for f in os.listdir(fingering_dir) if f.endswith('.txt'))

    pipeline = PianoFingering(bpm=120.0)

    total_notes = correct = 0
    file_scores = []

    print(f"{'─'*60}")
    print(f"  Simplified Physics-First Engine — PIG Benchmark")
    print(f"  Files: {min(n_files, len(txt_files))} / {len(txt_files)}")
    print(f"{'─'*60}")

    for filename in txt_files[:n_files]:
        filepath = os.path.join(fingering_dir, filename)
        rh, lh = parse_pig_file(filepath)
        if not rh:
            continue

        gt_fingers = [n.finger for n in rh]
        for n in rh:
            n.finger = None

        pred = pipeline.run(rh)

        match = valid = 0
        for p, g in zip(pred, gt_fingers):
            if g is not None and g != 0:
                valid += 1
                if p == g:
                    match += 1

        if valid > 0:
            acc = match / valid
            file_scores.append(acc)
            total_notes += valid
            correct += match
            print(f"  {filename[:18]:<18} {acc*100:5.1f}%  ({match}/{valid})")

    if total_notes > 0:
        overall = correct / total_notes
        print(f"{'─'*60}")
        print(f"  Total RH notes : {total_notes}")
        print(f"  Correct        : {correct}")
        print(f"  Overall MR     : {overall*100:.2f}%")
        print(f"  Avg / file     : {np.mean(file_scores)*100:.2f}%")
        print(f"{'─'*60}")


if __name__ == '__main__':
    pig_dir = sys.argv[1] if len(sys.argv) > 1 else \
        '/Users/lap02459/AutomaticPianoFingering/PianoFingeringDataset_v1.2'
    n_files = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    run_benchmark(pig_dir, n_files)
