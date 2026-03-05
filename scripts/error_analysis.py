"""
Error Categorizer — deep analysis of wrong fingering predictions.

Classifies each wrong prediction into one of these error types:
  1. OFF_BY_ONE   — predicted finger is adjacent to GT (e.g. GT=2, pred=3)
  2. THUMB_MISS   — GT uses thumb(1), predicted doesn't (or vice versa)
  3. WEAK_OVER    — predicted uses a weaker finger than GT
  4. WRONG_HAND_POSITION — predicted implies a completely different hand position
  5. CHORD_ERROR  — error on a chord note
  6. SCALE_ERROR  — error inside a scale passage
  7. LARGE_JUMP   — error near a large pitch jump (≥ 7 semitones)
  8. BLACK_KEY    — error where the note is a black key
  9. OTHER        — everything else

Prints a summary table and per-category match rate.
"""

import os
import re
import sys
from collections import defaultdict
from typing import List, Tuple, Dict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from fingering.models.note_event import NoteEvent
from fingering.phrasing.pipeline import PhraseAwareFingering

# Inline parse_pig_file to avoid circular module import
def pitch_name_to_midi(name: str) -> int:
    note_basemap = {'C': 0, 'C#': 1, 'Db': 1, 'D': 2, 'D#': 3, 'Eb': 3,
                    'E': 4, 'F': 5, 'F#': 6, 'Gb': 6, 'G': 7, 'G#': 8,
                    'Ab': 8, 'A': 9, 'A#': 10, 'Bb': 10, 'B': 11}
    match = re.match(r"([A-G][#b]?)(-?\d+)", name)
    if not match:
        return 60
    pc = match.group(1)
    octave = int(match.group(2))
    return note_basemap[pc] + (octave + 1) * 12

def parse_pig_file(filepath: str):
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
        onset = float(parts[1])
        offset = float(parts[2])
        pitch_name = parts[3]
        channel = int(parts[6])
        finger_str = parts[7]
        finger = None
        if finger_str and finger_str != 'N':
            base_f = finger_str.split('_')[0]
            try:
                finger = abs(int(base_f))
            except ValueError:
                pass
        pitch_midi = pitch_name_to_midi(pitch_name)
        beat = onset / 0.5
        measure = int(onset / 2.0) + 1
        note = NoteEvent(pitch=pitch_midi, onset=beat,
                         offset=beat + (offset - onset) / 0.5,
                         hand='right' if channel == 0 else 'left',
                         measure=measure, beat=beat, finger=finger)
        if channel == 0:
            rh_notes.append(note)
        elif channel == 1:
            lh_notes.append(note)
    rh_notes.sort(key=lambda x: x.beat)
    lh_notes.sort(key=lambda x: x.beat)
    return rh_notes, lh_notes



# ─────────────────────────────────────────────────────────────
# Error type definitions
# ─────────────────────────────────────────────────────────────

ERROR_TYPES = [
    'OFF_BY_ONE',
    'THUMB_MISS',
    'WEAK_OVER',
    'WRONG_POSITION',
    'CHORD_ERROR',
    'SCALE_ERROR',
    'LARGE_JUMP',
    'BLACK_KEY',
    'OTHER',
]

_BLACK_PC = {1, 3, 6, 8, 10}  # C#, D#, F#, G#, A#

def _is_black(pitch: int) -> bool:
    return (pitch % 12) in _BLACK_PC

def _is_scale_run(notes: List[NoteEvent], idx: int) -> bool:
    """True if note[idx] is inside a 4+ note stepwise run."""
    n = len(notes)
    if idx == 0 or idx >= n - 1:
        return False
    asc = notes[idx+1].pitch > notes[idx-1].pitch
    count = 0
    for di in range(-3, 4):
        j = idx + di
        if 0 < j < n:
            iv = abs(notes[j].pitch - notes[j-1].pitch)
            if iv in (1, 2):
                count += 1
    return count >= 3

def _is_large_jump(notes: List[NoteEvent], idx: int) -> bool:
    """True if there's a jump of ≥ 7 semitones before or after this note."""
    n = len(notes)
    if idx > 0 and abs(notes[idx].pitch - notes[idx-1].pitch) >= 7:
        return True
    if idx < n-1 and abs(notes[idx+1].pitch - notes[idx].pitch) >= 7:
        return True
    return False

def _is_chord(notes: List[NoteEvent], idx: int) -> bool:
    """True if this note shares onset with the adjacent note."""
    n = len(notes)
    if idx > 0 and abs(notes[idx].onset - notes[idx-1].onset) < 0.05:
        return True
    if idx < n-1 and abs(notes[idx].onset - notes[idx+1].onset) < 0.05:
        return True
    return False

def classify_error(
    notes: List[NoteEvent],
    idx: int,
    gt: int,
    pred: int,
) -> str:
    """Classify a single wrong prediction into an error category."""
    note = notes[idx]

    # Is it a chord note?
    if _is_chord(notes, idx):
        return 'CHORD_ERROR'

    # Thumb-related
    if gt == 1 and pred != 1:
        return 'THUMB_MISS'
    if pred == 1 and gt != 1:
        return 'THUMB_MISS'

    # Off-by-one
    if abs(gt - pred) == 1:
        return 'OFF_BY_ONE'

    # Scale context
    if _is_scale_run(notes, idx):
        return 'SCALE_ERROR'

    # Large jump nearby
    if _is_large_jump(notes, idx):
        return 'LARGE_JUMP'

    # Black key note
    if _is_black(note.pitch):
        return 'BLACK_KEY'

    # Predicted a weaker finger (higher number) than GT
    gt_strength = {1: 2, 2: 5, 3: 5, 4: 3, 5: 1}  # approximate strength 1-5
    if gt_strength.get(pred, 0) < gt_strength.get(gt, 0):
        return 'WEAK_OVER'

    # Completely different hand position (difference >= 3)
    if abs(gt - pred) >= 3:
        return 'WRONG_POSITION'

    return 'OTHER'


# ─────────────────────────────────────────────────────────────
# Main analysis
# ─────────────────────────────────────────────────────────────

def run_error_analysis(pig_dir: str, n_files: int = 20):
    fingering_dir = os.path.join(pig_dir, "FingeringFiles")
    txt_files = sorted([f for f in os.listdir(fingering_dir) if f.endswith('.txt')])

    pipeline = PhraseAwareFingering()

    error_counts: Dict[str, int] = defaultdict(int)
    correct_by_cat: Dict[str, int] = defaultdict(int)  # for files where we see the category
    total_wrong = 0
    total_correct = 0

    # Also track: confusion matrix GT → pred (top 10 pairs)
    confusion: Dict[Tuple[int,int], int] = defaultdict(int)

    print(f"\nError Analysis on first {n_files} PIG files\n{'='*55}")

    for filename in txt_files[:n_files]:
        filepath = os.path.join(fingering_dir, filename)
        rh, lh = parse_pig_file(filepath)
        if not rh:
            continue

        gt_fingers = [n.finger for n in rh]
        for n in rh:
            n.finger = None
        pred_fingers = pipeline.run(rh, companion_notes=lh)

        for idx, (pred, gt) in enumerate(zip(pred_fingers, gt_fingers)):
            if gt is None or gt == 0:
                continue
            if pred == gt:
                total_correct += 1
            else:
                total_wrong += 1
                cat = classify_error(rh, idx, gt, pred)
                error_counts[cat] += 1
                confusion[(gt, pred)] += 1

    # ─── Print summary ───────────────────────────────────────
    total = total_correct + total_wrong
    print(f"\nTotal notes evaluated : {total}")
    print(f"Correct predictions   : {total_correct} ({100*total_correct/total:.1f}%)")
    print(f"Wrong predictions     : {total_wrong}  ({100*total_wrong/total:.1f}%)")
    print(f"\nError breakdown ({total_wrong} total errors):\n")
    print(f"  {'Category':<20} {'Count':>6}  {'% of errors':>12}  {'Impact':>8}")
    print(f"  {'-'*52}")

    for cat in ERROR_TYPES:
        count = error_counts.get(cat, 0)
        pct   = 100 * count / total_wrong if total_wrong else 0
        impact = count / total if total else 0
        bar   = '█' * int(pct / 2)
        print(f"  {cat:<20} {count:>6}  {pct:>10.1f}%  {impact*100:>6.2f}%  {bar}")

    # ─── Confusion matrix top-15 ─────────────────────────────
    print(f"\nTop 15 GT→Pred confusion pairs:\n")
    print(f"  {'GT':>4}  {'Pred':>5}  {'Count':>6}")
    print(f"  {'-'*20}")
    for (gt, pred), count in sorted(confusion.items(), key=lambda x: -x[1])[:15]:
        print(f"  f{gt:>2}  →  f{pred:<2}  {count:>6}x")

    # ─── Insight summary ─────────────────────────────────────
    print(f"\n{'='*55}")
    print("KEY INSIGHTS:")
    top_cat = max(error_counts, key=error_counts.get)
    print(f"  → Biggest error category: {top_cat} ({error_counts[top_cat]} errors)")
    off1 = error_counts.get('OFF_BY_ONE', 0)
    print(f"  → Off-by-one: {off1} ({100*off1/total_wrong:.0f}% of errors) — mostly fixable by cost tuning")
    thumb = error_counts.get('THUMB_MISS', 0)
    print(f"  → Thumb miss: {thumb} — suggests boundary/crossing logic issues")
    scale = error_counts.get('SCALE_ERROR', 0)
    print(f"  → Scale errors: {scale} — suggests tone-specific fingering gaps")


if __name__ == '__main__':
    pig_dir = '/Users/lap02459/AutomaticPianoFingering/PianoFingeringDataset_v1.2'
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    run_error_analysis(pig_dir, n_files=n)
