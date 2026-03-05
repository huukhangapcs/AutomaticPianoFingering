import os
import re
from typing import List, Tuple
from collections import defaultdict
import numpy as np
from pathlib import Path

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from fingering.models.note_event import NoteEvent
from fingering.phrasing.pipeline import PhraseAwareFingering

def pitch_name_to_midi(name: str) -> int:
    """Convert C4 to 60, etc."""
    # Sometimes there might be sharps/flats: C#4, Bb3
    note_basemap = {'C': 0, 'C#': 1, 'Db': 1, 'D': 2, 'D#': 3, 'Eb': 3,
                    'E': 4, 'F': 5, 'F#': 6, 'Gb': 6, 'G': 7, 'G#': 8,
                    'Ab': 8, 'A': 9, 'A#': 10, 'Bb': 10, 'B': 11}
    
    match = re.match(r"([A-G][#b]?)(-?\d+)", name)
    if not match:
        return 60
    
    pitch_class = match.group(1)
    octave = int(match.group(2))
    
    return note_basemap[pitch_class] + (octave + 1) * 12


def parse_pig_file(filepath: str) -> Tuple[List[NoteEvent], List[NoteEvent]]:
    """Parse a PIG fingering file into RH and LH NoteEvent lists."""
    rh_notes = []
    lh_notes = []

    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    for line in lines:
        line = line.strip()
        if not line or line.startswith('//'):
            continue
            
        parts = line.split('\t')
        if len(parts) < 8:
            continue
            
        note_id = int(parts[0])
        onset = float(parts[1])
        offset = float(parts[2])
        pitch_name = parts[3]
        channel = int(parts[6])
        finger_str = parts[7]
        
        # Parse finger (e.g., "4_1" -> 4, "N" -> None, "-5" -> 5)
        finger = None
        if finger_str and finger_str != 'N':
            # take first number if substitution
            base_f = finger_str.split('_')[0]
            try:
                finger = abs(int(base_f))
            except ValueError:
                finger = None
                
        pitch_midi = pitch_name_to_midi(pitch_name)
        
        # Approximate beat and measure (assuming 120BPM, 4/4)
        # 1 beat = 0.5s, 1 measure = 2.0s
        beat = onset / 0.5
        measure = int(onset / 2.0) + 1
        duration = offset - onset
        
        note = NoteEvent(
            pitch=pitch_midi,
            onset=beat,           # mapping seconds directly to beat index
            offset=beat + (offset - onset) / 0.5,
            hand='right' if channel == 0 else 'left',
            measure=measure,
            beat=beat,
            finger=finger
        )
        
        if channel == 0:
            rh_notes.append(note)
        elif channel == 1:
            lh_notes.append(note)
            
    # Sort by onset time
    rh_notes.sort(key=lambda x: x.beat)
    lh_notes.sort(key=lambda x: x.beat)
    
    return rh_notes, lh_notes


def evaluate_pig_dataset(pig_dir: str):
    """Run pipeline on all RH sequences in the dataset."""
    fingering_dir = os.path.join(pig_dir, "FingeringFiles")
    
    if not os.path.exists(fingering_dir):
        print(f"Error: {fingering_dir} not found.")
        return
        
    txt_files = sorted([f for f in os.listdir(fingering_dir) if f.endswith('.txt')])
    
    pipeline = PhraseAwareFingering(use_motif_engine=True)
    
    total_notes = 0
    correct_matches = 0
    file_scores = []
    
    print(f"Evaluating {len(txt_files)} files from PIG dataset...")
    
    for filename in txt_files[:20]: # Start with first 20 for speed
        filepath = os.path.join(fingering_dir, filename)
        rh, lh = parse_pig_file(filepath)
        
        if not rh:
            continue
            
        # Get ground truth
        gt_fingers = [n.finger for n in rh]
        
        # Predict
        # Clear fingerings before prediction
        for n in rh:
            n.finger = None
            
        pred_fingers = pipeline.run(rh, companion_notes=lh)
        
        # Compare
        match_count = 0
        valid_notes = 0
        for p, g in zip(pred_fingers, gt_fingers):
            if g is not None and g != 0: # 0 or None means unannotated
                valid_notes += 1
                if p == g:
                    match_count += 1
                    
        if valid_notes > 0:
            acc = match_count / valid_notes
            file_scores.append(acc)
            total_notes += valid_notes
            correct_matches += match_count
            print(f"{filename[:15]:<15} : Match Rate = {acc*100:5.1f}% ({match_count}/{valid_notes})")
            
    if total_notes > 0:
        overall_acc = correct_matches / total_notes
        print(f"\nOVERALL RESULT (first 20 files):")
        print(f"Total valid RH notes : {total_notes}")
        print(f"Correct predictions  : {correct_matches}")
        print(f"Overall Match Rate   : {overall_acc*100:.2f}%")
        print(f"Average File Acc     : {np.mean(file_scores)*100:.2f}%")

if __name__ == '__main__':
    evaluate_pig_dataset('/Users/lap02459/AutomaticPianoFingering/PianoFingeringDataset_v1.2')
