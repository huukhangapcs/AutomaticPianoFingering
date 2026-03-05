import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from fingering.io.musicxml_reader import MusicXMLReader
from fingering.phrasing.pipeline import PhraseAwareFingering
from dataclasses import asdict

def evaluate_phrasing(musicxml_path='test_file/FN7ALfpGxiI.musicxml'):
    try:
        reader = MusicXMLReader()
        rh, lh, tonic_pc, mode = reader.parse_grand_staff_with_key(musicxml_path)
    except Exception as e:
        print(f"Error parsing {musicxml_path}: {e}")
        return

    print(f"--- Evaluating Phrasing on {os.path.basename(musicxml_path)} ---")
    print(f"Key detected: Tonic PC = {tonic_pc}, Mode = {mode}")
    print(f"Notes: {len(rh)} RH, {len(lh)} LH")

    paf = PhraseAwareFingering(
        use_motif_engine=True,
        tonic_pc=tonic_pc,
        max_phrase_measures=12 # Syncing with standard phrase threshold
    )

    try:
        phrases = paf.get_phrases(rh, companion_notes=lh)
        
        print("\n--- Phrase Segmentation Output ---")
        print(f"Total Phrases Detected: {len(phrases)}")
        
        lengths = []
        for i, p in enumerate(phrases):
            if not p.notes:
                continue
            start_m = p.notes[0].measure
            end_m = p.notes[-1].measure
            length_m = end_m - start_m + 1
            lengths.append(length_m)
            
            period_label = getattr(p, 'period_role', '')
            flag = "⚠️ L" if length_m > 12 else "⚠️ S" if length_m <= 2 else "  "
            intent = f"{p.intent.name if hasattr(p, 'intent') and p.intent else '?'}"
            
            print(f"P{i+1:<2} | m.{start_m:03d}-m.{end_m:03d} ({length_m:2d}m) | {flag} | {period_label[:10]:<10} | {intent}")
            
        print("\n--- Phrasing Statistics ---")
        avg_len = sum(lengths)/len(lengths) if lengths else 0
        print(f"Average length: {avg_len:.1f} measures")
        print(f"Max length: {max(lengths) if lengths else 0} measures")
        print(f"Min length: {min(lengths) if lengths else 0} measures")
        
        # Pianist alignment metrics
        standard_4m = sum(1 for l in lengths if l in [4, 8, 16])
        print(f"Standard Classical Lengths (4m/8m): {standard_4m}/{len(lengths)} ({standard_4m/len(lengths)*100:.1f}%)")
        
        print("\n--- Diagnostic Score ---")
        score = 0
        if 3.5 <= avg_len <= 8.5:
            score += 3  # Normal average phrase
            print("+3: Average phrase length is normative (3.5 - 8.5m)")
        else:
            print(f"0/3: Average phrase length skewed ({avg_len:.1f}m)")
            
        if standard_4m/len(lengths) > 0.4:
            score += 3
            print(f"+3: High percentage of 4m/8m/16m classical phrases ({standard_4m/len(lengths)*100:.1f}%)")
        elif standard_4m/len(lengths) > 0.2:
            print(f"+1: Moderate classical phrase structures detected ({standard_4m/len(lengths)*100:.1f}%)")
            score += 1
        else:
            print(f"0/3: Low standard classical lengths ({standard_4m/len(lengths)*100:.1f}%)")
            
        over_12m = sum(1 for l in lengths if l > 12)
        if over_12m == 0:
            score += 4
            print("+4: Strict compliance with max_phrase_measures (12m)")
        else:
            print(f"+{max(0, 4-over_12m)}/4: {over_12m} phrases exceed max_phrase_measures constraint")
            score += max(0, 4-over_12m)
            
        print(f"=> Syntactic Segmentation Score: {score}/10")
        
        # Analyze the long phrase separately
        if over_12m > 0:
            print("\n--- Deep Dive: Why are phrases violating constraints? ---")
            for p in phrases:
                start_m = p.notes[0].measure
                end_m = p.notes[-1].measure
                if end_m - start_m + 1 > 12:
                    print(f"Analyzing P{phrases.index(p)+1} (m.{start_m}-m.{end_m}):")
                    # Check motif boundaries
            print("Note: In classical music, a 16m+ pedal point or cadenza is musically correct as a single phrase. A 10/10 requires context-aware overriding of the cap.")

    except Exception as e:
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    target = 'test_file/FN7ALfpGxiI.musicxml'
    if len(sys.argv) > 1:
        target = sys.argv[1]
    evaluate_phrasing(target)
