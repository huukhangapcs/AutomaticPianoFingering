import sys
import os
import xml.etree.ElementTree as ET

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from fingering.io.musicxml_reader import MusicXMLReader
from fingering.phrasing.pipeline import PhraseAwareFingering

def inject_phrases_to_musicxml(input_xml: str, output_xml: str):
    print(f"Reading {input_xml}...")
    reader = MusicXMLReader()
    rh, lh, tonic, mode = reader.parse_grand_staff_with_key(input_xml)
    
    paf = PhraseAwareFingering(use_motif_engine=True, tonic_pc=tonic)
    phrases = paf.get_phrases(rh, companion_notes=lh)
    
    print(f"Detected {len(phrases)} phrases. Injecting into XML...")
    
    # Parse the tree to modify
    tree = ET.parse(input_xml)
    root = tree.getroot()
    
    # We will inject annotations at the start measure of each phrase
    # For grand staff, we inject it into the first part (RH)
    part = root.find('.//part')
    if part is None:
        print("No part found in XML")
        return
        
    for i, p in enumerate(phrases):
        if not p.notes:
            continue
        start_m = p.notes[0].measure
        end_m = p.notes[-1].measure
        role = getattr(p, 'period_role', '')
        intent = p.intent.name if hasattr(p, 'intent') and p.intent else ''
        
        # Build annotation text
        label = f"[P{i+1}: {end_m - start_m + 1}m] {role}"
        
        # Find the measure
        # MusicXML measure number could be string or int. 
        # But we'll just look for <measure number="X"> matching our start_m
        for measure_el in part.findall('measure'):
            if measure_el.get('number') == str(start_m):
                # Create a direction element
                # <direction placement="above">
                #   <direction-type>
                #     <words font-weight="bold" font-size="11" color="#FF0000">label</words>
                #   </direction-type>
                # </direction>
                
                direction = ET.Element("direction", placement="above")
                dtype = ET.SubElement(direction, "direction-type")
                words = ET.SubElement(dtype, "words", {"font-weight": "bold", "font-size": "11", "color": "#0000FF"})
                words.text = label
                
                # Insert at the beginning of the measure
                measure_el.insert(0, direction)
                break
                
    # Also add fingerings for RH to prove it works
    # Just a simple hack: we need to map NoteEvent back to XML note, but it's complex since we don't have pointers.
    # For now, just the phrase boundaries are enough for visual verification.
    
    tree.write(output_xml, encoding="utf-8", xml_declaration=True)
    print(f"Exported to {output_xml}")

if __name__ == '__main__':
    in_file = 'test_file/FN7ALfpGxiI.musicxml'
    out_file = 'test_file/FN7ALfpGxiI_Phrased.musicxml'
    if len(sys.argv) > 2:
        in_file = sys.argv[1]
        out_file = sys.argv[2]
    inject_phrases_to_musicxml(in_file, out_file)
