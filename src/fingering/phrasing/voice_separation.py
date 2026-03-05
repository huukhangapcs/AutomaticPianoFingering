from typing import List
from fingering.models.note_event import NoteEvent

def extract_melody_stream(notes: List[NoteEvent]) -> List[NoteEvent]:
    """
    Extracts the melody voice (highest pitch) from a polyphonic sequence of notes.
    If multiple notes share the exact same onset time, the one with the highest pitch is kept.
    This serves as Layer 2 (Voice Separation) for Phrase Segmentation v2.
    """
    if not notes:
        return []
        
    melody = []
    # Group notes by onset time
    current_onset = notes[0].onset
    current_group = []
    
    for note in notes:
        if abs(note.onset - current_onset) < 0.001:  # Same onset chunk
            current_group.append(note)
        else:
            # Add the highest note from the previous group to melody stream
            highest_note = max(current_group, key=lambda n: n.pitch)
            melody.append(highest_note)
            
            # Start new group
            current_onset = note.onset
            current_group = [note]
            
    # Add the last group
    if current_group:
        highest_note = max(current_group, key=lambda n: n.pitch)
        melody.append(highest_note)
        
    return melody
