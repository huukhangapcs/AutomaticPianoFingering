"""
Phrase-Aware Fingering Pipeline — main entry point.

Orchestrates all 4 layers:
  A: PhraseBoundaryDetector
  B: PhraseIntentAnalyzer
  C: PhraseScopedDP
  D: CrossPhraseStitch
"""
from __future__ import annotations
from typing import List, Optional

from fingering.models.note_event import NoteEvent
from fingering.phrasing.phrase import Phrase
from fingering.phrasing.boundary_detector import PhraseBoundaryDetector
from fingering.phrasing.intent_analyzer import PhraseIntentAnalyzer
from fingering.phrasing.phrase_dp import PhraseScopedDP
from fingering.phrasing.cross_phrase import CrossPhraseStitch


class PhraseAwareFingering:
    """
    High-level pipeline that produces finger assignments for a stream
    of NoteEvents by reading them as musical phrases.

    Example usage:
        paf = PhraseAwareFingering()
        notes = [...]     # list of NoteEvent
        fingering = paf.run(notes)
        # fingering[i] = finger (1–5) for notes[i]
    """

    def __init__(
        self,
        boundary_threshold: float = 0.40,
        time_sig_numerator: int = 4,
        min_phrase_notes: int = 3,
    ):
        self.detector  = PhraseBoundaryDetector(
            boundary_threshold=boundary_threshold,
            time_sig_numerator=time_sig_numerator,
            min_phrase_notes=min_phrase_notes,
        )
        self.analyzer  = PhraseIntentAnalyzer()
        self.dp_solver = PhraseScopedDP()
        self.stitcher  = CrossPhraseStitch()

    def run(self, notes: List[NoteEvent]) -> List[int]:
        """
        Full phrase-aware fingering pipeline.

        Returns a flat list of finger indices parallel to `notes`.
        """
        if not notes:
            return []

        # --- Layer A: Detect phrase boundaries ---
        phrases = self.detector.detect(notes)

        if not phrases:
            return [1] * len(notes)

        # --- Layer B: Analyze intent + tension arc for each phrase ---
        phrases = self.analyzer.analyze_all(phrases)

        # --- Layers C + D: solve each phrase with cross-phrase awareness ---
        all_fingering: List[int] = []
        prev_phrase: Optional[Phrase] = None
        prev_fingering: Optional[List[int]] = None

        for i, phrase in enumerate(phrases):
            # Layer D: compute stitch constraint from previous phrase
            stitch = None
            if prev_phrase is not None and prev_fingering is not None:
                stitch = self.stitcher.compute_constraint(
                    prev_phrase, phrase, prev_fingering
                )

            # Layer C: phrase-scoped DP
            fingering = self.dp_solver.solve(phrase, stitch_constraint=stitch)
            all_fingering.extend(fingering)

            prev_phrase    = phrase
            prev_fingering = fingering

        return all_fingering

    def run_and_annotate(self, notes: List[NoteEvent]) -> List[NoteEvent]:
        """
        Run the pipeline and write finger assignments back into NoteEvents.
        Returns the modified note list.
        """
        fingering = self.run(notes)
        for note, finger in zip(notes, fingering):
            note.finger = finger
        return notes

    def get_phrases(self, notes: List[NoteEvent]) -> List[Phrase]:
        """
        Expose detected + analyzed phrases for inspection/debugging.
        """
        phrases = self.detector.detect(notes)
        return self.analyzer.analyze_all(phrases)
