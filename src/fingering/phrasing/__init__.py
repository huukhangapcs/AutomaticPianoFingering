"""Public API for the phrasing sub-package."""
from fingering.phrasing.phrase import Phrase, PhraseIntent, ArcType, PhraseBoundarySignal
from fingering.phrasing.boundary_detector import PhraseBoundaryDetector, detect_arc_type
from fingering.phrasing.intent_analyzer import PhraseIntentAnalyzer
from fingering.phrasing.phrase_dp import PhraseScopedDP
from fingering.phrasing.cross_phrase import CrossPhraseStitch
from fingering.phrasing.pipeline import PhraseAwareFingering

__all__ = [
    "Phrase", "PhraseIntent", "ArcType", "PhraseBoundarySignal",
    "PhraseBoundaryDetector", "detect_arc_type",
    "PhraseIntentAnalyzer",
    "PhraseScopedDP",
    "CrossPhraseStitch",
    "PhraseAwareFingering",
]
