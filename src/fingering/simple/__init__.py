"""
Simplified Piano Fingering — Physics-First Model.

3-module architecture:
  keyboard.py      (shared with core) — physical key positions
  hand_reset.py    (shared with core) — reset point detection
  fingering_dp.py  — Viterbi DP (shift + stretch + crossing + direction + OFOK)
  pipeline.py      — thin entry point
"""
from fingering.simple.pipeline import PianoFingering

__all__ = ['PianoFingering']
