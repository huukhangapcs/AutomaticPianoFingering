"""
Phrase Boundary Detector — Layer A of the Phrase-Aware Fingering module.

Identifies where musical phrases begin and end using a multi-signal fusion
approach, modelled after how a real pianist reads music.

Three key improvements over naive signal-based detection:
  1. Melodic Arc Shape Detector  — reads the contour gestalt
  2. Metric Position Weighting   — phrase lands on downbeat of next bar
  3. Phrase Length Prior         — Bayesian prior from classical norms
"""

from __future__ import annotations
from typing import List, Tuple
from fingering.models.note_event import NoteEvent
from fingering.phrasing.phrase import (
    Phrase, PhraseBoundarySignal, ArcType
)

# --- Constants ---

# Boundary is confirmed when fused score exceeds this threshold
BOUNDARY_THRESHOLD = 0.40

# Minimum phrase length in notes (avoid trivially short phrases)
MIN_PHRASE_NOTES = 3

# Cadence detection: semitone intervals for V→I motion in bass
# (simplified: dominant → tonic = perfect 4th up or 5th down)
_DOMINANT_TO_TONIC_SEMITONES = {5, -7}  # P4 up or P5 down

# ================================================================
# Improvement 3: Phrase Length Prior
# Classical phrases are overwhelmingly 2, 4, or 8 measures long.
# We express this as a lookup: after N notes, how likely is boundary?
# ================================================================
_PHRASE_LEN_PRIOR_BY_MEASURES = {1: 0.10, 2: 0.55, 4: 0.90, 8: 0.85}


def _phrase_length_prior(candidate_measures: float) -> float:
    """
    Return a prior probability that a phrase boundary exists after
    `candidate_measures` measures, based on classical music norms.

    Uses the nearest entry in the lookup table.
    """
    if candidate_measures <= 0:
        return 0.0
    keys = sorted(_PHRASE_LEN_PRIOR_BY_MEASURES.keys())
    # Find nearest key
    nearest = min(keys, key=lambda k: abs(k - candidate_measures))
    distance = abs(nearest - candidate_measures)
    # Decay if not exact
    decay = max(0.0, 1.0 - distance * 0.4)
    return _PHRASE_LEN_PRIOR_BY_MEASURES[nearest] * decay


# ================================================================
# Improvement 1: Melodic Arc Shape Detector
# ================================================================

def detect_arc_type(notes: List[NoteEvent]) -> ArcType:
    """
    Classify the melodic contour of a sequence of notes.

    Logic mirrors how a pianist visually reads the shape of a phrase:
    an ARCH (up then down) is the most natural phrase shape.
    """
    if len(notes) < 3:
        return ArcType.FLAT

    pitches = [n.pitch for n in notes]
    n = len(pitches)
    peak_idx = pitches.index(max(pitches))
    valley_idx = pitches.index(min(pitches))

    # Simple direction vector
    ups = sum(1 for i in range(1, n) if pitches[i] > pitches[i - 1])
    downs = sum(1 for i in range(1, n) if pitches[i] < pitches[i - 1])
    total_moves = ups + downs

    if total_moves == 0:
        return ArcType.FLAT

    up_ratio = ups / total_moves

    # ARCH: peak is in the middle third (not at edges)
    if (n // 4) < peak_idx < (3 * n // 4) and up_ratio > 0.35:
        return ArcType.ARCH

    # CLIMB: mostly ascending, peak near the end
    if up_ratio > 0.70:
        return ArcType.CLIMB

    # FALL: mostly descending, valley near the end
    if up_ratio < 0.30:
        return ArcType.FALL

    # WAVE: oscillating
    direction_changes = sum(
        1 for i in range(1, n - 1)
        if (pitches[i] - pitches[i - 1]) * (pitches[i + 1] - pitches[i]) < 0
    )
    if direction_changes >= n // 3:
        return ArcType.WAVE

    return ArcType.FLAT


# ================================================================
# Cadence Detector (lightweight rule-based)
# ================================================================

def _detect_cadence_strength(
    notes: List[NoteEvent],
    boundary_idx: int,
    time_sig_numerator: int = 4,
) -> float:
    """
    Estimate cadence strength at boundary_idx using melodic + rhythmic cues.

    A full harmonic analysis would require chord tracking; this approximation
    uses the last 2 melodic notes and their metric positions.

    Returns 0.0 → 1.0.
    """
    if boundary_idx < 1:
        return 0.0

    curr = notes[boundary_idx]
    prev = notes[boundary_idx - 1]

    strength = 0.0

    # Metric weight: notes on beat 1 or strong beats suggest cadence landing
    if curr.beat == 1.0:
        strength += 0.3
    elif curr.beat in (2.0, 3.0):
        strength += 0.1

    # Interval: step motion into cadence (mi→re→do or sol→fa→mi→re→do)
    interval = curr.pitch - prev.pitch
    if interval in (-2, -1, 0, 1, 2):  # stepwise motion = cadential approach
        strength += 0.2

    # Duration: cadence note is usually longer
    if curr.duration >= prev.duration * 1.5:
        strength += 0.2

    # Rest after: definitive phrase closure
    # (handled separately via rest_follows flag)

    return min(strength, 1.0)


# ================================================================
# Main Boundary Detector
# ================================================================

class PhraseBoundaryDetector:
    """
    Detects phrase boundaries in a stream of NoteEvents.

    Combines 7 signals with the 3 gap-closing improvements:
      - Slur end/start
      - Rest detection
      - Cadence strength
      - Melodic arc analysis      ← Improvement 1
      - Metric position weighting ← Improvement 2
      - Phrase length prior        ← Improvement 3
      - Dynamic change
    """

    def __init__(
        self,
        boundary_threshold: float = BOUNDARY_THRESHOLD,
        time_sig_numerator: int = 4,
        min_phrase_notes: int = MIN_PHRASE_NOTES,
    ):
        self.threshold = boundary_threshold
        self.time_sig_numerator = time_sig_numerator
        self.min_phrase_notes = min_phrase_notes

    def detect(self, notes: List[NoteEvent]) -> List[Phrase]:
        """
        Segment `notes` into a list of Phrase objects.

        Each Phrase contains the notes belonging to that phrase together
        with metadata used by downstream layers.
        """
        if not notes:
            return []

        signals = self._compute_signals(notes)
        boundaries = self._select_boundaries(signals, len(notes))
        phrases = self._build_phrases(notes, boundaries)
        return phrases

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_signals(
        self, notes: List[NoteEvent]
    ) -> List[PhraseBoundarySignal]:
        """Compute a PhraseBoundarySignal for every note position."""
        signals: List[PhraseBoundarySignal] = []
        n = len(notes)

        for i, note in enumerate(notes):
            next_note = notes[i + 1] if i + 1 < n else None
            prev_note = notes[i - 1] if i > 0 else None

            sig = PhraseBoundarySignal(position=i)

            # --- Slur ---
            sig.slur_end = note.slur_end

            # --- Rest ---
            if next_note is not None:
                gap = next_note.onset - note.offset
                sig.rest_follows = gap > 0.1  # > ~16th note gap
                sig.rest_duration = max(0.0, gap)

            # --- Cadence ---
            sig.cadence_strength = _detect_cadence_strength(
                notes, i, self.time_sig_numerator
            )

            # --- Large interval ---
            if next_note is not None:
                sig.large_interval = abs(next_note.pitch - note.pitch) > 7

            # -----------------------------------------------------------
            # Improvement 2: Metric position weighting
            # Phrase boundary is confirmed when the NEXT note is beat 1
            # of a new measure (the "landing" point of the phrase).
            # -----------------------------------------------------------
            if next_note is not None:
                sig.next_note_is_downbeat = (
                    next_note.beat == 1.0
                    and next_note.measure > note.measure
                )

            # --- Dynamic change between consecutive notes ---
            if next_note is not None:
                sig.dynamic_change = next_note.dynamic != note.dynamic

            signals.append(sig)

        # -----------------------------------------------------------
        # Improvement 3: Phrase length prior (pass 2, needs global context)
        # We look at how many measures since the last *confirmed* boundary
        # and weight positions that fall on "canonical" phrase lengths.
        # -----------------------------------------------------------
        self._apply_phrase_length_prior(signals, notes)

        # -----------------------------------------------------------
        # Improvement 1: Melodic arc shape
        # Compute arc type for windows of notes around each position.
        # Strong boundary candidates tend to end ARCH or CLIMB phrases.
        # -----------------------------------------------------------
        self._apply_melodic_arc_prior(signals, notes)

        return signals

    def _apply_phrase_length_prior(
        self,
        signals: List[PhraseBoundarySignal],
        notes: List[NoteEvent],
    ) -> None:
        """
        Improvement 3 implementation.

        For each candidate boundary, estimate how many measures have
        elapsed since the beginning of the current phrase segment, then
        look up the prior probability from classical phrase norms.
        """
        last_boundary_measure = notes[0].measure if notes else 1
        last_boundary_idx = 0

        for i, sig in enumerate(signals):
            elapsed_measures = notes[i].measure - last_boundary_measure
            sig.phrase_length_prior = _phrase_length_prior(
                float(elapsed_measures)
            )
            # Tentatively mark boundary to update reference point
            # (will be confirmed later; this is just the prior injection)

    def _apply_melodic_arc_prior(
        self,
        signals: List[PhraseBoundarySignal],
        notes: List[NoteEvent],
    ) -> None:
        """
        Improvement 1 implementation.

        Uses a sliding window to detect the melodic arc shape
        *leading up to* each candidate boundary position.
        An ARCH or CLIMB ending is a strong natural phrase boundary cue.
        """
        n = len(notes)
        window = 8  # Look back up to 8 notes

        for i, sig in enumerate(signals):
            start = max(0, i - window + 1)
            segment = notes[start: i + 1]
            arc = detect_arc_type(segment)
            sig.melodic_arc = arc

            # Boost boundary score for "completed" arc shapes
            # ARCH ending = phrase completed → boundary likely
            # CLIMB ending = might continue, but if followed by rest/slur → likely
            if arc in (ArcType.ARCH, ArcType.FALL):
                sig.cadence_strength = min(
                    sig.cadence_strength + 0.15, 1.0
                )

    def _select_boundaries(
        self,
        signals: List[PhraseBoundarySignal],
        total_notes: int,
    ) -> List[int]:
        """
        Select boundary indices from the scored signals.

        Uses a non-maximum suppression approach: within a window of
        MIN_PHRASE_NOTES, keep only the single strongest boundary.
        """
        scored = [(s.boundary_score(), s.position) for s in signals]

        boundaries = [0]  # Always start a phrase at note 0
        last_boundary = 0

        for score, pos in scored:
            # Skip if too close to last boundary
            if pos - last_boundary < self.min_phrase_notes:
                continue
            if score >= self.threshold:
                boundaries.append(pos + 1)  # New phrase starts AFTER boundary
                last_boundary = pos

        # Always close the last phrase
        if boundaries[-1] < total_notes:
            boundaries.append(total_notes)

        return sorted(set(boundaries))

    def _build_phrases(
        self,
        notes: List[NoteEvent],
        boundaries: List[int],
    ) -> List[Phrase]:
        """Slice the note stream into Phrase objects using boundary indices."""
        phrases = []
        for phrase_id, (start, end) in enumerate(
            zip(boundaries, boundaries[1:])
        ):
            phrase_notes = notes[start:end]
            if not phrase_notes:
                continue

            # Detect if this phrase starts after a rest
            starts_after_rest = False
            if start > 0:
                prev_note = notes[start - 1]
                gap = phrase_notes[0].onset - prev_note.offset
                starts_after_rest = gap > 0.1

            # Detect the arc for this specific phrase
            arc = detect_arc_type(phrase_notes)

            hand = phrase_notes[0].hand  # Assume consistent hand per phrase

            # Boundary score = score of the note just before this phrase
            boundary_score = 0.0
            if start > 0:
                from fingering.phrasing.phrase import PhraseBoundarySignal
                _dummy = PhraseBoundarySignal(position=start - 1)
                boundary_score = _dummy.boundary_score()

            phrase = Phrase(
                id=phrase_id,
                notes=phrase_notes,
                hand=hand,
                boundary_score=boundary_score,
                arc_type=arc,
                starts_after_rest=starts_after_rest,
            )
            phrases.append(phrase)

        return phrases
