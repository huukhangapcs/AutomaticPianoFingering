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
# (Legacy: used in v1. Kept here for compatibility if needed elsewhere)
BOUNDARY_THRESHOLD = 0.40

# Phrase Segmentation v2 Parameters
MIN_PHRASE_MEASURES = 2
MAX_PHRASE_MEASURES = 12

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
# Fix 2: Agogic Accent
# A note that is significantly longer than its neighbors signals
# a natural breath point / phrase ending for the pianist.
# ================================================================

def _agogic_strength(notes: List[NoteEvent], idx: int, window: int = 4) -> float:
    """
    Return the agogic accent strength at `idx` (0–1).

    A note is an agogic accent if its duration is notably longer
    than surrounding notes in a local context window.
    """
    if not notes:
        return 0.0
    local = notes[max(0, idx - window): min(len(notes), idx + window + 1)]
    if len(local) < 2:
        return 0.0

    curr_dur = notes[idx].duration
    local_durs = [n.duration for n in local if n is not notes[idx]]
    if not local_durs:
        return 0.0

    avg = sum(local_durs) / len(local_durs)
    if avg <= 0:
        return 0.0

    ratio = curr_dur / avg
    # Require at least 3x to be "strong" agogic:
    # half (2 beats) vs quarter (1 beat) = 2x, fires too often in classical music
    # dotted half (3 beats) vs quarter = 3x → clear phrase-end breath
    if ratio >= 3.0:
        return 1.0
    elif ratio >= 2.0:
        return 0.5
    elif ratio >= 1.5:
        return 0.2
    return 0.0


# ================================================================
# Fix 3: Harmonic Cadence via Bass Movement
# Track the lowest pitch per onset group across a window.
# A bass movement by P4 (5 semitones) or P5 (7 semitones)
# is characteristic of V→I or IV→I cadential motion.
# ================================================================

_CADENTIAL_BASS_INTERVALS = {
    5:  0.8,   # Perfect 4th up   (dominant → tonic)
    -7: 0.8,   # Perfect 5th down (dominant → tonic)
    -5: 0.6,   # Perfect 4th down (subdominant → tonic)
    7:  0.6,   # Perfect 5th up   (subdominant below tonic)
    # Step motion excluded: too common in melody, causes over-segmentation
}


def _harmonic_cadence_strength(
    notes: List[NoteEvent],
    idx: int,
    window: int = 8,
) -> float:
    """
    Estimate harmonic cadence strength at `idx` by analyzing bass
    movement from the previous local "bass note" to the note at idx.

    Only considers left-hand notes (hand='left') when both staves are
    present, otherwise uses all notes.
    """
    if idx < 1:
        return 0.0

    # Get bass notes: lowest pitch within each onset cluster
    # in the window before idx
    lh = [n for n in notes[max(0, idx - window): idx + 1]
          if n.hand == 'left'] or notes[max(0, idx - window): idx + 1]

    if len(lh) < 2:
        return 0.0

    # Simple approach: compare pitch of the note at idx to note window//2 back
    prev_bass = min(lh[:-1], key=lambda n: n.pitch)
    curr_note = notes[idx]
    interval = curr_note.pitch - prev_bass.pitch
    # Normalize to within an octave
    interval = interval % 12 if interval > 0 else -((-interval) % 12)
    return _CADENTIAL_BASS_INTERVALS.get(interval, 0.0)


# ================================================================
# Fix 2b: Melodic Resolution Signal
# A stepwise descent into a note with longer duration (the "arrival")
# suggests phrase resolution.
# ================================================================

def _melodic_resolution_strength(
    notes: List[NoteEvent],
    idx: int,
    agogic: float,
) -> float:
    """
    Return melodic resolution strength at `idx`.

    Combines:
    - Stepwise (half or whole step) motion into `idx`
    - The note at `idx` is longer than its predecessor (agogic)
    - Descending motion (resolution feeling)
    """
    if idx < 1 or agogic <= 0:
        return 0.0

    prev = notes[idx - 1]
    curr = notes[idx]
    interval = curr.pitch - prev.pitch

    # Descending stepwise motion (-1 or -2 semitones = half or whole step down)
    if interval in (-1, -2) and agogic >= 0.3:
        return min(0.8, agogic + 0.2)
    # Ascending step resolution (uncommon but valid: leading tone → tonic)
    if interval in (1, 2) and agogic >= 0.5:
        return agogic * 0.5
    return 0.0


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
        max_phrase_measures: int = 4,
    ):
        self.threshold = boundary_threshold
        self.time_sig_numerator = time_sig_numerator
        self.min_phrase_notes = min_phrase_notes
        self.max_phrase_measures = max_phrase_measures

    def detect(self, notes: List[NoteEvent]) -> List[Phrase]:
        """
        Public API: run boundary detection on a sequence of NoteEvents.
        
        Applies Layer 2 Voice Separation so that signals are scored purely
        against the top melody line, avoiding noise from internal polyphony.
        """
        if not notes:
            return []

        from fingering.phrasing.voice_separation import extract_melody_stream
        melody_notes = extract_melody_stream(notes)

        # Calculate boundary candidates by analyzing ONLY the melody notes
        signals = self._compute_signals(melody_notes)
        
        # Run Viterbi on the melody to find the optimal melody note splits
        melody_boundaries = self._select_boundaries(
            signals, len(melody_notes), melody_notes
        )
        
        # Map melody boundary indices back to the original `notes` array
        original_boundaries = [0]
        # Ignore the first dummy index 0, and the last len(melody)
        for i in melody_boundaries:
            if 0 < i < len(melody_notes):
                target_onset = melody_notes[i].onset
                # Find the first raw note with >= target_onset
                for orig_idx, orig_note in enumerate(notes):
                    if orig_note.onset >= target_onset:
                        original_boundaries.append(orig_idx)
                        break
        original_boundaries.append(len(notes))
        original_boundaries = sorted(set(original_boundaries))

        return self._build_phrases(notes, original_boundaries)

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

            sig = PhraseBoundarySignal(
                position=i,
                note_duration=note.duration
            )

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

            # --- Melodic Leap Compensation ---
            # Phrase ends after a large leap (>= P4, 5 semitones) that is followed by a stepwise resolution (<= 2 semitones)
            if prev_note is not None and next_note is not None:
                leap = abs(note.pitch - prev_note.pitch)
                resolution = abs(next_note.pitch - note.pitch)
                if leap >= 5 and resolution <= 2:
                    sig.melodic_leap_compensation = 1.0

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

            # --- Fix 2: Agogic accent ---
            sig.agogic_accent = _agogic_strength(notes, i)

            # --- Fix 2b: Melodic resolution ---
            sig.melodic_resolution = _melodic_resolution_strength(
                notes, i, sig.agogic_accent
            )

            # --- Fix 3: Harmonic cadence via bass movement ---
            sig.harmonic_cadence = _harmonic_cadence_strength(notes, i)

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
        Legacy function (v1). Kept for external compatibility.
        In v2 (Viterbi), phrase length prior is computed dynamically 
        during the DP edge transition phase, not statically on the signal.
        """
        pass

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
        notes: List[NoteEvent] = None,
    ) -> List[int]:
        """
        [Phrase Segmentation v2]
        Select boundary indices using Viterbi Dynamic Programming.
        Maximizes global phrase score (Boundary Probability + Length Prior).
        """
        if total_notes == 0:
            return []
            
        return self._viterbi_segmentation(signals, notes)

    def _viterbi_segmentation(
        self,
        signals: List[PhraseBoundarySignal],
        notes: List[NoteEvent],
    ) -> List[int]:
        """
        Global optimization: find the sequence of boundaries that maximizes
        the total sum of (boundary_probability + phrase_length_prior).
        
        State space: dp[i] = max score of a segmentation ending EXACTLY at note index i.
        Transition: dp[j] = max_{i} (dp[i] + Score(i -> j))
        
        Where i is the previous boundary, and j is the current boundary candidate.
        Subject to: bounds on minimum and maximum phrase measures.
        """
        n = len(notes)
        if n == 0:
            return []
            
        # dp[i] stores the maximum score for a segmentation ending exactly at note index i (-1 to n-1)
        # Note: index -1 represents the start of the piece (before note 0)
        dp = { -1: 0.0 }
        backtrack = { -1: None }
        
        # Precompute boundary probabilities
        # p_bound[j] = probability [0..1] that j is a phrase end
        p_bound = { s.position: s.boundary_score() for s in signals }
        # Force the last note to be a valid boundary with high probability
        p_bound[n - 1] = 1.0

        for j in range(0, n):
            max_score = -float('inf')
            best_prev = None
            
            # Boundary score at j determines the intrinsic value of cutting here
            # For the last note, we don't penalize it if no signals exist.
            prob_j = p_bound.get(j, 0.0)
            
            # We only evaluate cutting at j if the raw signal isn't essentially zero
            # (unless it's the very last note)
            if prob_j < 0.05 and j != n - 1:
                continue

            j_measure = notes[j].measure
            
            # Search backwards for the optimal previous boundary i
            # Lookback bound: max_phrase_measures
            for i in dp.keys():
                # Measure constraint check
                if i == -1:
                    i_measure = notes[0].measure
                    i_onset = 0.0
                else:
                    i_measure = notes[i].measure
                    i_onset = notes[i].onset
                    
                measures_length = j_measure - i_measure + 1
                
                # Validation limits
                if measures_length > self.max_phrase_measures:
                    continue  # i is too far back
                    
                if measures_length < self.min_phrase_notes and j != n - 1:
                    # Too short, skip (unless forcing close at piece end)
                    continue
                    
                # Compute edge score: Score(i -> j)
                # 1. How good of a boundary is j?
                edge_score = prob_j
                
                # 2. How good is the phrase length (i -> j)?
                prior = _phrase_length_prior(measures_length)
                
                # Combine them (equal weighting for now, tunable)
                edge_score += prior
                
                # Calculate total journey score
                candidate_score = dp[i] + edge_score
                
                if candidate_score > max_score:
                    max_score = candidate_score
                    best_prev = i

            if best_prev is not None:
                dp[j] = max_score
                backtrack[j] = best_prev

        # If dp couldn't reach the end (n-1) due to constraint gaps, fallback
        if n - 1 not in dp:
            # Fallback to greedy if Viterbi fails to span the entire piece
            return self._fallback_greedy(signals, n)

        # Backtrack to find the optimal boundary path
        curr = n - 1
        boundaries = []
        while curr is not None and curr != -1:
            boundaries.append(curr)
            curr = backtrack[curr]
            
        boundaries.reverse()
        
        # Format: boundary indices denote the first note of the *new* phrase, 
        # so if 'idx' is a phrase end, the split list needs (idx + 1)
        splits = [0]
        for b in boundaries:
            if b + 1 < n:
                splits.append(b + 1)
        splits.append(n)
        
        return sorted(set(splits))

    def _fallback_greedy(self, signals: List[PhraseBoundarySignal], total_notes: int) -> List[int]:
        """Safety fallback if Viterbi graph disconnects."""
        boundaries = [0]
        for s in signals:
            if s.boundary_score() >= 0.5:
                boundaries.append(s.position + 1)
        boundaries.append(total_notes)
        return sorted(set(boundaries))

    def _force_measure_boundaries(
        self,
        boundaries: List[int],
        notes: List[NoteEvent],
        total_notes: int,
    ) -> List[int]:
        """
        For any phrase segment longer than max_phrase_measures, insert
        forced splits at every max_phrase_measures interval, aligned to
        the nearest measure boundary in the note stream.
        """
        result = list(boundaries)
        pairs = list(zip(sorted(result), sorted(result)[1:]))

        for start_idx, end_idx in pairs:
            if end_idx > total_notes or start_idx >= end_idx:
                continue
            seg_notes = notes[start_idx:end_idx]
            if not seg_notes:
                continue
            seg_measures = seg_notes[-1].measure - seg_notes[0].measure + 1
            if seg_measures <= self.max_phrase_measures:
                continue

            # Need to split — find measure boundaries within segment
            split_every = self.max_phrase_measures
            start_measure = seg_notes[0].measure
            target_measures = [
                start_measure + split_every * k
                for k in range(1, int(seg_measures / split_every) + 1)
            ]
            for target_m in target_measures:
                # Find note index closest to start of target_measure
                for j in range(start_idx, end_idx):
                    if notes[j].measure >= target_m:
                        if j not in result:
                            result.append(j)
                        break

        return sorted(set(result))

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

            # The boundary score logic has moved entirely to Viterbi
            # For Phrase object storage, we can reconstruct the raw logit probability
            boundary_score = 0.0
            if start > 0:
                from fingering.phrasing.phrase import PhraseBoundarySignal
                _dummy = PhraseBoundarySignal(position=start - 1)
                # Find the actual signal in the pipeline, or default to 0
                # For simplicity, we'll leave it 0 since it isn't crucial downstream 
                pass

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

    def _build_phrases_from_indices(
        self,
        notes: List[NoteEvent],
        boundary_note_indices: List[int],
    ) -> List[Phrase]:
        """
        Adapter: convert PhraseSelector's flat boundary list to Phrase objects.

        PhraseSelector returns indices of the LAST note of each phrase segment.
        This converts them to (start, end) slice pairs for _build_phrases.

        Args:
            notes: full note stream
            boundary_note_indices: sorted list of note indices where phrases end

        Returns:
            List[Phrase], one per segment
        """
        if not notes:
            return []

        if not boundary_note_indices:
            # No boundaries → one big phrase
            return self._build_phrases(notes, [0, len(notes)])

        # Convert "last note of phrase" indices → start positions
        # A boundary at idx i means phrase ends at notes[i] → next phrase starts at i+1
        starts = [0]
        for idx in sorted(set(boundary_note_indices)):
            next_start = idx + 1
            if 0 < next_start < len(notes):
                starts.append(next_start)

        # Close with the total length
        starts.append(len(notes))

        return self._build_phrases(notes, sorted(set(starts)))
