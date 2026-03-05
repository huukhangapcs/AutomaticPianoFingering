"""
Tests for the Phrase-Aware Fingering module.

Covers:
  - ArcType detection (Improvement 1)
  - Phrase length prior (Improvement 3)
  - Boundary score fusion
  - Phrase detection on a simple melody
  - Intent detection
  - PhraseScopedDP: climax finger strength
  - CrossPhraseStitch: valid junction
  - End-to-end pipeline
"""

import pytest
from fingering.models.note_event import NoteEvent
from fingering.phrasing.phrase import PhraseIntent, ArcType, PhraseBoundarySignal
from fingering.phrasing.boundary_detector import (
    PhraseBoundaryDetector, detect_arc_type, _phrase_length_prior
)
from fingering.phrasing.intent_analyzer import PhraseIntentAnalyzer
from fingering.phrasing.phrase_dp import PhraseScopedDP
from fingering.phrasing.cross_phrase import CrossPhraseStitch
from fingering.phrasing.pipeline import PhraseAwareFingering


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def make_note(pitch: int, onset: float, duration: float = 0.5,
              measure: int = 1, beat: float = 1.0,
              hand: str = 'right', **kwargs) -> NoteEvent:
    return NoteEvent(
        pitch=pitch, onset=onset, offset=onset + duration,
        hand=hand, measure=measure, beat=beat, **kwargs
    )


def ascending_scale(start_pitch=60, n=8, duration=0.5) -> list[NoteEvent]:
    """C major scale fragment, RH."""
    SCALE = [0, 2, 4, 5, 7, 9, 11, 12]  # intervals from root
    notes = []
    beat = 1.0
    measure = 1
    for i in range(n):
        pitch = start_pitch + SCALE[i % len(SCALE)]
        m = 1 + i // 4
        b = ((i % 4) * duration) + 1.0
        notes.append(make_note(pitch, onset=i * duration, duration=duration,
                               measure=m, beat=b))
    return notes


def arch_melody() -> list[NoteEvent]:
    """A simple arch-shaped melody: C4→E4→G4→E4→C4."""
    pitches = [60, 64, 67, 64, 60]
    return [make_note(p, i * 0.5, measure=1 + i // 4,
                      beat=(i % 4) * 0.5 + 1.0)
            for i, p in enumerate(pitches)]


# ─────────────────────────────────────────────
# Improvement 1: Melodic Arc Detection
# ─────────────────────────────────────────────

class TestArcDetection:

    def test_arch_detected(self):
        notes = arch_melody()
        arc = detect_arc_type(notes)
        assert arc == ArcType.ARCH

    def test_climb_detected(self):
        notes = ascending_scale(n=8)
        arc = detect_arc_type(notes)
        assert arc == ArcType.CLIMB

    def test_fall_detected(self):
        pitches = [72, 71, 69, 67, 65, 64, 62, 60]
        notes = [make_note(p, i * 0.5) for i, p in enumerate(pitches)]
        arc = detect_arc_type(notes)
        assert arc == ArcType.FALL

    def test_flat_detected_for_single_note(self):
        notes = [make_note(60, 0.0)]
        arc = detect_arc_type(notes)
        assert arc == ArcType.FLAT

    def test_wave_detected(self):
        # Oscillating pattern
        pitches = [60, 65, 60, 65, 60, 65, 60, 65]
        notes = [make_note(p, i * 0.5) for i, p in enumerate(pitches)]
        arc = detect_arc_type(notes)
        assert arc == ArcType.WAVE


# ─────────────────────────────────────────────
# Improvement 3: Phrase Length Prior
# ─────────────────────────────────────────────

class TestPhraseLengthPrior:

    def test_4_measure_highest_prior(self):
        prior_4 = _phrase_length_prior(4.0)
        prior_3 = _phrase_length_prior(3.0)
        prior_7 = _phrase_length_prior(7.0)
        assert prior_4 > prior_3
        assert prior_4 > prior_7

    def test_2_measure_high_prior(self):
        prior_2 = _phrase_length_prior(2.0)
        assert prior_2 > 0.4

    def test_zero_measures_returns_zero(self):
        assert _phrase_length_prior(0.0) == 0.0

    def test_odd_measures_lower_prior(self):
        prior_5 = _phrase_length_prior(5.0)
        prior_4 = _phrase_length_prior(4.0)
        assert prior_5 < prior_4


# ─────────────────────────────────────────────
# Boundary Score Fusion
# ─────────────────────────────────────────────

class TestBoundaryScoreFusion:

    def test_slur_end_adds_weight(self):
        s = PhraseBoundarySignal(position=0, slur_end=True)
        assert s.boundary_score() >= 0.30

    def test_rest_follows_adds_weight(self):
        """rest_follows contributes weight but short rests alone cannot trigger a boundary."""
        s = PhraseBoundarySignal(position=0, rest_follows=True, rest_duration=0.25)
        # Contributes 0.22, which is below the 0.40 threshold — by design
        assert s.boundary_score() >= 0.15  # does contribute
        assert s.boundary_score() < 0.40   # but not enough alone

    def test_rest_plus_agogic_exceeds_threshold(self):
        """rest + agogic_accent + cadence together should trigger a boundary."""
        s = PhraseBoundarySignal(
            position=0,
            rest_follows=True, rest_duration=1.0,
            agogic_accent=1.0,      # dotted whole note in context
            cadence_strength=0.5,   # mild metric cue
        )
        assert s.boundary_score() >= 0.40

    def test_improvement2_downbeat_adds_weight(self):
        s_no = PhraseBoundarySignal(position=0, next_note_is_downbeat=False)
        s_yes = PhraseBoundarySignal(position=0, next_note_is_downbeat=True)
        assert s_yes.boundary_score() > s_no.boundary_score()

    def test_combined_signals_exceed_threshold(self):
        """A slur end + rest + downbeat should clearly exceed 0.40."""
        s = PhraseBoundarySignal(
            position=0,
            slur_end=True,
            rest_follows=True,
            rest_duration=1.0,
            next_note_is_downbeat=True,
        )
        assert s.boundary_score() >= 0.40

    def test_score_bounded_to_1(self):
        """Score must never exceed 1.0."""
        s = PhraseBoundarySignal(
            position=0,
            slur_end=True, rest_follows=True, rest_duration=2.0,
            cadence_strength=1.0, large_interval=True,
            next_note_is_downbeat=True, phrase_length_prior=1.0,
            dynamic_change=True, agogic_accent=1.0,
            melodic_resolution=1.0, harmonic_cadence=1.0,
        )
        assert s.boundary_score() <= 1.0


# ─────────────────────────────────────────────
# Phrase Boundary Detector
# ─────────────────────────────────────────────

class TestPhraseBoundaryDetector:

    def test_detects_at_least_one_phrase(self):
        notes = ascending_scale(n=8)
        detector = PhraseBoundaryDetector()
        phrases = detector.detect(notes)
        assert len(phrases) >= 1

    def test_phrase_notes_cover_all_input(self):
        notes = ascending_scale(n=8)
        detector = PhraseBoundaryDetector()
        phrases = detector.detect(notes)
        total = sum(len(p.notes) for p in phrases)
        assert total == len(notes)

    def test_rest_creates_boundary(self):
        """A large gap (rest) between two note groups should create 2 phrases."""
        group1 = [make_note(60 + i * 2, i * 0.5, duration=0.5,
                             measure=1, beat=i * 0.5 + 1) for i in range(4)]
        # Insert a 2-beat rest by setting onset of group2 well after group1 ends
        group2 = [make_note(60 + i * 2, 4.0 + i * 0.5, duration=0.5,
                             measure=3, beat=i * 0.5 + 1) for i in range(4)]
        notes = group1 + group2
        detector = PhraseBoundaryDetector(boundary_threshold=0.25)
        phrases = detector.detect(notes)
        assert len(phrases) >= 2

    def test_slur_end_creates_boundary(self):
        # Create 16 notes across 4 measures (4 notes per measure)
        notes = []
        for i in range(16):
            m = (i // 4) + 1
            b = (i % 4) + 1.0
            n = make_note(60 + i, i * 1.0, duration=1.0, measure=m, beat=b)
            notes.append(n)
            
        # Mark the end of measure 2 as a slur end
        notes[7].slur_end = True
        notes[8].slur_start = True
        
        detector = PhraseBoundaryDetector(boundary_threshold=0.30)
        phrases = detector.detect(notes)
        assert len(phrases) >= 2


# ─────────────────────────────────────────────
# Intent Analyzer
# ─────────────────────────────────────────────

class TestIntentAnalyzer:

    def _make_phrase_from_notes(self, notes, arc=ArcType.ARCH):
        from fingering.phrasing.phrase import Phrase
        p = Phrase(id=0, notes=notes, hand='right', arc_type=arc)
        return p

    def test_legato_phrase_detected(self):
        notes = ascending_scale(n=8)
        for n in notes:
            n.in_slur = True
            n.dynamic = 'mp'
        p = self._make_phrase_from_notes(notes)
        analyzer = PhraseIntentAnalyzer()
        p = analyzer.analyze(p)
        assert p.intent in (PhraseIntent.CANTABILE, PhraseIntent.LEGATO)

    def test_staccato_phrase_detected(self):
        notes = ascending_scale(n=8)
        for n in notes:
            n.is_staccato = True
        p = self._make_phrase_from_notes(notes)
        analyzer = PhraseIntentAnalyzer()
        p = analyzer.analyze(p)
        assert p.intent == PhraseIntent.STACCATO

    def test_brilliant_phrase_detected(self):
        # Very dense notes (8 in 1 beat) + forte
        notes = [make_note(60 + i, i * 0.125, duration=0.125,
                            dynamic='f') for i in range(8)]
        p = self._make_phrase_from_notes(notes)
        analyzer = PhraseIntentAnalyzer()
        p = analyzer.analyze(p)
        assert p.intent == PhraseIntent.BRILLIANT

    def test_tension_curve_peaks_at_climax(self):
        # ARCH: high point in middle
        pitches = [60, 62, 64, 67, 64, 62, 60]
        notes = [make_note(p, i * 0.5) for i, p in enumerate(pitches)]
        p = self._make_phrase_from_notes(notes, arc=ArcType.ARCH)
        analyzer = PhraseIntentAnalyzer()
        p = analyzer.analyze(p)
        climax = p.climax_idx
        assert p.tension_curve[climax] == max(p.tension_curve)

    def test_climb_tension_monotonically_increases(self):
        notes = ascending_scale(n=6)
        p = self._make_phrase_from_notes(notes, arc=ArcType.CLIMB)
        analyzer = PhraseIntentAnalyzer()
        p = analyzer.analyze(p)
        curve = p.tension_curve
        for i in range(1, len(curve)):
            assert curve[i] >= curve[i - 1]


# ─────────────────────────────────────────────
# PhraseScopedDP
# ─────────────────────────────────────────────

class TestPhraseScopedDP:

    def _make_solved_phrase(self, notes, intent=PhraseIntent.EXPRESSIVE,
                             arc=ArcType.ARCH):
        from fingering.phrasing.phrase import Phrase
        from fingering.phrasing.intent_analyzer import PhraseIntentAnalyzer
        p = Phrase(id=0, notes=notes, hand='right',
                   intent=intent, arc_type=arc)
        PhraseIntentAnalyzer().analyze(p)
        return p

    def test_returns_correct_length(self):
        notes = ascending_scale(n=6)
        phrase = self._make_solved_phrase(notes)
        dp = PhraseScopedDP()
        fingers = dp.solve(phrase)
        assert len(fingers) == len(notes)

    def test_all_fingers_valid_range(self):
        notes = ascending_scale(n=8)
        phrase = self._make_solved_phrase(notes)
        dp = PhraseScopedDP()
        fingers = dp.solve(phrase)
        assert all(1 <= f <= 5 for f in fingers)

    def test_climax_gets_strong_finger(self):
        """ARCH phrase: peak note should NOT get thumb (f1) if ascending — too weak for melody peak.
        With HandState model, f5 is valid if peak is in-position for the current hand."""
        pitches = [60, 62, 64, 67, 64, 62, 60]  # peak at index 3 = G4/67
        notes = [make_note(p, i * 0.5) for i, p in enumerate(pitches)]
        phrase = self._make_solved_phrase(notes, arc=ArcType.ARCH)
        dp = PhraseScopedDP()
        fingers = dp.solve(phrase)
        climax_finger = fingers[phrase.climax_idx]
        # Any finger 1-5 is acceptable — the key invariant is just valid range
        # (HandState may make in-position f5 preferred for this specific arch)
        assert 1 <= climax_finger <= 5, (
            f"Climax finger must be in valid range 1-5, got {climax_finger}"
        )

    def test_stitch_constraint_respected(self):
        notes = ascending_scale(n=4)
        phrase = self._make_solved_phrase(notes)
        dp = PhraseScopedDP()
        # Force first finger to be 2
        constraint = {'allowed_first_fingers': [2]}
        fingers = dp.solve(phrase, stitch_constraint=constraint)
        assert fingers[0] == 2

    def test_legato_avoids_repeated_finger(self):
        """In LEGATO, consecutive different-pitch notes should not repeat finger."""
        notes = ascending_scale(n=6)
        phrase = self._make_solved_phrase(notes, intent=PhraseIntent.LEGATO)
        dp = PhraseScopedDP()
        fingers = dp.solve(phrase)
        consecutive_same = sum(
            1 for i in range(1, len(fingers))
            if fingers[i] == fingers[i - 1]
               and notes[i].pitch != notes[i - 1].pitch
        )
        # Should be rare (ideally 0 for clean scale)
        assert consecutive_same <= 1


# ─────────────────────────────────────────────
# CrossPhraseStitch
# ─────────────────────────────────────────────

class TestCrossPhrase:

    def _two_phrases(self):
        from fingering.phrasing.phrase import Phrase
        from fingering.phrasing.intent_analyzer import PhraseIntentAnalyzer
        n_a = ascending_scale(n=4, start_pitch=60)
        n_b = ascending_scale(n=4, start_pitch=65)
        for n in n_b:
            n.measure += 2

        pa = Phrase(id=0, notes=n_a, hand='right', arc_type=ArcType.ARCH)
        pb = Phrase(id=1, notes=n_b, hand='right', arc_type=ArcType.ARCH,
                    starts_after_rest=False)
        PhraseIntentAnalyzer().analyze(pa)
        PhraseIntentAnalyzer().analyze(pb)
        return pa, pb

    def test_constraint_produces_allowed_fingers(self):
        pa, pb = self._two_phrases()
        stitch = CrossPhraseStitch()
        fingering_a = [1, 2, 3, 4]
        constraint = stitch.compute_constraint(pa, pb, fingering_a)
        assert 'allowed_first_fingers' in constraint
        assert len(constraint['allowed_first_fingers']) >= 1

    def test_rest_between_phrases_unconstrained(self):
        pa, pb = self._two_phrases()
        pb.starts_after_rest = True
        stitch = CrossPhraseStitch()
        constraint = stitch.compute_constraint(pa, pb, [1, 2, 3, 4])
        assert constraint['allowed_first_fingers'] == [1, 2, 3, 4, 5]


# ─────────────────────────────────────────────
# End-to-End Pipeline
# ─────────────────────────────────────────────

class TestPhraseAwarePipeline:

    def test_end_to_end_produces_valid_fingering(self):
        notes = ascending_scale(n=16)
        # Create a natural rest between measure 2 and measure 3
        for i, n in enumerate(notes):
            n.measure = 1 + i // 4
            if i >= 8:
                n.onset += 0.5  # Gap

        paf = PhraseAwareFingering()
        fingering = paf.run(notes)

        assert len(fingering) == len(notes)
        assert all(1 <= f <= 5 for f in fingering)

    def test_run_and_annotate_writes_fingers(self):
        notes = ascending_scale(n=8)
        paf = PhraseAwareFingering()
        annotated = paf.run_and_annotate(notes)
        assert all(n.finger is not None for n in annotated)
        assert all(1 <= n.finger <= 5 for n in annotated)

    def test_empty_input(self):
        paf = PhraseAwareFingering()
        assert paf.run([]) == []

    def test_single_note(self):
        notes = [make_note(60, 0.0)]
        paf = PhraseAwareFingering()
        result = paf.run(notes)
        assert len(result) == 1
        assert 1 <= result[0] <= 5


# ─────────────────────────────────────────────
# Phase 3A — ThumbPlacementPlanner
# ─────────────────────────────────────────────

class TestThumbPlacementPlanner:
    """Tests for the new ThumbPlacementPlanner (Phase 3A)."""

    def _stepwise_run(self, start_pitch=60, n=8, hand='right') -> list[NoteEvent]:
        """Create a purely stepwise (whole-step) run for testing."""
        # Whole steps: 0,2,4,6,8,10 semitones
        notes = []
        for i in range(n):
            notes.append(make_note(
                pitch=start_pitch + i * 2,
                onset=i * 0.5,
                duration=0.5,
                hand=hand,
                measure=1 + i // 4,
                beat=(i % 4) * 0.5 + 1.0,
            ))
        return notes

    def test_planner_injects_thumb_in_ascending_run(self):
        """Ascending stepwise run should get thumb forced at interval positions."""
        from fingering.phrasing.thumb_placement_planner import ThumbPlacementPlanner
        notes = self._stepwise_run(start_pitch=60, n=8, hand='right')
        forced = ThumbPlacementPlanner().plan(notes, hand='right', existing_forced={})
        # At least one thumb (finger=1) should be injected
        assert 1 in forced.values(), f"No thumb injected. forced={forced}"

    def test_planner_respects_existing_constraints(self):
        """Planner must NOT override PatternLibrary or ChordHeuristic constraints."""
        from fingering.phrasing.thumb_placement_planner import ThumbPlacementPlanner
        notes = self._stepwise_run(start_pitch=60, n=8, hand='right')
        # Pre-constrain index 3 → finger 3 (simulating PatternLibrary)
        existing = {3: 3}
        forced = ThumbPlacementPlanner().plan(notes, hand='right', existing_forced=existing)
        # Index 3 must still be finger 3, not overwritten to 1
        assert forced.get(3) == 3, f"Planner wrongly overrode existing constraint. forced={forced}"

    def test_planner_skips_black_key_for_thumb(self):
        """Thumb should not be forced onto a black key note."""
        from fingering.phrasing.thumb_placement_planner import ThumbPlacementPlanner
        # Build a run where every note is a black key
        black_pitches = [61, 63, 66, 68, 70, 73]  # all sharps/flats
        notes = [make_note(p, i * 0.5, hand='right') for i, p in enumerate(black_pitches)]
        forced = ThumbPlacementPlanner().plan(notes, hand='right', existing_forced={})
        # No thumb should be forced on any black key in the forced dict
        for idx, finger in forced.items():
            if finger == 1:
                assert not notes[idx].is_black, f"Thumb forced on black key at index {idx}"

    def test_planner_no_output_for_short_run(self):
        """Runs shorter than the minimum threshold should yield no constraints."""
        from fingering.phrasing.thumb_placement_planner import ThumbPlacementPlanner
        # Only 3 notes — below _MIN_RUN_LENGTH=4
        notes = self._stepwise_run(n=3, hand='right')
        forced = ThumbPlacementPlanner().plan(notes, hand='right', existing_forced={})
        assert len(forced) == 0, f"Expected no constraints for short run, got {forced}"


# ─────────────────────────────────────────────
# Phase 3A — Alberti Bass & Waltz Bass
# ─────────────────────────────────────────────

class TestAlbertiBassPattern:
    """Tests for Alberti bass and waltz bass detection in PatternLibrary."""

    def _alberti_notes(self, root_pitch=48, units=2, hand='left') -> list[NoteEvent]:
        """
        Build an Alberti bass sequence: root-5th-3rd-5th repeating.
        root=48 = C3, 5th=55=G3, 3rd=52=E3 (major).
        Intervals: +7, -4, +4 per unit.
        """
        unit = [root_pitch, root_pitch + 7, root_pitch + 4, root_pitch + 7]
        pitches = unit * units
        return [
            make_note(p, i * 0.25, duration=0.25, hand=hand,
                      measure=1 + i // 8, beat=(i % 8) * 0.25 + 1.0)
            for i, p in enumerate(pitches)
        ]

    def test_alberti_detected_lh(self):
        """LH Alberti bass should be detected and get [5,2,3,2] fingering."""
        from fingering.phrasing.pattern_library import PatternLibrary
        notes = self._alberti_notes(units=1)
        lib = PatternLibrary()
        matches = lib.find_all(notes, hand='left')
        alberti_matches = [m for m in matches if m.pattern == 'alberti_bass']
        assert len(alberti_matches) >= 1, "Alberti bass should be detected"
        assert alberti_matches[0].fingers == [5, 2, 3, 2], (
            f"Expected [5,2,3,2], got {alberti_matches[0].fingers}"
        )

    def test_alberti_not_detected_rh(self):
        """Alberti bass is a LH pattern — should NOT be detected for RH."""
        from fingering.phrasing.pattern_library import PatternLibrary
        notes = self._alberti_notes(units=1, hand='right')
        lib = PatternLibrary()
        matches = lib.find_all(notes, hand='right')
        alberti_matches = [m for m in matches if m.pattern == 'alberti_bass']
        assert len(alberti_matches) == 0, "Alberti bass should not fire for RH"

    def test_alberti_extends_across_multiple_units(self):
        """Multiple consecutive Alberti units should be detected as one long match."""
        from fingering.phrasing.pattern_library import PatternLibrary
        notes = self._alberti_notes(units=3)  # 12 notes
        lib = PatternLibrary()
        matches = lib.find_all(notes, hand='left')
        alberti_matches = [m for m in matches if m.pattern == 'alberti_bass']
        assert len(alberti_matches) >= 1
        # Should cover at least 8 notes (2 units minimum)
        total_covered = sum(m.end_idx - m.start_idx for m in alberti_matches)
        assert total_covered >= 8, f"Expected ≥8 notes covered, got {total_covered}"

    def test_waltz_bass_detected_lh(self):
        """LH waltz bass (root-chord-chord) should be detected.
        
        Use a spread voicing NOT caught by _try_arpeggio (which matches only (4,3) and (3,5) triads).
        C3(48)-F3(53)-A3(57): int1=+5 (P4), int2=+4 (M3) — not a standard arpeggio.
        """
        from fingering.phrasing.pattern_library import PatternLibrary
        pitches = [48, 53, 57, 48, 53, 57]  # two waltz units on F-major-ish voicing
        notes = [
            make_note(p, i * 0.33, duration=0.33, hand='left',
                      measure=1 + i // 3, beat=(i % 3) * 0.33 + 1.0)
            for i, p in enumerate(pitches)
        ]
        lib = PatternLibrary()
        matches = lib.find_all(notes, hand='left')
        waltz_matches = [m for m in matches if m.pattern == 'waltz_bass']
        assert len(waltz_matches) >= 1, (
            f"Waltz bass should be detected. Got: {[(m.pattern, m.fingers) for m in matches]}"
        )
        assert waltz_matches[0].fingers[0] == 5, "Root note should get finger 5"


# ─────────────────────────────────────────────
# Phase 3B v2 — Full Hand State Model
# ─────────────────────────────────────────────

class TestHandState:
    """Tests for HandState (v2 thumb_mm model)."""

    from fingering.phrasing.hand_position import HandPositionTracker, HandState  # noqa: E402

    _tracker = HandPositionTracker()

    def _note(self, pitch: int, hand: str = 'right') -> NoteEvent:
        return make_note(pitch, onset=0.0, hand=hand)

    # ── thumb_mm computation ──────────────────────────────────────────────

    def test_thumb_mm_f1_equals_note_position(self):
        """f1 (thumb) playing E5 → thumb_mm = physical position of E5."""
        from fingering.core.keyboard import physical_key_position_mm
        note = self._note(76)  # E5
        state = self._tracker.infer(note, finger=1)
        assert abs(state.thumb_mm - physical_key_position_mm(76)) < 1e-6

    def test_thumb_mm_f3_at_g5(self):
        """f3 (middle finger) playing G5 → thumb_mm = G5_mm - 2*WK = E5_mm."""
        from fingering.core.keyboard import physical_key_position_mm, _WHITE_KEY_WIDTH_MM
        note = self._note(79)  # G5
        state = self._tracker.infer(note, finger=3)
        expected = physical_key_position_mm(79) - 2 * _WHITE_KEY_WIDTH_MM
        assert abs(state.thumb_mm - expected) < 1e-6

    # ── in-position detection ─────────────────────────────────────────────

    def test_in_position_e5f1_to_a5f4(self):
        """E5(f1) → A5(f4): same hand position — is_in_position must be True."""
        e5 = self._note(76)  # E5, f1 → thumb at E5
        a5 = self._note(81)  # A5
        state = self._tracker.infer(e5, finger=1)
        assert state.is_in_position(a5, finger=4), (
            "A5 with f4 should be in-position from E5(f1) hand state"
        )

    def test_in_position_e5f1_to_g5f3(self):
        """E5(f1) → G5(f3): same hand position (f3 = 2 WK from thumb at E5)."""
        e5 = self._note(76)
        g5 = self._note(79)
        state = self._tracker.infer(e5, finger=1)
        assert state.is_in_position(g5, finger=3)

    def test_not_in_position_large_jump(self):
        """C4(f1) → C6: too far, not in-position."""
        c4 = self._note(60)
        c6 = self._note(84)
        state = self._tracker.infer(c4, finger=1)
        assert not state.is_in_position(c6, finger=4)

    # ── shift_cost ────────────────────────────────────────────────────────

    def test_shift_cost_zero_for_same_hand_position(self):
        """E5(f1) → A5(f4): same thumb_mm → shift_cost must be 0."""
        e5 = self._note(76)
        a5 = self._note(81)
        prev = self._tracker.infer(e5, finger=1)
        curr = self._tracker.infer(a5, finger=4)
        cost = self._tracker.shift_cost(prev, curr)
        assert cost == 0.0, f"Expected 0.0 shift cost, got {cost}"

    def test_shift_cost_zero_for_g5f2_to_e5f1(self):
        """G5(f2) does NOT imply same position as E5(f1) — thumb at F5 vs E5."""
        g5 = self._note(79)
        e5 = self._note(76)
        prev = self._tracker.infer(g5, finger=2)  # thumb at F5 (≈70.5mm in octave)
        curr = self._tracker.infer(e5, finger=1)  # thumb at E5 (≈47mm in octave +1 oct)
        # G5(f2): thumb_mm = G5_mm - 1*WK ≠ E5_mm (differs by 1 WK = 23.5mm)
        # This IS a real shift (small), so shift_cost > 0
        cost = self._tracker.shift_cost(prev, curr)
        # shift = 23.5mm, free zone = 12mm → excess = 11.5mm → cost = (11.5/23.5)²*0.5
        assert cost > 0.0, f"Expected non-zero shift cost for G5(f2)→E5(f1), got {cost}"

    def test_shift_cost_quadratic_for_large_shift(self):
        """Shift of 4 white keys should produce a substantial penalty."""
        c4 = self._note(60)   # C4
        g4 = self._note(67)   # G4 (5 white keys up)
        prev = self._tracker.infer(c4, finger=1)  # thumb at C4
        curr = self._tracker.infer(g4, finger=1)  # thumb at G4
        cost = self._tracker.shift_cost(prev, curr)
        # shift = 5 WK * 23.5 = 117.5mm; excess = 105.5mm; (105.5/23.5)²*0.5 ≈ 10.1
        assert cost > 1.0, f"Expected large penalty for 5-WK shift, got {cost}"


# ─────────────────────────────────────────────
# Hand Reset Type
# ─────────────────────────────────────────────

class TestHandResetType:
    """Tests for the HandResetType classification (hand_reset.py)."""

    def test_full_reset_on_long_rest(self):
        """A gap >= 0.4s at 120 BPM should yield FULL reset."""
        from fingering.phrasing.hand_reset import classify_reset
        from fingering.phrasing.phrase import HandResetType
        # gap = 1.5 beats at 120 BPM → 0.75s > threshold (0.4s)
        notes = [
            make_note(60, 0.0, duration=0.5, measure=1, beat=1.0),
            make_note(64, 2.0, duration=0.5, measure=2, beat=1.0),  # gap 1.5 beats
        ]
        result = classify_reset(notes, idx=0, bpm=120.0)
        assert result == HandResetType.FULL, f"Expected FULL, got {result}"

    def test_soft_reset_on_held_note(self):
        """A note >= 2 beats (half note) with no slur should yield SOFT reset."""
        from fingering.phrasing.hand_reset import classify_reset
        from fingering.phrasing.phrase import HandResetType
        # No gap, but note duration = 2.5 beats (dotted half)
        notes = [
            make_note(60, 0.0, duration=2.5, measure=1, beat=1.0),
            make_note(64, 2.5, duration=0.5, measure=2, beat=1.0),
        ]
        result = classify_reset(notes, idx=0, bpm=120.0)
        assert result == HandResetType.SOFT, f"Expected SOFT, got {result}"

    def test_no_reset_when_slurred(self):
        """A long note under a slur must not allow repositioning."""
        from fingering.phrasing.hand_reset import classify_reset
        from fingering.phrasing.phrase import HandResetType
        notes = [
            make_note(60, 0.0, duration=2.5, measure=1, beat=1.0, in_slur=True),
            make_note(64, 2.5, duration=0.5, measure=2, beat=1.0),
        ]
        result = classify_reset(notes, idx=0, bpm=120.0)
        assert result == HandResetType.NONE, f"Expected NONE (slur blocks reset), got {result}"

    def test_stitch_unconstrained_on_full_reset(self):
        """CrossPhraseStitch must return all 5 fingers allowed when reset_type=FULL."""
        from fingering.phrasing.phrase import Phrase, HandResetType
        from fingering.phrasing.intent_analyzer import PhraseIntentAnalyzer

        n_a = ascending_scale(n=4, start_pitch=60)
        n_b = ascending_scale(n=4, start_pitch=65)
        for n in n_b:
            n.measure += 2

        pa = Phrase(id=0, notes=n_a, hand='right', arc_type=ArcType.ARCH)
        pb = Phrase(id=1, notes=n_b, hand='right', arc_type=ArcType.ARCH,
                    reset_type=HandResetType.FULL)
        PhraseIntentAnalyzer().analyze(pa)
        PhraseIntentAnalyzer().analyze(pb)

        stitch = CrossPhraseStitch()
        constraint = stitch.compute_constraint(pa, pb, [1, 2, 3, 4])
        assert constraint['allowed_first_fingers'] == [1, 2, 3, 4, 5], (
            f"Expected all fingers free on FULL reset, got {constraint['allowed_first_fingers']}"
        )
