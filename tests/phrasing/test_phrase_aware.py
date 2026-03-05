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
        """rest_follows contributes weight but alone cannot trigger a boundary."""
        s = PhraseBoundarySignal(position=0, rest_follows=True, rest_duration=1.0)
        # Contributes 0.20, which is below the 0.40 threshold — by design
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
        notes = ascending_scale(n=8)
        # Mark note 3 as slur_end
        notes[3].slur_end = True
        notes[4].slur_start = True
        notes[4].beat = 1.0
        notes[4].measure = 2
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
        """ARCH phrase: peak note should get finger 2 or 3."""
        pitches = [60, 62, 64, 67, 64, 62, 60]  # peak at index 3 = G4/67
        notes = [make_note(p, i * 0.5) for i, p in enumerate(pitches)]
        phrase = self._make_solved_phrase(notes, arc=ArcType.ARCH)
        dp = PhraseScopedDP()
        fingers = dp.solve(phrase)
        climax_finger = fingers[phrase.climax_idx]
        assert climax_finger in {1, 2, 3}, (
            f"Climax finger should be strong, got {climax_finger}"
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
