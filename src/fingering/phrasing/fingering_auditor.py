"""
Fingering Auditor — Post-DP validation layer.

After the Viterbi DP generates a fingering sequence, this module scans
every transition for physically impossible or pedagogically wrong assignments.

Severity levels:
  - HARD_VIOLATION:  Physically impossible (e.g. span > anatomical max)
                     → Must be fixed before output
  - WARNING:         Unreasonable but not impossible (e.g. thumb on black in run)
                     → Flagged and optionally fixed
  - OK:              Transition is clean

The auditor also offers a `repair()` method that attempts to fix HARD violations
by reassigning the offending finger using a greedy local search within ±2 notes.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Tuple, Optional
import math

from fingering.models.note_event import NoteEvent
from fingering.core.keyboard import (
    white_key_span, finger_span_limits, is_ascending,
    thumb_crossing_natural,
)


class Severity(Enum):
    OK             = auto()
    WARNING        = auto()
    HARD_VIOLATION = auto()


@dataclass
class AuditIssue:
    idx: int                   # Index of SECOND note in the transition (or single note)
    note: NoteEvent
    finger: int
    prev_finger: Optional[int]
    prev_note:   Optional[NoteEvent]
    severity:    Severity
    rule:        str           # Which rule triggered
    detail:      str           # Human-readable explanation

    def __str__(self) -> str:
        loc = f"m{self.note.measure} b{self.note.beat:.1f}"
        pitch_name = _midi_to_name(self.note.pitch)
        prev = f"f{self.prev_finger}→" if self.prev_finger else ""
        return (
            f"[{self.severity.name:14s}] {loc:12s} {pitch_name:5s} "
            f"{prev}f{self.finger}  ← {self.rule}: {self.detail}"
        )


@dataclass
class AuditReport:
    issues: List[AuditIssue] = field(default_factory=list)

    @property
    def hard_violations(self) -> List[AuditIssue]:
        return [i for i in self.issues if i.severity == Severity.HARD_VIOLATION]

    @property
    def warnings(self) -> List[AuditIssue]:
        return [i for i in self.issues if i.severity == Severity.WARNING]

    @property
    def is_clean(self) -> bool:
        return len(self.hard_violations) == 0

    def summary(self) -> str:
        h = len(self.hard_violations)
        w = len(self.warnings)
        return f"Audit: {h} HARD violation(s), {w} WARNING(s)"

    def print_report(self, max_lines: int = 50) -> None:
        print(self.summary())
        for i, issue in enumerate(self.issues):
            if i >= max_lines:
                print(f"  ... ({len(self.issues) - max_lines} more)")
                break
            prefix = "  ✖" if issue.severity == Severity.HARD_VIOLATION else "  ⚠"
            print(f"{prefix} {issue}")


# ──────────────────────────────────────────────────────────────
# Individual Rules
# ──────────────────────────────────────────────────────────────

class FingeredAuditor:
    """
    Audits a (notes, fingering) pair for physical and pedagogical correctness.

    Usage:
        auditor = FingeredAuditor()
        report = auditor.audit(notes, fingering, hand='right')
        report.print_report()

        # Attempt to fix hard violations in-place
        fixed_fingering = auditor.repair(notes, fingering, hand='right')
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def audit(
        self,
        notes: List[NoteEvent],
        fingering: List[int],
        hand: str = 'right',
    ) -> AuditReport:
        """Run all audit rules and return an AuditReport."""
        report = AuditReport()
        n = min(len(notes), len(fingering))

        for i in range(n):
            note = notes[i]
            f    = fingering[i]
            prev_note = notes[i - 1] if i > 0 else None
            prev_f    = fingering[i - 1] if i > 0 else None

            issues = []

            # ---- Per-note rules (no previous context needed) ----
            issues += self._rule_finger_range(i, note, f)
            issues += self._rule_thumb_on_black_in_run(i, note, f, notes, fingering, hand)
            issues += self._rule_pinky_on_black_fast(i, note, f, notes, hand)
            issues += self._rule_weak_finger_on_climax(i, note, f, notes, fingering)

            # ---- Transition rules (need prev) ----
            if prev_note is not None and prev_f is not None:
                issues += self._rule_impossible_span(i, note, f, prev_note, prev_f, hand)
                issues += self._rule_ofok_legato(i, note, f, prev_note, prev_f)
                issues += self._rule_wrong_direction_cross(i, note, f, prev_note, prev_f, hand)
                issues += self._rule_double_cross(i, notes, fingering, hand)
                issues += self._rule_span_too_compressed(i, note, f, prev_note, prev_f)
                issues += self._rule_consecutive_weak_fingers(i, notes, fingering)

            report.issues.extend(issues)

        return report

    def repair(
        self,
        notes: List[NoteEvent],
        fingering: List[int],
        hand: str = 'right',
    ) -> List[int]:
        """
        Attempt to fix HARD violations with a local greedy search.

        Strategy: for each hard violation at index i, try all 5 fingers
        at position i and pick the one that produces valid transitions
        with both (i-1) and (i+1).
        """
        fixed = list(fingering)
        report = self.audit(notes, fixed, hand)

        for issue in report.hard_violations:
            idx = issue.idx
            best_f = fixed[idx]
            best_score = float('inf')

            for candidate_f in range(1, 6):
                fixed[idx] = candidate_f
                # Score this candidate: sum of violation counts in local window
                window_start = max(0, idx - 2)
                window_end   = min(len(notes), idx + 3)
                window_notes  = notes[window_start:window_end]
                window_finger = fixed[window_start:window_end]
                local_report = self.audit(window_notes, window_finger, hand)
                score = (
                    len(local_report.hard_violations) * 100
                    + len(local_report.warnings) * 1
                )
                if score < best_score:
                    best_score = score
                    best_f = candidate_f

            fixed[idx] = best_f

        return fixed

    # ------------------------------------------------------------------
    # Rule Implementations
    # ------------------------------------------------------------------

    def _rule_finger_range(
        self, idx: int, note: NoteEvent, f: int
    ) -> List[AuditIssue]:
        """Finger must be 1–5."""
        if not (1 <= f <= 5):
            return [AuditIssue(
                idx=idx, note=note, finger=f,
                prev_finger=None, prev_note=None,
                severity=Severity.HARD_VIOLATION,
                rule="FINGER_RANGE",
                detail=f"Finger {f} is out of range [1-5]",
            )]
        return []

    def _rule_impossible_span(
        self, idx: int,
        note: NoteEvent, f: int,
        prev_note: NoteEvent, prev_f: int,
        hand: str,
    ) -> List[AuditIssue]:
        """Physical span between two consecutive assigned fingers must be achievable."""
        span = white_key_span(prev_note, note)
        _, max_span = finger_span_limits(prev_f, f)
        min_span, _ = finger_span_limits(prev_f, f)

        if span > max_span:
            return [AuditIssue(
                idx=idx, note=note, finger=f,
                prev_finger=prev_f, prev_note=prev_note,
                severity=Severity.HARD_VIOLATION,
                rule="IMPOSSIBLE_SPAN",
                detail=f"Span {span} exceeds max {max_span} for fingers {prev_f}→{f}",
            )]
        if span < min_span:
            return [AuditIssue(
                idx=idx, note=note, finger=f,
                prev_finger=prev_f, prev_note=prev_note,
                severity=Severity.WARNING,
                rule="SPAN_TOO_COMPRESSED",
                detail=f"Span {span} below min {min_span} for fingers {prev_f}→{f} (squashed)",
            )]
        return []

    def _rule_thumb_on_black_in_run(
        self, idx: int, note: NoteEvent, f: int,
        notes: List[NoteEvent], fingering: List[int], hand: str,
    ) -> List[AuditIssue]:
        """Thumb should not land on a black key in the middle of a stepwise run."""
        if f != 1 or not note.is_black:
            return []
        # In a stepwise run (prev and next are both adjacent semitones), it's bad
        n = len(notes)
        if idx == 0 or idx == n - 1:
            return []
        prev_interval = abs(note.pitch - notes[idx - 1].pitch)
        next_interval = abs(notes[idx + 1].pitch - note.pitch) if idx + 1 < n else 99
        if prev_interval <= 2 and next_interval <= 2:
            return [AuditIssue(
                idx=idx, note=note, finger=f,
                prev_finger=fingering[idx - 1] if idx > 0 else None,
                prev_note=notes[idx - 1] if idx > 0 else None,
                severity=Severity.WARNING,
                rule="THUMB_ON_BLACK_IN_RUN",
                detail=f"Thumb on black key {_midi_to_name(note.pitch)} in stepwise run",
            )]
        return []

    def _rule_pinky_on_black_fast(
        self, idx: int, note: NoteEvent, f: int,
        notes: List[NoteEvent], hand: str,
    ) -> List[AuditIssue]:
        """Pinky on black key is uncomfortable; warn if duration is short (fast passage)."""
        if f != 5 or not note.is_black:
            return []
        if note.duration < 0.5:  # Shorter than an eighth note at q=120
            return [AuditIssue(
                idx=idx, note=note, finger=f,
                prev_finger=None, prev_note=None,
                severity=Severity.WARNING,
                rule="PINKY_ON_BLACK_FAST",
                detail=f"Pinky on short black key note {_midi_to_name(note.pitch)} (dur={note.duration:.2f})",
            )]
        return []

    def _rule_ofok_legato(
        self, idx: int,
        note: NoteEvent, f: int,
        prev_note: NoteEvent, prev_f: int,
    ) -> List[AuditIssue]:
        """Same finger on different pitch (OFOK) is a hard violation when slurred."""
        if f == prev_f and note.pitch != prev_note.pitch:
            is_slurred = getattr(note, 'in_slur', False) or getattr(prev_note, 'in_slur', False)
            severity = Severity.HARD_VIOLATION if is_slurred else Severity.WARNING
            return [AuditIssue(
                idx=idx, note=note, finger=f,
                prev_finger=prev_f, prev_note=prev_note,
                severity=severity,
                rule="OFOK",
                detail=f"Finger {f} reused on different pitch "
                       f"({_midi_to_name(prev_note.pitch)}→{_midi_to_name(note.pitch)})"
                       + (" [in slur]" if is_slurred else ""),
            )]
        return []

    def _rule_wrong_direction_cross(
        self, idx: int,
        note: NoteEvent, f: int,
        prev_note: NoteEvent, prev_f: int,
        hand: str,
    ) -> List[AuditIssue]:
        """
        Detect backwards crossing: e.g., RH ascending but using finger-over
        (higher finger going under lower) instead of thumb-under.
        """
        ascending = note.pitch > prev_note.pitch

        # RH ascending: f_curr should be >= f_prev, UNLESS it's thumb-under (f_curr=1)
        if hand == 'right' and ascending:
            if f < prev_f and f != 1:
                return [AuditIssue(
                    idx=idx, note=note, finger=f,
                    prev_finger=prev_f, prev_note=prev_note,
                    severity=Severity.WARNING,
                    rule="WRONG_CROSS_DIRECTION",
                    detail=f"RH ascending but f{prev_f}→f{f} is finger-over (should be thumb-under)",
                )]
        # RH descending: f_curr should be <= f_prev, UNLESS it's finger-over post-thumb (f_prev=1)
        if hand == 'right' and not ascending:
            if f > prev_f and prev_f != 1:
                return [AuditIssue(
                    idx=idx, note=note, finger=f,
                    prev_finger=prev_f, prev_note=prev_note,
                    severity=Severity.WARNING,
                    rule="WRONG_CROSS_DIRECTION",
                    detail=f"RH descending but f{prev_f}→f{f} is thumb-under (should be finger-over)",
                )]
        return []

    def _rule_double_cross(
        self, idx: int,
        notes: List[NoteEvent], fingering: List[int],
        hand: str,
    ) -> List[AuditIssue]:
        """
        Detect two consecutive thumb-under or finger-over without a note in between.
        This is physically impossible at normal tempo.
        """
        if idx < 2:
            return []
        f0 = fingering[idx - 2]
        f1 = fingering[idx - 1]
        f2 = fingering[idx]
        asc01 = notes[idx - 1].pitch > notes[idx - 2].pitch
        asc12 = notes[idx].pitch > notes[idx - 1].pitch

        # Two consecutive thumb-unders in same direction
        thumb_under_01 = (f1 == 1 and asc01 and hand == 'right')
        thumb_under_12 = (f2 == 1 and asc12 and hand == 'right')

        if thumb_under_01 and thumb_under_12:
            return [AuditIssue(
                idx=idx, note=notes[idx], finger=f2,
                prev_finger=f1, prev_note=notes[idx - 1],
                severity=Severity.HARD_VIOLATION,
                rule="DOUBLE_CROSS",
                detail=f"Two consecutive thumb-unders (f{f0}→f{f1}→f{f2}) without intervening notes",
            )]
        return []

    def _rule_span_too_compressed(
        self, idx: int,
        note: NoteEvent, f: int,
        prev_note: NoteEvent, prev_f: int,
    ) -> List[AuditIssue]:
        """Adjacent fingers on notes too far apart (span mismatch)."""
        span = white_key_span(prev_note, note)
        pair = (min(prev_f, f), max(prev_f, f))
        # Specific bad combination: 4→5 or 3→4 trying to span > 4 white keys
        if pair in {(4, 5), (3, 4)} and span > 4:
            return [AuditIssue(
                idx=idx, note=note, finger=f,
                prev_finger=prev_f, prev_note=prev_note,
                severity=Severity.WARNING,
                rule="WEAK_PAIR_OVERSPAN",
                detail=f"Weak pair f{prev_f}→f{f} spanning {span} white keys (uncomfortable for 3-4 or 4-5)",
            )]
        return []

    def _rule_weak_finger_on_climax(
        self, idx: int,
        note: NoteEvent, f: int,
        notes: List[NoteEvent], fingering: List[int],
    ) -> List[AuditIssue]:
        """Pinky (5) should not play the highest pitch note in a phrase."""
        if f != 5:
            return []
        if not notes:
            return []
        max_pitch = max(n.pitch for n in notes)
        if note.pitch == max_pitch and len(notes) > 3:
            return [AuditIssue(
                idx=idx, note=note, finger=f,
                prev_finger=None, prev_note=None,
                severity=Severity.WARNING,
                rule="WEAK_ON_CLIMAX",
                detail=f"Pinky (5) on climax note {_midi_to_name(note.pitch)} — poor dynamic control",
            )]
        return []

    def _rule_consecutive_weak_fingers(
        self, idx: int,
        notes: List[NoteEvent], fingering: List[int],
    ) -> List[AuditIssue]:
        """Three or more consecutive weak fingers (4,5) in a row is ergonomically stressful."""
        if idx < 2:
            return []
        window = [fingering[i] for i in range(max(0, idx - 2), idx + 1)]
        if all(f in {4, 5} for f in window):
            return [AuditIssue(
                idx=idx, note=notes[idx], finger=fingering[idx],
                prev_finger=fingering[idx - 1], prev_note=notes[idx - 1],
                severity=Severity.WARNING,
                rule="CONSECUTIVE_WEAK",
                detail=f"Three consecutive weak fingers: {window}",
            )]
        return []


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────

_NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

def _midi_to_name(midi: int) -> str:
    pc = midi % 12
    octave = (midi // 12) - 1
    return f"{_NOTE_NAMES[pc]}{octave}"
