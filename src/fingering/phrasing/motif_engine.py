"""
Layer 1: Motif Engine — Top-Down Pattern Recognition

A pianist sees a recurring melodic cell (e.g. E-D#-E-D#-E in Für Elise)
and immediately groups all occurrences into the same "section A".
This module identifies motifs and their recurrences to infer musical form.

Pipeline:
  1. Extract motif fingerprints (interval profile + rhythm profile)
  2. Hash and cluster similar fingerprints
  3. Detect recurrences across the piece
  4. Infer named sections (A, B, A', C, ...)
  5. Force phrase boundaries at section transition points
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
from collections import defaultdict
import math
from fingering.models.note_event import NoteEvent


# ──────────────────────────────────────────────────────────────
# Data Structures
# ──────────────────────────────────────────────────────────────

@dataclass
class Motif:
    """A short recurring melodic-rhythmic cell."""
    start_idx: int              # Index in the source note list
    length: int                 # Number of notes
    measure: int                # Starting measure
    interval_key: tuple         # Normalized interval fingerprint
    rhythm_key: tuple           # Normalized rhythm fingerprint
    fingerprint: tuple          # Combined (interval_key, rhythm_key)


@dataclass
class RecurrencePair:
    """Two motifs with the same fingerprint at different positions."""
    motif_a: Motif
    motif_b: Motif
    similarity: float           # 0–1 (1 = exact match)
    measures_apart: int


@dataclass
class Section:
    """
    A top-level structural section inferred from motif recurrence.
    Corresponds to the pianist's A, B, A', C labels.
    """
    label: str                  # 'A', 'B', "A'", 'C', ...
    start_measure: int
    end_measure: int
    motif_fingerprint: tuple    # Defining motif of this section
    is_repeat: bool = False     # True = repeat of an earlier section


# ──────────────────────────────────────────────────────────────
# Fingerprinting Helpers
# ──────────────────────────────────────────────────────────────

_INTERVAL_TOLERANCE = 1   # semitone tolerance for fuzzy matching

def _interval_key(notes: List[NoteEvent], start: int, length: int) -> tuple:
    """
    Compute a normalized interval fingerprint for a note window.
    Values are quantized to reduce transposition sensitivity.
    """
    if start + length > len(notes):
        return ()
    intervals = []
    for i in range(start, start + length - 1):
        diff = notes[i + 1].pitch - notes[i].pitch
        # Quantize: map to nearest 1 semitone (already integers)
        intervals.append(diff)
    return tuple(intervals)


def _rhythm_key(notes: List[NoteEvent], start: int, length: int) -> tuple:
    """
    Compute a rhythm fingerprint: relative duration ratios,
    quantized into 3 levels: SHORT / MEDIUM / LONG.
    """
    if start + length > len(notes):
        return ()
    durs = [notes[i].duration for i in range(start, start + length)]
    if not durs or max(durs) == 0:
        return tuple([1] * len(durs))
    # Quantize: SHORT=1, MEDIUM=2, LONG=3
    min_d = min(d for d in durs if d > 0)
    quant = []
    for d in durs:
        ratio = d / min_d if min_d > 0 else 1.0
        if ratio < 1.5:
            quant.append(1)    # SHORT
        elif ratio < 2.5:
            quant.append(2)    # MEDIUM
        else:
            quant.append(3)    # LONG
    return tuple(quant)


def _combined_fingerprint(
    notes: List[NoteEvent], start: int, length: int,
) -> tuple:
    """Combined fingerprint: (interval_key, rhythm_key)."""
    return (
        _interval_key(notes, start, length),
        _rhythm_key(notes, start, length),
    )


def _fingerprint_similarity(fp_a: tuple, fp_b: tuple) -> float:
    """
    Compare two fingerprints (interval_key, rhythm_key).
    Returns similarity 0–1 (1 = exact match).
    """
    if fp_a == fp_b:
        return 1.0

    iv_a, rh_a = fp_a
    iv_b, rh_b = fp_b

    if len(iv_a) != len(iv_b) or len(rh_a) != len(rh_b):
        return 0.0

    # Interval similarity: allow ±1 semitone tolerance
    iv_matches = sum(
        1 for a, b in zip(iv_a, iv_b) if abs(a - b) <= _INTERVAL_TOLERANCE
    )
    iv_sim = iv_matches / max(1, len(iv_a))

    # Rhythm similarity: exact match required (rhythm is key identity signal)
    rh_sim = 1.0 if rh_a == rh_b else (
        sum(1 for a, b in zip(rh_a, rh_b) if a == b) / max(1, len(rh_a)) * 0.5
    )

    return 0.6 * iv_sim + 0.4 * rh_sim


# ──────────────────────────────────────────────────────────────
# Main Motif Engine
# ──────────────────────────────────────────────────────────────

class MotifEngine:
    """
    Detects recurring melodic cells and infers musical form.

    Usage:
        engine = MotifEngine()
        sections = engine.infer_sections(rh_notes)
        # sections: list of Section (A, B, A', ...)
        # Use sections to force phrase boundaries at section starts
    """

    def __init__(
        self,
        motif_lengths: tuple = (8, 12, 16),   # Fix 2: phrase-level (not sub-motif)
        min_recurrences: int = 2,              # need >= 2 occurrences to be a motif
        similarity_threshold: float = 0.80,   # minimum similarity to group
        min_measures_apart: int = 8,           # Fix 2: phrase-level gap (was 4)
    ):
        self.motif_lengths = motif_lengths
        self.min_recurrences = min_recurrences
        self.similarity_threshold = similarity_threshold
        self.min_measures_apart = min_measures_apart

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def infer_sections(self, notes: List[NoteEvent]) -> List[Section]:
        """
        Top-level entry point.
        Returns a list of Section objects sorted by start_measure.
        """
        if len(notes) < 8:
            return []

        motifs = self._extract_motifs(notes)
        groups = self._cluster_motifs(motifs)
        recurrences = self._find_recurrences(groups)
        sections = self._build_sections(recurrences, notes)
        return sections

    def get_forced_boundaries(self, sections: List[Section]) -> List[int]:
        """
        Returns measure numbers where section transitions force phrase breaks.
        """
        boundaries = set()
        for s in sections:
            boundaries.add(s.start_measure)
            boundaries.add(s.end_measure + 1)
        return sorted(boundaries)

    # ------------------------------------------------------------------
    # Step 1: Extract motifs
    # ------------------------------------------------------------------

    def _extract_motifs(self, notes: List[NoteEvent]) -> List[Motif]:
        motifs: List[Motif] = []
        n = len(notes)
        for length in self.motif_lengths:
            for start in range(0, n - length + 1):
                fp = _combined_fingerprint(notes, start, length)
                if not fp[0]:   # Empty interval key → skip
                    continue
                m = Motif(
                    start_idx=start,
                    length=length,
                    measure=notes[start].measure,
                    interval_key=fp[0],
                    rhythm_key=fp[1],
                    fingerprint=fp,
                )
                motifs.append(m)
        return motifs

    # ------------------------------------------------------------------
    # Step 2: Cluster by fingerprint similarity
    # ------------------------------------------------------------------

    def _cluster_motifs(
        self, motifs: List[Motif],
    ) -> Dict[tuple, List[Motif]]:
        """
        Group motifs by their fingerprint.
        Returns dict mapping canonical fingerprint → list of Motif.
        Only keeps groups with >= min_recurrences occurrences.
        """
        groups: Dict[tuple, List[Motif]] = defaultdict(list)

        for motif in motifs:
            # Try to find an existing group this motif belongs to
            matched = False
            for canon_fp, group in groups.items():
                if _fingerprint_similarity(motif.fingerprint, canon_fp) >= self.similarity_threshold:
                    group.append(motif)
                    matched = True
                    break
            if not matched:
                groups[motif.fingerprint].append(motif)

        # Filter: only keep groups that recur at least min_recurrences times
        # and whose members are spread >= min_measures_apart
        filtered = {}
        for fp, group in groups.items():
            # Sort by measure
            group.sort(key=lambda m: m.measure)
            # Keep only non-overlapping occurrences
            non_overlapping = [group[0]]
            for motif in group[1:]:
                if motif.measure - non_overlapping[-1].measure >= self.min_measures_apart:
                    non_overlapping.append(motif)
            if len(non_overlapping) >= self.min_recurrences:
                filtered[fp] = non_overlapping

        return filtered

    # ------------------------------------------------------------------
    # Step 3: Build recurrence pairs
    # ------------------------------------------------------------------

    def _find_recurrences(
        self, groups: Dict[tuple, List[Motif]],
    ) -> List[RecurrencePair]:
        pairs: List[RecurrencePair] = []
        for fp, group in groups.items():
            for i in range(len(group) - 1):
                for j in range(i + 1, len(group)):
                    a, b = group[i], group[j]
                    sim = _fingerprint_similarity(a.fingerprint, b.fingerprint)
                    pairs.append(RecurrencePair(
                        motif_a=a,
                        motif_b=b,
                        similarity=sim,
                        measures_apart=b.measure - a.measure,
                    ))
        return pairs

    # ------------------------------------------------------------------
    # Step 4: Infer sections from recurrences  
    # ------------------------------------------------------------------

    def _build_sections(
        self,
        recurrences: List[RecurrencePair],
        notes: List[NoteEvent],
    ) -> List[Section]:
        # Convert recurrence pairs into a concise section list.
        # Output: typically 4-10 Section objects for a full piece.
        if not recurrences:
            return []

        total_measures = max(n.measure for n in notes)

        # Rebuild group structure from pairs
        fp_occurrences: Dict[tuple, List[int]] = defaultdict(list)
        fp_to_motif: Dict[tuple, Motif] = {}
        for pair in recurrences:
            canon = pair.motif_a.fingerprint
            fp_occurrences[canon].append(pair.motif_a.measure)
            fp_occurrences[canon].append(pair.motif_b.measure)
            if canon not in fp_to_motif:
                fp_to_motif[canon] = pair.motif_a

        # Deduplicate + filter to non-overlapping occurrences per fingerprint
        cleaned: Dict[tuple, List[int]] = {}
        for fp, measures in fp_occurrences.items():
            measures = sorted(set(measures))
            deduped = [measures[0]]
            for m in measures[1:]:
                if m - deduped[-1] >= self.min_measures_apart:
                    deduped.append(m)
            if len(deduped) >= self.min_recurrences:
                cleaned[fp] = deduped

        if not cleaned:
            return []

        # Select top 5 groups by occurrence count
        MAX_GROUP = 5
        top_groups = sorted(cleaned.items(), key=lambda kv: (-len(kv[1]), kv[1][0]))[:MAX_GROUP]
        # Sort by first appearance so A appears before B
        top_groups.sort(key=lambda kv: kv[1][0])

        labels = ["A", "B", "C", "D", "E"]
        label_map: Dict[tuple, str] = {fp: labels[i] for i, (fp, _) in enumerate(top_groups)}

        # Build timeline: (measure, fingerprint) sorted by measure
        timeline: List[Tuple[int, tuple]] = []
        for fp, measures in top_groups:
            for m in measures:
                timeline.append((m, fp))
        timeline.sort(key=lambda x: x[0])

        sections: List[Section] = []
        seen_count: Dict[tuple, int] = {}
        for i, (start_m, fp) in enumerate(timeline):
            end_m = (
                timeline[i + 1][0] - 1
                if i + 1 < len(timeline)
                else total_measures
            )
            if end_m < start_m:
                continue
            base_label = label_map.get(fp, "?")
            count = seen_count.get(fp, 0)
            seen_count[fp] = count + 1
            label = base_label if count == 0 else base_label + chr(39) * count
            sections.append(Section(
                label=label,
                start_measure=start_m,
                end_measure=end_m,
                motif_fingerprint=fp,
                is_repeat=(count > 0),
            ))

        return sections

