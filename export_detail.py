#!/usr/bin/env python3
"""
export_detail.py — Export chi tiết fingering + loại di chuyển ra file .txt

Usage:
    python export_detail.py --input test_file/FN7ALfpGxiI.musicxml --output output/detail.txt
"""

import argparse
import os
import sys
import math

sys.path.insert(0, os.path.dirname(__file__))

from src.musicxml_parser import parse_hand_notes, get_primary_notes, NoteEvent
from src.fingering_solver import solve
from src.physics_model import (
    HandState, MoveType, classify_move, is_valid_transition,
    is_reset_point, pitch_to_coord, COMF_SPAN,
)


# ─────────────────────────────────────────────────────────────────────────────
# Movement label helpers
# ─────────────────────────────────────────────────────────────────────────────

MOVE_ICONS = {
    MoveType.IN_FORM:     "◆ IN_FORM",
    MoveType.STRETCH:     "↔ STRETCH",
    MoveType.THUMB_UNDER: "⟱ THUMB_UNDER",
    MoveType.CROSS_OVER:  "⟰ CROSS_OVER",
    MoveType.HAND_SHIFT:  "⇒ HAND_SHIFT",
    MoveType.RESET:       "↺ RESET",
}

MOVE_DESC = {
    MoveType.IN_FORM:
        "Nốt trong comfortable span — bàn tay không cần duỗi thêm",
    MoveType.STRETCH:
        "Duỗi ngón để với tới nốt — vẫn trong giới hạn sinh học",
    MoveType.THUMB_UNDER:
        "Ngón cái luồn dưới (ascending), tạo điểm tựa cho chuỗi tiếp theo",
    MoveType.CROSS_OVER:
        "Ngón vượt qua ngón cái (descending), tiếp tục chuỗi đi xuống",
    MoveType.HAND_SHIFT:
        "Nhảy quãng lớn (>1 octave), toàn bộ bàn tay relocate",
    MoveType.RESET:
        "Reset bàn tay tại điểm nghỉ / nốt dài, bắt đầu phrase mới",
}


def _detect_move_type(
    f_prev: int, f_curr: int,
    note_prev: NoteEvent, note_curr: NoteEvent,
    at_reset: bool,
) -> MoveType:
    """Phân loại chuyển động trực tiếp từ (f_prev, f_curr, delta_x)."""
    delta_x = note_curr.x - note_prev.x
    abs_dx = abs(delta_x)

    # RESET: tại reset point mà ngón thay đổi bất thường
    if at_reset and math.isclose(note_prev.x, note_curr.x, abs_tol=0.01) is False:
        # Nếu là same note → không phải reset nhưng vẫn tại reset point
        pass

    # HAND_SHIFT: nhảy quãng lớn hơn 1 octave
    if abs_dx > 7.0:
        return MoveType.HAND_SHIFT

    # THUMB_UNDER: ascending, f_prev ∈ {2,3,4}, f_curr = 1
    if delta_x > 0 and f_prev in (2, 3, 4) and f_curr == 1:
        return MoveType.THUMB_UNDER

    # CROSS_OVER: descending, f_prev = 1, f_curr ∈ {2,3}
    if delta_x < 0 and f_prev == 1 and f_curr in (2, 3):
        return MoveType.CROSS_OVER

    # RESET: at reset point và ngón tay thay đổi không theo quy luật IN_FORM/STRETCH
    if at_reset and abs(f_curr - f_prev) > 2:
        return MoveType.RESET

    # STRETCH: nếu khoảng cách x lớn hơn comfortable
    pair_idx = min(f_prev, f_curr) - 1
    comf = COMF_SPAN[pair_idx] if 0 <= pair_idx < 4 else 2.0
    if abs_dx > comf:
        return MoveType.STRETCH

    return MoveType.IN_FORM


# ─────────────────────────────────────────────────────────────────────────────
# Main export
# ─────────────────────────────────────────────────────────────────────────────

def export_detail(input_path: str, output_path: str):
    notes, divisions, tempo = parse_hand_notes(input_path)
    assignments = solve(notes, divisions)

    # Chỉ lấy primary notes (chord_rank=0)
    primary_assignments = [(n, f) for n, f in assignments if n.chord_rank == 0]

    lines = []

    # ── Header ──────────────────────────────────────────────────────────────
    lines.append("=" * 72)
    lines.append("  AUTOMATIC PIANO FINGERING — DETAILED MOVEMENT LOG")
    lines.append(f"  File   : {os.path.basename(input_path)}")
    lines.append(f"  Tempo  : {tempo} BPM  |  Divisions : {divisions}/quarter")
    lines.append(f"  Notes  : {len(primary_assignments)} (primary, chord_rank=0)")
    lines.append("=" * 72)
    lines.append("")

    # ── Legend ───────────────────────────────────────────────────────────────
    lines.append("LEGEND — MOVEMENT TYPES:")
    for mt in MoveType:
        lines.append(f"  {MOVE_ICONS[mt]:<18}  {MOVE_DESC[mt]}")
    lines.append("")
    lines.append("COLUMNS: Meas | Note | x-coord | Finger (Pred/GT) | Move Type | HandState | Note")
    lines.append("-" * 72)
    lines.append("")

    # ── Per-note detail ───────────────────────────────────────────────────────
    prev_note = None
    prev_finger = None
    prev_dur = 0
    prev_is_rest = False
    current_measure = -1

    # Build a fast lookup: note_id → assigned finger (including chord members)
    assignment_map = {n.note_id: f for n, f in assignments}

    stats: dict[MoveType, int] = {mt: 0 for mt in MoveType}

    for idx, (note, finger) in enumerate(primary_assignments):

        # ── Measure header ──────────────────────────────────────────────────
        if note.measure != current_measure:
            current_measure = note.measure
            lines.append(f"┌─── MEASURE {current_measure} {'─' * 56}")

        # ── Movement classification ─────────────────────────────────────────
        if prev_note is None:
            move_label = "── START"
            move_detail = "Nốt đầu tiên — khởi tạo HandState"
            hand_str = _hand_str(HandState.snap(note.x, finger))
        else:
            at_reset = is_reset_point(prev_dur, divisions, prev_is_rest)
            move_type = _detect_move_type(prev_finger, finger, prev_note, note, at_reset)
            move_label = MOVE_ICONS[move_type]
            move_detail = MOVE_DESC[move_type]
            hand_str = _hand_str(HandState.snap(note.x, finger))
            stats[move_type] += 1

            # Thêm chi tiết số lượng
            delta_x = note.x - prev_note.x
            dir_str = "↑" if delta_x > 0 else ("↓" if delta_x < 0 else "=")
            span_str = f"Δx={delta_x:+.1f} {dir_str}"
            move_detail = f"{move_detail}  [{span_str}]"

        # ── Match indicator ─────────────────────────────────────────────────
        gt_str = f"GT={note.gt_finger}" if note.gt_finger else "GT=-"
        match_icon = "✓" if note.gt_finger == finger else "✗"

        # ── Chord info ──────────────────────────────────────────────────────
        # Lấy các nốt trong chord (nếu có)
        chord_others = [
            (n, f2) for n, f2 in assignments
            if n.onset_division == note.onset_division
               and n.voice == note.voice
               and n.chord_rank > 0
        ]
        chord_str = ""
        if chord_others:
            parts = [f"{n.step}{n.octave}→f{f2}" for n, f2 in sorted(chord_others, key=lambda x: x[0].chord_rank)]
            chord_str = f"  +chord[{', '.join(parts)}]"

        # ── Write line ──────────────────────────────────────────────────────
        note_name = f"{note.step}{note.octave}"
        black_mark = "♯" if note.is_black else " "

        lines.append(
            f"│ m{note.measure:<4} {note_name:<3}{black_mark} "
            f"x={note.x:<5.1f} "
            f"f={finger} ({gt_str}) {match_icon}  "
            f"{move_label:<18}"
        )
        lines.append(
            f"│       Hand: {hand_str}"
            f"{chord_str}"
        )
        lines.append(f"│       ↳ {move_detail}")
        lines.append("│")

        # ── Update state ────────────────────────────────────────────────────
        prev_note = note
        prev_finger = finger
        prev_dur = note.duration

        if idx + 1 < len(primary_assignments):
            next_note = primary_assignments[idx + 1][0]
            expected_next = note.onset_division + note.duration
            prev_is_rest = (next_note.onset_division > expected_next)
        else:
            prev_is_rest = False

    lines.append("└" + "─" * 71)
    lines.append("")

    # ── Summary stats ─────────────────────────────────────────────────────────
    total_moves = sum(stats.values())
    lines.append("=" * 72)
    lines.append("  MOVEMENT TYPE STATISTICS")
    lines.append("=" * 72)
    lines.append(f"  {'Type':<20} {'Count':>6}  {'%':>6}  Bar")
    lines.append("  " + "-" * 50)
    for mt in MoveType:
        count = stats[mt]
        pct = count / total_moves * 100 if total_moves > 0 else 0
        bar = "█" * int(pct / 2)
        lines.append(f"  {MOVE_ICONS[mt]:<20} {count:>6}  {pct:>5.1f}%  {bar}")
    lines.append("")
    lines.append(f"  Total transitions classified: {total_moves}")
    lines.append("=" * 72)

    # ── Write file ────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(output_path), exist_ok=True) if os.path.dirname(output_path) else None

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"✅ Exported {len(primary_assignments)} notes → {output_path}")
    print(f"   Movement stats: {dict((MOVE_ICONS[k], v) for k,v in stats.items() if v > 0)}")


def _hand_str(s: HandState) -> str:
    """In HandState dưới dạng compact: [x1|x2|x3|x4|x5]"""
    return "[" + "|".join(f"{p:.1f}" for p in s.pos) + "]"


# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Export detailed fingering movement log')
    parser.add_argument('--input', required=True, help='Input MusicXML file')
    parser.add_argument('--output', default='output/detail.txt', help='Output .txt file')
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"❌ File không tồn tại: {args.input}")
        sys.exit(1)

    export_detail(args.input, args.output)


if __name__ == '__main__':
    main()
