"""
fingering_solver.py — Viterbi DP solver cho bài toán gán ngón tay.

Thuật toán:
    - State: finger ∈ {1,2,3,4,5} tại mỗi nốt
    - HandState được reconstruct từ (note.x, finger) qua Snap model (Option C)
      → HandState = f(note.x, finger) là deterministic
    - Transition: kiểm tra hard constraints, rồi tính soft cost
    - RESET được tích hợp như checkpoint node trong DP:
      tại mỗi reset point, solver có THÊM 5 lựa chọn
      (dùng bất kỳ ngón nào như ngón đầu tiên, cộng RESET cost)
    - Chord handling: nốt cao nhất (chord_rank=0) được solver gán finger f
      các nốt còn lại nhận f+1, f+2, ... (từ cao xuống thấp)
"""

from __future__ import annotations
from typing import List, Optional, Tuple
import math

from src.physics_model import (
    HandState, MoveType,
    transition_cost, is_valid_transition, classify_move,
    is_reset_point,
    INF, COMF_SPAN, MAX_SPAN, MOVE_TYPE_COST,
)
from src.musicxml_parser import NoteEvent, get_primary_notes, get_chord_secondaries


# ---------------------------------------------------------------------------
# Initial State
# ---------------------------------------------------------------------------

def _initial_costs(note: NoteEvent) -> dict[int, Tuple[float, HandState]]:
    """Chi phí khởi đầu cho nốt đầu tiên, thử tất cả 5 ngón."""
    costs = {}
    for f in range(1, 6):
        s = HandState.snap(note.x, f)
        if not s.is_valid():
            costs[f] = (INF, s)
            continue
        cost = 0.0
        # Neutral point mapping - removed bias against fingers 1/5 to ensure broader Viterbi state tree search
        finger_pref = {1: 0.1, 2: 0.0, 3: 0.0, 4: 0.1, 5: 0.2}
        cost += finger_pref[f]
        costs[f] = (cost, s)
    return costs


# ---------------------------------------------------------------------------
# Main Viterbi Solver
# ---------------------------------------------------------------------------

def solve(
    all_notes: List[NoteEvent],
    divisions: int,
    tempo: float = 120.0,
    is_lh: bool = False
) -> List[Tuple[NoteEvent, int]]:
    """Gán ngón tay tối ưu cho toàn bộ notes tay phải.

    Args:
        all_notes:  Tất cả NoteEvent (kể cả chord members)
        divisions:  Số division per quarter note
        tempo:      BPM (Beats per minute)
        is_lh:      Tay trái hay phải (True = Tay trái -> Kích hoạt Pattern Lock, False = Tay Phải -> Kích hoạt Lookahead)


    Returns:
        List[(NoteEvent, finger)] cho TẤT CẢ notes
        (chord members được gán tự động từ primary finger)
    """
    # Chỉ solve cho primary notes (chord_rank=0)
    primaries = get_primary_notes(all_notes)
    if not primaries:
        return []

    N = len(primaries)

    # DP tables
    # dp[f] = (min_cost, prev_finger)  tại bước hiện tại
    # Dùng 2 dict thay vì 2D array để readable hơn

    # --- Step 0: khởi tạo ---
    dp_curr: dict[int, Tuple[float, HandState]] = {}
    bt_curr: dict[int, Optional[int]] = {}  # backtrack: ngón tối ưu từ bước trước

    init = _initial_costs(primaries[0])
    for f in range(1, 6):
        if f in init:
            dp_curr[f] = init[f]
        else:
            dp_curr[f] = (INF, HandState.snap(primaries[0].x, f))
        bt_curr[f] = None

    # Lưu toàn bộ backtrack table
    # backtrack[i][f] = f_prev tối ưu dẫn đến (note[i], finger=f)
    backtrack: list[dict[int, Optional[int]]] = [dict(bt_curr)]

    prev_note = primaries[0]
    prev_dur = primaries[0].duration
    prev_is_rest = False

    # --- Pre-processing: Mẫu Tay Trái (LHS Pattern Matching) ---
    locked_fingers: dict[int, int] = {}
    if is_lh:
        from src.pattern_library import identify_and_lock_patterns
        locked_fingers = identify_and_lock_patterns(primaries)
        # Lock first note if it's part of a pattern
        if 0 in locked_fingers:
            req_f = locked_fingers[0]
            for f in range(1, 6):
                if f != req_f:
                    dp_curr[f] = (INF, HandState.snap(primaries[0].x, f))

    # --- Pre-processing: Phase-Peak Lookahead (RHS Gravity Penalty) ---
    lookahead_penalties: dict[int, dict[int, float]] = {i: {f: 0.0 for f in range(1, 6)} for i in range(N)}
    if not is_lh:
        # Step 1: Phân tách bản nhạc thành các Phrases bằng `is_reset_point`
        phrases: List[List[int]] = []
        curr_phrase = [0]
        
        for i in range(1, N):
            p_note = primaries[i]
            prev = primaries[i-1]
            if is_reset_point(prev.duration, divisions, prev_is_rest, tempo, prev.x, p_note.x):
                phrases.append(curr_phrase)
                curr_phrase = [i]
            else:
                curr_phrase.append(i)
        if curr_phrase:
            phrases.append(curr_phrase)
            
        # Step 2 & 3: Tìm Đỉnh (Peak) trong mỗi Phrase và Gắn Trọng lực dội ngược
        for phrase in phrases:
            if len(phrase) < 4:
                continue # Bỏ qua câu quá ngắn
                
            # Tìm note cao nhất trên trục X
            peak_idx = max(phrase, key=lambda idx: primaries[idx].x)
            peak_x = primaries[peak_idx].x
            
            # Decay Function: Bắn ngược penalty cho ngón 5 (để dành ngón 5 cho đỉnh)
            # Phạm vi ảnh hưởng: 5 notes trước đỉnh
            for step_back in range(1, 4):
                target_idx = peak_idx - step_back
                if target_idx not in phrase:
                    break
                    
                # Chỉ phạt các đoạn đi lên dốc tới đỉnh (Ascending)
                if primaries[target_idx].x < peak_x:
                    # Decay mũ: 5.0 -> 1.66 -> 0.55
                    penalty = 5.0 / (3 ** (step_back - 1))
                    
                    # Phạt ngón 5 (và ngón 4) nếu xài sớm
                    lookahead_penalties[target_idx][5] += penalty
                    lookahead_penalties[target_idx][4] += penalty * 0.4

    # Precompute chord dict for fast lookup: onset_division -> list of note.x
    chord_xs_by_onset: dict[int, List[float]] = {}
    for n in all_notes:
        if n.onset_division not in chord_xs_by_onset:
            chord_xs_by_onset[n.onset_division] = []
        chord_xs_by_onset[n.onset_division].append(n.x)
        
    for i in range(1, N):
        note = primaries[i]
        dp_next: dict[int, Tuple[float, HandState]] = {f: (INF, HandState.snap(note.x, f)) for f in range(1, 6)}
        bt_next: dict[int, Optional[int]] = {f: None for f in range(1, 6)}

        # Detect reset point: rest trước note này, hoặc note trước rất dài
        # Mới: Check thêm vận tốc tịnh tiến (Velocity) không vượt ngưỡng
        at_reset = is_reset_point(
            note_duration=prev_dur, 
            divisions=divisions, 
            rest_before=prev_is_rest,
            tempo=tempo,
            note_prev_x=prev_note.x,
            note_curr_x=note.x
        )
        
        # Mới: True Polyphony Span Penalties
        # Tìm nốt dãn xa nhất (thường là nốt thấp nhất ở rank cuối)
        current_chord_xs = chord_xs_by_onset.get(note.onset_division, [])
        max_chord_span = 0.0
        if current_chord_xs:
            max_chord_span = max(current_chord_xs) - min(current_chord_xs)

        for f_curr in range(1, 6):
        
            # --- Mới: Pattern Bypass ---
            # Nếu index i của nốt hiện tại đã bị hard-lock bởi Pattern Library
            if is_lh and i in locked_fingers:
                if f_curr != locked_fingers[i]:
                    continue # Bỏ qua hoàn toàn nhánh ngón tay này
        
            # Kiểm tra nếu f_curr cho nốt đỉnh kết hợp với bề rộng hợp âm vượt quá sải tay con người.
            # Giả định nốt dưới cùng do ngón 5 (hoặc ngón 1 cho tay trái) phụ trách. COMF_SPAN * 2.5 ~ MAX SPAN thực tế.
            # Rất thô sơ nhưng hiệu quả chặn gãy tay.
            if f_curr == 5 and max_chord_span > 0:
                # Nếu đỉnh dùng ngón 5, đáy dùng ngón 1 -> Span tối đa 1-5
                if max_chord_span > 15.0: # Khoảng 1 quãng 9-10
                    continue
            elif f_curr < 5 and max_chord_span > 0:
                # Nếu đỉnh dùng ngón 3, 4 -> Đáy dùng ngón phụ hoặc 1. Span thu hẹp lại.
                if max_chord_span > (15.0 - (5-f_curr)*3.0): 
                    continue

            for f_prev in range(1, 6):
                cost_prev, s_prev = dp_curr[f_prev]
                if cost_prev >= INF:
                    continue

                # Chỉ dịch chuyển tay nếu vượt qua khoảng cách thoải mái (Dynamic realignment)
                s_curr_update = s_prev.update(note.x, f_curr)
                s_curr_reset = HandState.snap(note.x, f_curr)

                # --- Hard constraint gate ---
                valid = is_valid_transition(
                    f_prev, f_curr,
                    prev_note.x, note.x,
                    s_curr_update,
                    is_reset=False,
                )

                # Nếu bình thường invalid nhưng tại reset point
                # thì thử lại với is_reset=True (hồi tay về form thoải mái quanh nốt)
                is_reset_move = False
                if not valid and at_reset:
                    valid = is_valid_transition(
                        f_prev, f_curr,
                        prev_note.x, note.x,
                        s_curr_reset,
                        is_reset=True,
                    )
                    is_reset_move = valid
                    s_curr = s_curr_reset
                else:
                    s_curr = s_curr_update

                if not valid:
                    continue

                # --- Phân loại movement ---
                if is_reset_move:
                    move_type = MoveType.RESET
                else:
                    move_type = classify_move(
                        f_prev, f_curr, prev_note.x, note.x
                    )

                # --- Soft cost ---
                cost = cost_prev + transition_cost(
                    s_prev=s_prev,
                    s_curr=s_curr,
                    f_curr=f_curr,
                    note_curr_x=note.x,
                    note_curr_is_black=note.is_black,
                    move_type=move_type,
                )
                
                # --- Mới: Cộng dồn Phase-Peak Gravity Penalty (nếu có) ---
                if not is_lh:
                    cost += lookahead_penalties[i][f_curr]

                # Update nếu đường đi này tốt hơn
                if cost < dp_next[f_curr][0]:
                    dp_next[f_curr] = (cost, s_curr)
                    bt_next[f_curr] = f_prev

        dp_curr = dp_next
        backtrack.append(dict(bt_next))
        prev_note = note
        # Track nếu có rest trước note tiếp theo
        # (dùng onset gap để detect rest)
        if i + 1 < N:
            next_note = primaries[i + 1]
            expected_next = note.onset_division + note.duration
            prev_is_rest = (next_note.onset_division > expected_next)
            prev_dur = note.duration
        else:
            prev_is_rest = False
            prev_dur = note.duration

    # --- Backtracking ---
    # Tìm ngón tốt nhất cho nốt cuối
    best_final = min(range(1, 6), key=lambda f: dp_curr[f][0])
    if dp_curr[best_final][0] >= INF:
        # Fallback: gán tất cả finger=3 nếu không có path hợp lệ
        finger_seq = [3] * N
    else:
        finger_seq = [0] * N
        finger_seq[N - 1] = best_final
        for i in range(N - 2, -1, -1):
            finger_seq[i] = backtrack[i + 1][finger_seq[i + 1]] or 3

    # --- Build result: primary notes ---
    results: List[Tuple[NoteEvent, int]] = []
    primary_assignments: dict[str, int] = {}  # note_id → finger

    for i, note in enumerate(primaries):
        f = finger_seq[i]
        results.append((note, f))
        primary_assignments[note.note_id] = f

    # --- Chord members: gán finger tự động ---
    results = _assign_chord_members(all_notes, primaries, finger_seq, results)

    return results


def _assign_chord_members(
    all_notes: List[NoteEvent],
    primaries: List[NoteEvent],
    finger_seq: List[int],
    results: List[Tuple[NoteEvent, int]],
) -> List[Tuple[NoteEvent, int]]:
    """Gán ngón cho các nốt phụ trong chord (chord_rank > 0).

    Strategy: nốt thấp hơn trong chord → ngón cao hơn (số lớn hơn).
    Ví dụ: primary (highest) = finger 2 → next lower = finger 3 → finger 4 ...
    Nếu vượt quá 5, dùng finger 5.
    """
    # Build lookup: (onset_division, voice) → primary_finger
    primary_map: dict[tuple[int, int], int] = {}
    for primary, f in zip(primaries, finger_seq):
        primary_map[(primary.onset_division, primary.voice)] = f

    for note in all_notes:
        if note.chord_rank == 0:
            continue  # Đã xử lý trong results
        key = (note.onset_division, note.voice)
        primary_f = primary_map.get(key, 3)
        # Chord rank 1 → primary_f + 1, rank 2 → primary_f + 2, ...
        chord_finger = min(primary_f + note.chord_rank, 5)
        results.append((note, chord_finger))

    # Sort theo onset_division để output theo thứ tự
    results.sort(key=lambda x: (x[0].onset_division, x[0].chord_rank))
    return results


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

def solve_file(musicxml_path: str, is_lh: bool = False) -> List[Tuple[NoteEvent, int]]:
    """Parse + solve trong 1 lần gọi."""
    from src.musicxml_parser import parse_hand_notes
    staff_id = 2 if is_lh else 1
    notes, divisions, tempo = parse_hand_notes(musicxml_path, staff_id)
    return solve(notes, divisions, tempo, is_lh)
