"""
physics_model.py — Mô hình vật lý bàn tay trên bàn phím piano.

Tọa độ bàn phím:
    Mỗi phím được map thành một số thực x trên trục 1D:
    C=0, D=1, E=2, F=3, G=4, A=5, B=6  (+7 mỗi octave)
    Phím đen = white_key_index của phím trắng bên trái + 0.5
    Ví dụ: F#4 = 4*7 + 3 + 0.5 = 31.5

HandState:
    5 tọa độ (x₁, x₂, x₃, x₄, x₅) biểu diễn vị trí 5 ngón tay.
    Dùng Option C (Snap): sau mỗi nốt, bàn tay snap về cấu hình
    comfortable xung quanh ngón đang chơi → HandState là hàm
    deterministic của (note.x, finger).
"""

from __future__ import annotations
from dataclasses import dataclass
from enum import Enum, auto
import math

# ---------------------------------------------------------------------------
# Keyboard Coordinate System
# ---------------------------------------------------------------------------

WHITE_KEY_OFFSET: dict[str, int] = {
    'C': 0, 'D': 1, 'E': 2, 'F': 3, 'G': 4, 'A': 5, 'B': 6
}

BLACK_KEYS: set[tuple[str, float]] = {
    ('C', 1.0), ('D', 1.0), ('F', 1.0), ('G', 1.0), ('A', 1.0),
    ('D', -1.0), ('E', -1.0), ('G', -1.0), ('A', -1.0), ('B', -1.0),
}


def pitch_to_coord(step: str, octave: int, alter: float = 0.0) -> float:
    """Chuyển pitch thành tọa độ thực trên trục bàn phím.

    Args:
        step:   Tên nốt ('C'..'B')
        octave: Quãng tám (thường 0-8)
        alter:  -1.0=flat, 0.0=natural, +1.0=sharp

    Returns:
        Tọa độ x (float), phím đen ở x + 0.5 so với phím trắng bên trái.
    """
    base = octave * 7 + WHITE_KEY_OFFSET[step.upper()]
    return base + alter * 0.5


def is_black_key(step: str, alter: float = 0.0) -> bool:
    """True nếu nốt là phím đen."""
    return (step.upper(), float(alter)) in BLACK_KEYS


# ---------------------------------------------------------------------------
# Span Tables (tay phải, bàn tay adult trung bình)
# ---------------------------------------------------------------------------
# Adjacent finger pairs: (1-2), (2-3), (3-4), (4-5)
# Index 0 = cặp 1-2, index 1 = cặp 2-3, ...

MIN_SPAN:  list[float] = [0.5, 0.5, 0.5, 0.5]   # co tối đa
COMF_SPAN: list[float] = [2.0, 1.5, 1.5, 1.5]   # thoải mái
MAX_SPAN:  list[float] = [4.5, 2.5, 2.5, 2.0]   # duỗi tối đa


# ---------------------------------------------------------------------------
# HandState — Snap Model (Option C)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class HandState:
    """5 tọa độ ngón tay, biểu diễn dưới dạng tuple bất biến.

    pos[0]=x₁(ngón cái), ..., pos[4]=x₅(ngón út)
    Thứ tự luôn: pos[0] < pos[1] < pos[2] < pos[3] < pos[4]
    """
    pos: tuple[float, float, float, float, float]

    @staticmethod
    def snap(note_x: float, finger: int) -> 'HandState':
        """Option C: Tạo HandState canonical bằng cách snap bàn tay
        về cấu hình comfortable xung quanh ngón đang chơi.

        Args:
            note_x: Tọa độ nốt vừa được gán
            finger: Ngón chơi nốt đó (1-indexed, 1=cái, 5=út)

        Returns:
            HandState với pos[finger-1] = note_x, các ngón khác
            được đặt cách nhau đúng COMF_SPAN.
        """
        f = finger - 1  # 0-indexed
        pos = [0.0] * 5
        pos[f] = note_x

        # Spread các ngón bên phải
        for i in range(f + 1, 5):
            pos[i] = pos[i - 1] + COMF_SPAN[i - 1]

        # Spread các ngón bên trái
        for i in range(f - 1, -1, -1):
            pos[i] = pos[i + 1] - COMF_SPAN[i]

        return HandState(tuple(pos))  # type: ignore[arg-type]

    def update(self, note_x: float, finger: int) -> 'HandState':
        """Chỉ dịch chuyển các ngón khác khi khoảng cách vượt quá giới hạn thoải mái."""
        f = finger - 1
        pos = list(self.pos)
        pos[f] = note_x

        # Update right fingers
        for i in range(f + 1, 5):
            span = pos[i] - pos[i - 1]
            if span > COMF_SPAN[i - 1]:
                pos[i] = pos[i - 1] + COMF_SPAN[i - 1]
            elif span < MIN_SPAN[i - 1]:
                pos[i] = pos[i - 1] + MIN_SPAN[i - 1]

        # Update left fingers
        for i in range(f - 1, -1, -1):
            span = pos[i + 1] - pos[i]
            if span > COMF_SPAN[i]:
                pos[i] = pos[i + 1] - COMF_SPAN[i]
            elif span < MIN_SPAN[i]:
                pos[i] = pos[i + 1] - MIN_SPAN[i]

        return HandState(tuple(pos))  # type: ignore[arg-type]

    def assign(self, finger: int, note_x: float) -> 'HandState':
        """Trả về HandState mới sau khi ngón 'finger' chơi 'note_x'.
        Dùng Update model: giữ nguyên vị trí, chỉ tịnh tiến khi quá giới hạn duỗi.
        """
        return self.update(note_x, finger)

    def is_valid(self) -> bool:
        """Kiểm tra tất cả ràng buộc vật lý."""
        for i in range(4):
            span = self.pos[i + 1] - self.pos[i]
            if span < MIN_SPAN[i] or span > MAX_SPAN[i]:
                return False
            if self.pos[i + 1] <= self.pos[i]:
                return False
        return True

    def stretch_cost(self) -> float:
        """Chi phí duỗi ngón: quadratic nếu vượt COMF_SPAN."""
        cost = 0.0
        for i in range(4):
            span = self.pos[i + 1] - self.pos[i]
            if span > COMF_SPAN[i]:
                cost += (span - COMF_SPAN[i]) ** 2
        return cost

    def centroid(self) -> float:
        """Trọng tâm của 5 ngón."""
        return sum(self.pos) / 5.0


# ---------------------------------------------------------------------------
# Movement Type
# ---------------------------------------------------------------------------

class MoveType(Enum):
    IN_FORM    = auto()   # nốt trong comfortable span — lý tưởng
    STRETCH    = auto()   # duỗi ngón, vẫn trong MAX_SPAN
    THUMB_UNDER = auto()  # ascending, cur∈{2,3,4} → next=1
    CROSS_OVER  = auto()  # descending, cur=1 → next∈{2,3}
    HAND_SHIFT  = auto()  # nhảy quãng lớn, toàn bộ bàn tay relocate
    RESET       = auto()  # tại reset point, bàn tay về vị trí mới


# Chi phí cơ bản theo loại di chuyển (soft cost additive)
MOVE_TYPE_COST: dict[MoveType, float] = {
    MoveType.IN_FORM:     0.0,
    MoveType.STRETCH:     0.5,   # reduced from 1.0 to encourage wider hand form before crossing
    MoveType.THUMB_UNDER: 2.5,   # increased slightly
    MoveType.CROSS_OVER:  3.0,   # increased from 2.0 to penalize unnecessary 3-over-1/4-over-1 crossings early
    MoveType.HAND_SHIFT:  5.0,
    MoveType.RESET:       8.0,
}

# ---------------------------------------------------------------------------
# Hard Constraint Weights
# ---------------------------------------------------------------------------

INF = float('inf')

# Soft cost weights (tunable)
MOVEMENT_WEIGHT  = 1.0   # [A] chi phí di chuyển ngón
STRETCH_WEIGHT   = 2.0   # [B] chi phí duỗi quadratic
INERTIA_WEIGHT   = 0.3   # [E] chi phí dịch chuyển trọng tâm bàn tay - reduced from 0.5

THUMB_BLACK_PEN  = 5.0   # [D] ngón 1 trên phím đen
PINKY_BLACK_PEN  = 3.0   # [D] ngón 5 trên phím đen


# ---------------------------------------------------------------------------
# Movement Classifier
# ---------------------------------------------------------------------------

def classify_move(
    f_prev: int, f_curr: int,
    note_prev_x: float, note_curr_x: float,
) -> MoveType:
    """Phân loại kiểu di chuyển giữa hai nốt liên tiếp.

    Args:
        f_prev, f_curr:       Ngón trước và ngón hiện tại (1-5)
        note_prev_x, note_curr_x: Tọa độ hai nốt
    """
    delta_x = note_curr_x - note_prev_x
    pair_idx = min(f_prev, f_curr) - 1  # 0-indexed span pair

    # Khoảng cách giữa hai nốt theo tọa độ
    abs_span = abs(delta_x)

    # HAND_SHIFT: nhảy quãng lớn hơn 1 octave (7 white keys)
    if abs_span > 7.0:
        return MoveType.HAND_SHIFT

    # THUMB_UNDER: ascending, ngón trước ∈ {2,3,4}, ngón sau = 1
    if delta_x > 0 and f_prev in (2, 3, 4) and f_curr == 1:
        return MoveType.THUMB_UNDER

    # CROSS_OVER: descending, ngón trước = 1, ngón sau ∈ {2,3}
    if delta_x < 0 and f_prev == 1 and f_curr in (2, 3):
        return MoveType.CROSS_OVER

    # STRETCH vs IN_FORM: dựa vào span giữa 2 nốt so với comfortable
    # Dùng span tuyệt đối giữa hai tọa độ
    # (không phải span của 1 cặp ngón cố định mà là khoảng đi được)
    comf = COMF_SPAN[pair_idx] if 0 <= pair_idx < 4 else 2.0
    if abs_span <= comf:
        return MoveType.IN_FORM
    else:
        return MoveType.STRETCH


# ---------------------------------------------------------------------------
# Hard Constraint Gate
# ---------------------------------------------------------------------------

def is_valid_transition(
    f_prev: int, f_curr: int,
    note_prev_x: float, note_curr_x: float,
    s_curr: HandState,
    is_reset: bool = False,
) -> bool:
    """Kiểm tra hard constraints. Trả về False nếu transition bị loại.

    Hard constraints:
        1. SAME_NOTE → SAME_FINGER  (ngoại lệ: sau reset)
        2. DIFF_NOTE → DIFF_FINGER
        3. HandState.is_valid() — ORDER + MAX_SPAN
    """
    same_note = math.isclose(note_prev_x, note_curr_x, abs_tol=0.01)

    # Constraint 1: same note → same finger (trừ sau reset)
    if same_note and not is_reset:
        if f_curr != f_prev:
            return False

    # Constraint 2: different note → different finger
    if not same_note:
        if f_curr == f_prev:
            return False

    # Constraint 3: HandState phải hợp lệ về vật lý
    if not s_curr.is_valid():
        return False

    return True


# ---------------------------------------------------------------------------
# Transition Cost (Soft)
# ---------------------------------------------------------------------------

def transition_cost(
    s_prev: HandState,
    s_curr: HandState,
    f_curr: int,
    note_curr_x: float,
    note_curr_is_black: bool,
    move_type: MoveType,
) -> float:
    """Tổng soft cost của transition.

    Args:
        s_prev:              HandState trước khi chơi nốt hiện tại
        s_curr:              HandState sau khi ngón f_curr chơi nốt
        f_curr:              Ngón đang chơi (1-5)
        note_curr_x:         Tọa độ nốt hiện tại
        note_curr_is_black:  True nếu là phím đen
        move_type:           Loại di chuyển đã phân loại

    Returns:
        Chi phí (float, nhỏ hơn = tốt hơn)
    """
    cost = 0.0

    # [A] Movement cost: ngón f_curr di chuyển bao nhiêu?
    f_idx = f_curr - 1  # 0-indexed
    delta = abs(s_curr.pos[f_idx] - s_prev.pos[f_idx])
    cost += MOVEMENT_WEIGHT * delta

    # [B] Stretch cost: hình dạng bàn tay có tự nhiên không?
    cost += STRETCH_WEIGHT * s_curr.stretch_cost()

    # [C] Movement type base cost
    cost += MOVE_TYPE_COST[move_type]

    # [D] Key color penalty
    if note_curr_is_black:
        if f_curr == 1:
            cost += THUMB_BLACK_PEN
        elif f_curr == 5:
            cost += PINKY_BLACK_PEN

    # [E] Inertia: penalize dịch chuyển trọng tâm bàn tay
    centroid_shift = abs(s_curr.centroid() - s_prev.centroid())
    cost += INERTIA_WEIGHT * centroid_shift

    # [F] Removed Finger index soft penalty to allow high extensions (4/5)
    
    return cost


# ---------------------------------------------------------------------------
# Reset Point Detection
# ---------------------------------------------------------------------------

def is_reset_point(
    note_duration: int, 
    divisions: int, 
    rest_before: bool,
    tempo: float = 120.0,
    note_prev_x: float = 0.0,
    note_curr_x: float = 0.0
) -> bool:
    """True nếu nốt hiện tại là điểm có thể reset bàn tay.
    Cho phép reset nếu:
    1. Nghỉ > 1 beat hoặc nốt giữ dài > 2 beats dài.
    2. Vận tốc tịnh tiến (Velocity) tay trên phím không văng quá giới hạn vật lý.

    Args:
        note_duration: Duration của nốt (hoặc rest) trước đó, in divisions.
        divisions:     Số divisions per quarter note.
        rest_before:   True nếu có rest trước note đó.
        tempo:         BPM (Beats per minute).
        note_prev_x:   Tọa độ nốt trước đó (mm).
        note_curr_x:   Tọa độ nốt hiện tại (mm).
    """
    beats = note_duration / divisions
    
    # Check basic musical boundary
    is_musical_boundary = rest_before and beats > 1.0 or beats > 2.0
    if not is_musical_boundary:
        return False
        
    # Check physical velocity constraint
    # 1 Beat (Quarter note) = 60000 / tempo (ms)
    time_ms = beats * (60000.0 / tempo)
    distance_mm = abs(note_curr_x - note_prev_x)
    
    # Tốc độ di chuyển tay (mm/s)
    # Giới hạn vật lý con người ~ 2000 mm/s cho những cú nhảy cục bộ
    velocity_mm_s = (distance_mm / time_ms) * 1000.0 if time_ms > 0 else 0
    MAX_VELOCITY = 2000.0
    
    if velocity_mm_s > MAX_VELOCITY:
        return False
        
    return True
