"""
pattern_library.py

Lớp tiền xử lý (Pre-processing) cho tay trái (LHS - staff 2).
Sử dụng hình học bước (Step Geometry) để nhận diện các mẫu đệm kinh điển rập khuôn,
từ đó Hard-lock số ngón chuẩn xác 100% trước khi đưa vào Viterbi DP, tránh lãng phí nhánh
và ngăn ngừa lỗi chéo tay.
"""

from typing import List, Dict, Tuple, Optional
from src.musicxml_parser import NoteEvent

def identify_and_lock_patterns(primaries: List[NoteEvent]) -> Dict[int, int]:
    """
    Quét qua danh sách nốt chính (sau khi đã gom hợp âm).
    Tìm kiếm các pattern và trả về Dictionary ánh xạ: note_index -> finger_lock.
    
    Returns:
        locked_fingers: Dict[int, int]
            Key là chỉ số (index) của nốt trong mảng primaries.
            Value là ngón tay yêu cầu (1-5).
    """
    locked_fingers: Dict[int, int] = {}
    n = len(primaries)
    
    i = 0
    while i < n - 3:
        n0, n1, n2, n3 = primaries[i], primaries[i+1], primaries[i+2], primaries[i+3]
        
        # --- Alberti Bass Detection ---
        # Đặc điểm: Low - High - Mid - High
        # Trục X của tay trái đã được đảo ngược (âm), nên Low < High nghĩa là x0 < x1.
        # Chúng ta dùng "sự chênh lệch toạ độ" (Delta X) để kiểm tra mô hình chóp.
        if (n0.x < n1.x) and (n1.x > n2.x) and (n2.x < n3.x) and (n0.x < n2.x):
            # Check duration gap don't exceed a normal accompaniment rhythm
            if abs(n1.onset_division - n0.onset_division) < 3000: # heuristic gap
                # Alberti chuẩn: Ngón 5 cho Bass, 1 cho Top, 2 hoặc 3 cho Mid
                locked_fingers[i]   = 5
                locked_fingers[i+1] = 1
                
                # Tính khoảng cách từ Bass đến Mid để quyết định ngón 2 (quãng hẹp) hay 3 (quãng rộng)
                mid_dist = abs(n2.x - n0.x)
                if mid_dist > 5.0:  # Rộng hơn quãng 4/5
                    locked_fingers[i+2] = 2
                else:
                    locked_fingers[i+2] = 3
                    
                locked_fingers[i+3] = 1
                
                # Jump forward since we locked 4 notes
                i += 4
                continue
                
        # --- Octave Bounce Detection (2 notes) ---
        # Đặc điểm: Nhảy đúng 1 quãng 8 (Khoảng x cách nhau ~ 7.0)
        if i < n - 1:
            nx, ny = primaries[i], primaries[i+1]
            dist = abs(ny.x - nx.x)
            if 6.0 <= dist <= 8.0:
                if nx.x < ny.x: # Nhảy lên
                    locked_fingers[i] = 5
                    locked_fingers[i+1] = 1
                else:           # Nhảy xuống
                    locked_fingers[i] = 1
                    locked_fingers[i+1] = 5
        
        i += 1

    return locked_fingers
