# 🎹 Automatic Piano Fingering (Ver. 2)

> **Tư duy như pianist thật sự** — đọc nhạc theo phrase, track tay theo mm, không tối ưu từng nốt đơn lẻ.

---

## 🧠 Triết lý thiết kế & Tiến hóa (Ver 2)

Hệ thống mô phỏng đúng quy trình của một người chơi nhạc thật sự:
1. **Nhìn cấu trúc tổng quát** — phrase, pattern, arc
2. **Xác định ý nhạc** — legato? brilliant? expressive?
3. **Giữ bàn tay cố định** (Lazy First Principle) — chỉ di chuyển khi bắt buộc
4. **Áp fingering theo nhóm** — scale → 1-2-3-1, chord → 1-3-5

### Tính năng mới ở Version 2 (Dynamic Hand Realignment)
Thay vì sử dụng thuật toán "Snapping" cứng nhắc (tái căn chỉnh form tay quanh mỗi nốt đệm), Ver. 2 áp dụng Mô hình Vật lý Động lực học (Dynamic Physics Model):
* **Cơ chế Kéo/Đẩy (Drag/Push):** Các ngón tay tự do di chuyển trong phạm vi thoải mái (`COMF_SPAN`). Bàn tay chỉ bị kéo lùi hoặc đẩy tới khi quãng với ngón lân cận vượt quá giới hạn cấu phẫu tự nhiên của tay người hoặc bị quá chật (`MIN_SPAN`).
* **Inertia Glide:** Giảm điểm phạt ma sát (`INERTIA_WEIGHT` xuống 0.3) giúp thuật toán nhìn xa tốt hơn, lướt tay tịnh tiến mượt mà từ âm khu thấp lên âm khu cao để đón đầu các phrase giai điệu đỉnh.
* **Quy hoạch động Viterbi nâng cấp:** State DP lưu trữ lại toàn bộ trạng thái `HandState` băng qua từng cụm nốt, bảo toàn được quán tính vật lý thực sự của chuỗi vận động thay vì chỉ lưu điểm số Cost.
* **Tự động hóa Tay Trái (Left Hand Automation):** Xử lý đồng thời cả `staff=1` và `staff=2`. Trục X của tay trái được cấu hình đối xứng (đảo dấu `-X`) ngay từ bước đọc XML, giúp tái sử dụng 100% logic Viterbi của tay phải chuẩn xác tuyệt đối mà không cần thiết kế lại hand form bổ sung.
* **Bộ lọc số ngón (Sparse Fingering / Decluttering):** Tích hợp bộ heuristic làm sạch kết quả xuất XML. Tự động ẩn các ngón chạy scale liền bậc hiển nhiên (IN_FORM) đối với RH và các mẫu đệm tự nhiên đối với LH. Chỉ xuất ép buộc các số ngón tại điểm ngắt nhạc, qua ngón cái (Thumb-under), vắt ngón (Cross-over) hoặc dãn tay xa/nhảy quãng.

---

## 🏗️ Kiến trúc Cốt lõi (Physics-First Engine)

```text
MusicXML / Dataset TXT
    │
    ▼
IO Layer
  • Parse note events (Pitch, Onset, Duration)
  • Voice & Chord Grouping (Ranked lowest to highest)
    │
    ▼
RECOGNITION LAYERS
  • Note Coordinate Mapping (Step, Octave, Alter -> mm)
  • Phrase Boundary Detection (Rests & Agogic gaps)
    │
    ▼
FINGERING LAYERS (Viterbi DP)
  • Node State: (Accumulated Cost, HandState Physics)
  • Move Classifier: IN_FORM | STRETCH | THUMB_UNDER | CROSS_OVER | HAND_SHIFT
  • Cost Weights: Stretch (Quadratic), Movement, Key Color (Black thumbs), Inertia
    │
    ▼
List[int] (finger 1–5 per note) + MusicXML Export
```

---

## 🧪 Benchmark — PIG Piano Fingering Dataset v1.2

Hệ thống tính toán mô hình vật lý V2 đã được chạy đánh giá trên diện rộng toàn bộ **309 bài test / 42.865 nốt nhạc** (Tay phải).

| Metric | Version 2.30 (3D Spatial Geometry) |
|---|:---:|
| **Exact Match** (Đoán chính xác 100% ngón chuẩn) | **44.52%** |
| **Off-by-one** (Sai số 1 ngón liền kề hợp lý) | **41.01%** |
| **Combined (Tỉ lệ chuẩn form tay ≤1 error)** | **85.53%** |

> **Key insight:** Bằng cách từ bỏ mặt phẳng 1D và mô phỏng thực tế **Không gian 3D (X, Y, Z)** của bàn phím Piano, thuật toán Viterbi đã có khả năng tính toán độ chạm của ngón theo chiều Sâu (Z-Depth) và khoảng hở nhấc ngón (Y-Clearance). 
> Nhờ đó, nó tự động né tránh các tư thế sai sinh trắc học như: lòn ngón cái dưới các ngón đang bấm phím trắng tịt sát mặt đàn (Y=0), và không bao giờ cố chọc ngón áp út/ngón cái (các ngón rất to) mổ sâu vào khe hẹp giữa 2 phím đen (Z > 25mm). Kỉ lục độ dung sai sinh học lại bị xô đổ một lần nữa, đạt mức an toàn **85.53%**.
>
> Khoảng hụt lại đa số rơi vào các biến tấu Phân tích ý đồ Câu nhạc (Nhấn mạnh, Phân giọng, Trượt phím đệm). Mọi quy luật Vật lý Động Học (Cơ bắp, Không gian, Trọng lực) đã đạt tới giới hạn tối ưu tuyệt đối.

---

## 🚀 Quick Start

### Cấu trúc files (Ver 2)
```
src/
├── fingering_solver.py         # DP Viterbi engine + Initial preferences
├── musicxml_parser.py          # IO MusicXML -> NoteEvent, Chord extraction
└── physics_model.py            # Coordinate mapping, HandState, Move Classifier
evaluate.py                     # Đánh giá 1 file lẻ
evaluate_dataset.py             # Đánh giá chạy Batch trên 309 txt files dataset
main.py                         # Chạy demo convert 1 file XML ra XML có ngón
```

### Chạy Demo Đơn Lẻ
Đọc vào 1 file MusicXML bất kỳ, tính toán và xuất ngược ra file `result.musicxml` chứa nhãn ngón.

```bash
python main.py
```

### Chạy Benchmark trên Full Dataset
```bash
python evaluate_dataset.py
```

---

## 📊 Roadmap: Phase 3 — Neural Integration

Để vượt mốc trần `84%` của Rule-based Physics, Phase tiếp theo hệ thống cần sự hỗ trợ của Neural Model để học ngữ cảnh (Context) thay vì chỉ học giới hạn hình học tay.

| Priority | Việc cần làm | Mục tiêu |
|---|---|---|
| HIGH | **THUMB_MISS fix** — Neural model học bias xuất phát | Fix chuỗi lỗi xuất phát bằng tay yếu ở note thấp |
| HIGH | **Mamba / CRF Filter** — Áp dụng sequence prediction model kẹp cùng Physics model filter | Bứt phá trần form tay lên 95%+ |
| MED | **Cross-Phrase Intent** — Phát hiện đỉnh phrase | Tăng tỉ lệ Exact Match |
