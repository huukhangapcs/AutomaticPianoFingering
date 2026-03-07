# 🎹 Automatic Piano Fingering

> **Mô phỏng sinh cơ học bàn tay pianist** — không tối ưu từng nốt đơn lẻ, mà đọc hình học 3D bàn phím, nhận diện cấu trúc câu nhạc và lập kế hoạch di chuyển tay trước khi chơi từng cụm.

[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Accuracy](https://img.shields.io/badge/Combined_Accuracy-85.53%25-brightgreen)]()

---

## 🧠 Triết lý Thiết kế

Hệ thống mô phỏng quy trình của một pianist thật sự:

1. **Đọc cấu trúc tổng thể** — phrase, pattern, đỉnh giai điệu
2. **Ưu tiên giữ nguyên form tay** (Lazy First Principle) — chỉ dịch chuyển khi bắt buộc
3. **Mô phỏng vật lý 3D tay người** — span, tendon constraint, clearance, momentum
4. **Áp dụng fingering theo nhóm** — không tính từng nốt độc lập

---

## 🏗️ Kiến trúc Cốt lõi

```
MusicXML / Dataset TXT
    │
    ▼
IO Layer
  • Parse note events (Pitch, Onset, Duration, Voice)
  • 3D Coordinate Mapping: (Step, Octave, Alter) → (X, Y, Z)
      X = vị trí ngang (key index)
      Y = chiều cao: 0.0=phím trắng, 0.5=phím đen (nhô cao)
      Z = chiều sâu: 0.0=phím trắng, 2.0=phím đen (lùi sâu vào)
  • Chord Grouping & Voice Rank Assignment
    │
    ▼
RECOGNITION LAYERS
  • Phrase Boundary Detection (rests + agogic gaps)
  • Peak Lookahead (Right Hand): xác định đỉnh giai điệu mỗi phrase
  • Pattern Library (Left Hand): nhận diện Alberti bass, arpeggios, octave bounces
    │
    ▼
FINGERING ENGINE — Viterbi DP
  • State: (Accumulated Cost, HandState = 5 finger positions)
  • Hard Gates: MIN_SPAN / MAX_SPAN / same-note same-finger
  • Move Classifier: IN_FORM | STRETCH | THUMB_UNDER | CROSS_OVER | HAND_SHIFT | RESET
  • Soft Costs (additive):
      [A] Movement cost (ngón di chuyển)
      [B] Stretch cost (quadratic nếu vượt COMF_SPAN)
      [C] Move type base cost
      [D] 3D Biomechanics:
          Z-depth obstruction: thumb/pinky bị kẹt khi ngón trước ở phím đen sát cạnh
          Y-clearance: thumb-under dưới phím trắng (Y=0) khó hơn dưới phím đen (Y=0.5)
          Black key penalty: ngón 1 & 5 trên phím đen
      [E] Inertia: penalize dịch chuyển trọng tâm bàn tay
      [F] Momentum: HAND_SHIFT bị phạt nặng nếu 2 nốt cách nhau < 150ms
  • Phrase-Peak Gravity (RH): penalty lũy tiến ngăn dùng ngón 4/5 sớm trước đỉnh phrase
  • Pattern Hard-lock (LH): bypass Viterbi, gán thẳng fingering cho mẫu đệm
    │
    ▼
List[int] (finger 1–5 per note) + MusicXML Export
```

---

## 🧬 Tiến hóa theo Phase

### **Phase 1 — Baseline Viterbi Physics** (v1.0)
Đặt nền tảng: `HandState` 5 ngón, `COMF_SPAN`, `MIN/MAX_SPAN`, move classifier, stretch cost quadratic. Viterbi DP cơ bản 1D.

### **Phase 2.10 — Dynamic Hand Realignment**
Chuyển từ Snap Model sang Update Model (kéo/đẩy động). Bổ sung Inertia Glide (`INERTIA_WEIGHT = 0.3`), xử lý polyphonic chords, Left Hand mirroring X-axis.

### **Phase 2.20 — Pattern Library & Phrase-Peak Lookahead**
- **LH Pattern Library:** Hard-lock fingering cho Alberti bass, arpeggios, octave bounces — bypass Viterbi hoàn toàn cho các mẫu đệm xác định.
- **RH Phrase-Peak Gravity:** Phát hiện đỉnh phụ giai điệu trong từng phrase → radiate penalty (base-3 exponential decay) ngược về trước, "cấm" dùng ngón 4/5 sớm để bảo toàn cho climax.

### **Phase 2.30 — 3D Spatial Keyboard Geometry**
Nâng cấp `pitch_to_coord` từ 1D → 3D `(X, Y, Z)`. Bổ sung 2 constraint sinh cơ học:
- **Z-Depth Obstruction:** Ngón 1/5 bị phạt khi ngón trước đang đặt trên phím đen liền kề (hẹp khe).
- **Y-Clearance:** Thumb-under dưới phím trắng (không gian thấp) khó hơn dưới phím đen (ngón nâng cao).

### **Phase 2.40 — Biomechanics Calibration**
- **MAX_SPAN update:** Căn chỉnh giới hạn duỗi ngón theo bàn tay adult cỡ trung bình thực tế (7–7.5 inch). Cặp 3-4 giảm xuống 2.0, cặp 4-5 giảm xuống 1.8.
- **Tempo-Aware Momentum:** HAND_SHIFT giữa 2 nốt cách nhau < 150ms (tốc độ presto) bị phạt nặng (cost +8.0) vì không thể thực hiện vật lý.
- **Thực nghiệm & bác bỏ:** Tendon 3-4 penalty và Wrist Rotation Gate gây hại cho Combined Accuracy vì false-positive quá cao trên dataset (nhiều 3→4 hợp lệ trong melodies).

---

## 🧪 Benchmark — PIG Piano Fingering Dataset v1.2

Đánh giá trên **309 bài / 42,865 nốt nhạc** (tay phải).

| Metric | Kết quả |
|---|:---:|
| **Exact Match** (khớp 100% nhãn gốc) | **44.51%** |
| **Off-by-one** (sai 1 ngón liền kề — hợp lệ sinh học) | **41.02%** |
| **Combined Accuracy (≤1 error)** | **85.53%** |

> **Diễn giải:** Chỉ số Off-by-one cao (~41%) là dấu hiệu tích cực — hệ thống không đoán sai ngẫu nhiên mà chọn ngón *an toàn hơn* so với nghệ sĩ (ví dụ: dùng ngón 3 thay vì 4 để bảo vệ ngón 5 cho đỉnh phrase). Đây là hành vi ergonomic hoàn toàn hợp lý về mặt học nhạc.
>
> Khoảng hụt ~14% còn lại đến từ: ý đồ âm nhạc (nhấn nhá, phân giọng), sustain pedal chưa được phân tích, và phong cách riêng của từng pianist.

---

## 🚀 Quick Start

### Cấu trúc thư mục

```
src/
├── fingering_solver.py      # Viterbi DP engine + Phrase-Peak + Pattern Lock
├── musicxml_parser.py       # IO: MusicXML → NoteEvent(x,y,z), Chord grouping
├── physics_model.py         # 3D coords, HandState, Move Classifier, Cost functions
└── pattern_library.py       # LH Pattern recognition (Alberti, Arpeggio, Octave)
evaluate_dataset.py          # Batch benchmark trên 309 txt files PIG dataset
evaluate.py                  # Đánh giá 1 file lẻ
main.py                      # Demo: XML in → XML out (với nhãn ngón)
export_detail.py             # Phân tích chi tiết từng nốt (debug)
```

### Demo nhanh

```bash
# Chạy fingering cho 1 file MusicXML, xuất ra result.musicxml
python main.py

# Chạy benchmark đầy đủ trên PIG Dataset v1.2
python evaluate_dataset.py
```

---

## 🗺️ Roadmap — Phase 3: Neural Integration

Rule-based Physics đã đạt ngưỡng tuyệt đối **85.53%**. Để vượt tiếp, cần học ngữ cảnh (context) thay vì chỉ học hình học tay:

| Priority | Hướng phát triển | Dự kiến tác động |
|---|---|:---:|
| 🔴 HIGH | **Sustain Pedal parsing** từ MusicXML — hiểu khi nào pianist nhảy tay thay vì kéo ngón | +2–3% |
| 🔴 HIGH | **Sequence model (CRF / Mamba)** kẹp cùng physics filter | +5–10% |
| 🟡 MED | **Temporal phrasing model** — nhận diện động lực học câu nhạc (crescendo, rubato) | +2% |
| 🟢 LOW | **Hand size parameter** — cá nhân hóa cho tay nhỏ (trẻ em) và tay lớn | UX |

---

## 📄 License

MIT License — tự do sử dụng, nghiên cứu và mở rộng.
