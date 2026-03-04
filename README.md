# 🎹 Automatic Piano Fingering

> **Tư duy như pianist thật sự** — hệ thống automatic fingering đọc nhạc theo phrase, không phải từng nốt đơn lẻ.

---

## 🧠 Triết lý thiết kế

Pianist thật sự **không** đọc từng nốt rồi tìm ngón. Họ:
1. **Nhìn tổng quát cấu trúc** — nhận ra phrase, pattern, harmonic arc
2. **Xác định ý nhạc** — legato? brilliant? expressive?
3. **Áp dụng fingering theo nhóm** — scale → 1-2-3-1-2-3, chord → 1-3-5

Hệ thống này mô phỏng đúng quy trình đó thay vì tối ưu ergonomic từng nốt theo kiểu brute-force.

---

## 🏗️ Kiến trúc

```
MusicXML File
     │
     ▼
┌─────────────────────────────────────────────────────┐
│  IO Layer: MusicXMLReader                           │
│  • Grand staff support (staff 1=RH, staff 2=LH)     │
│  • Per-staff time tracking với <backup> elements     │
│  • Slur, staccato, accent, fingering GT parsing     │
└───────────────────┬─────────────────────────────────┘
                    │ List[NoteEvent]
                    ▼
┌─────────────────────────────────────────────────────┐
│  Phrase-Aware Fingering Pipeline (PAF)              │
│                                                     │
│  Layer A: PhraseBoundaryDetector                    │
│    • Signal fusion (slur, rest, cadence, dynamic)   │
│    • [Imp 1] Melodic Arc Detector (ARCH/CLIMB/FALL) │
│    • [Imp 2] Metric Position Weighting (downbeat)   │
│    • [Imp 3] Phrase Length Prior (2/4/8 measures)   │
│    • [Fix 1] Forced segmentation (max 4 measures)   │
│                                                     │
│  Layer B: PhraseIntentAnalyzer                      │
│    • CANTABILE / BRILLIANT / LEGATO / STACCATO      │
│    • Tension curve + climax note detection          │
│                                                     │
│  Layer C: PhraseScopedDP (Viterbi)                  │
│    • Ergonomic cost: span, crossing, black keys     │
│    • [Fix 2] Pattern Library injection              │
│        – Major/minor scale → 1-2-3-1-2-3-4-5       │
│        – Partial scale (4+ notes) detection         │
│        – Broken chord arpeggio → 1-2-4 / 5-3-1     │
│    • [Fix 3] Chord Heuristic (simultaneous notes)   │
│        – RH chord [low→high] → [1, 3, 5]           │
│        – LH chord [low→high] → [5, 3, 1]           │
│    • LH-aware crossing & direction flip             │
│    • Intent modifiers (LEGATO subst / BRILLIANT amp)│
│    • Climax note → strong fingers (2, 3)            │
│                                                     │
│  Layer D: CrossPhraseStitch                         │
│    • Junction constraint (phòng clashing)           │
│    • Preferred end finger for smooth transition     │
└───────────────────┬─────────────────────────────────┘
                    │ List[int] (finger 1–5 per note)
                    ▼
              Fingering Output
```

---

## 📁 Cấu trúc project

```
AutomaticPianoFingering/
├── src/fingering/
│   ├── models/
│   │   └── note_event.py          # NoteEvent data model + keyboard geometry
│   ├── core/
│   │   └── keyboard.py            # Span limits, white-key index, LH/RH helpers
│   ├── io/
│   │   └── musicxml_reader.py     # MusicXML parser (grand staff aware)
│   └── phrasing/
│       ├── phrase.py              # Phrase dataclass, PhraseIntent, ArcType enums
│       ├── boundary_detector.py   # Layer A — 3 improvements + forced segmentation
│       ├── intent_analyzer.py     # Layer B — intent + tension curve
│       ├── phrase_dp.py           # Layer C — Viterbi DP (LH/RH aware)
│       ├── pattern_library.py     # Pattern detection: scale, arpeggio
│       ├── chord_heuristic.py     # Chord fingering: 1-3-5 / 5-3-1
│       ├── cross_phrase.py        # Layer D — cross-phrase stitch
│       └── pipeline.py            # Main orchestrator (PhraseAwareFingering)
├── tests/
│   └── phrasing/
│       └── test_phrase_aware.py   # 34 unit tests (7 test classes)
├── scripts/
│   ├── demo_musicxml.py           # Single-staff demo + GT comparison
│   └── demo_grand_staff.py        # Grand staff demo (RH + LH separately)
└── test_file/
    ├── cW8VLC9nnTo.musicxml       # Single-staff test (166 notes)
    └── FN7ALfpGxiI.musicxml       # Grand staff test (1273 notes, 199 measures)
```

---

## 🧪 Test Results

```bash
python -m pytest tests/ -q
# 34 passed in 0.08s ✅
```

### Match rate vs. ground truth annotations

Tested on `FN7ALfpGxiI.musicxml` (grand staff, 199 measures):

| Hand | Notes | GT Annotated | Match Rate |
|------|-------|:------------:|:----------:|
| Right Hand | 524 | 433 | **28.9%** |
| Left Hand  | 496 | 146 | **27.4%** |

> **Note:** GT annotations cover ~55% of total notes. Many mismatches reflect
> personal stylistic preferences (the same piece fingered by different pianists
> often differs 30–40% even among professionals).

---

## 🚀 Quick Start

### Install

```bash
pip install -e ".[dev]"
```

### Run on a MusicXML file

```python
from fingering.io.musicxml_reader import MusicXMLReader
from fingering.phrasing.pipeline import PhraseAwareFingering

# Grand staff (2 staves)
rh, lh = MusicXMLReader().parse_grand_staff("piece.musicxml")

paf = PhraseAwareFingering()
rh_fingering = paf.run(rh)   # [1, 2, 3, 1, 2, ...]
lh_fingering = paf.run(lh)   # [5, 3, 1, 5, 3, ...]

# Or annotate in-place
paf.run_and_annotate(rh)  # sets note.finger on each NoteEvent
```

### Demo scripts

```bash
# Single-staff demo
python scripts/demo_musicxml.py test_file/cW8VLC9nnTo.musicxml

# Grand staff demo (RH + LH breakdown)
python scripts/demo_grand_staff.py test_file/FN7ALfpGxiI.musicxml
```

---

## 📊 Hiện trạng & Roadmap

### ✅ Đã hoàn thành

| Module | Mô tả |
|--------|-------|
| `NoteEvent` | Data model với keyboard geometry (white-key index, black key) |
| `keyboard.py` | Span limits, LH/RH natural finger order, thumb crossing |
| `MusicXMLReader` | Parser grand staff (RH/LH auto-detect), onset tracking đúng |
| `PhraseBoundaryDetector` | Signal fusion + 3 improvements + forced segmentation |
| `PhraseIntentAnalyzer` | 5 intent types, tension curve, climax detection |
| `PhraseScopedDP` | Viterbi DP với LH-aware ergonomics |
| `PatternLibrary` | Scale/arpeggio detect → inject 1-2-3-1 fingering |
| `ChordHeuristic` | Simultaneous notes → 1-3-5 (RH) / 5-3-1 (LH) |
| `CrossPhraseStitch` | Junction constraint, smooth cross-phrase transition |
| `PhraseAwareFingering` | Pipeline orchestrator |
| Tests | 34 unit tests, 7 test classes |

### 🚧 Tiếp theo

| Priority | Việc cần làm |
|----------|-------------|
| HIGH | **Pattern Library mở rộng** — inject specific broken-chord patterns (GT đang khác vì player dùng 4-5-1-2 thay 1-2-3-4) |
| HIGH | **Hand Motion Segmentation (HMS)** — thay vì phrase-level, segment theo hand position shift |
| HIGH | **MusicXML parser: nốt nối (tie)** — hiện chưa skip tied notes đúng cách |
| MED  | **Data Layer** — PIG dataset loader (150 pieces, multi-annotator) |
| MED  | **Neural baseline** — BI-LSTM/CRF trained on PIG dataset |
| LOW  | **Score export** — ghi fingering annotation trở lại file MusicXML |

---

## 🔬 So sánh với pianist thật sự

| Cognitive step | Pianist | Hệ thống này |
|---------------|---------|--------------|
| Nhìn tổng quát | Scan toàn bản, nhận ra key/meter | ✅ MusicXML parse: key, time sig |
| Đọc theo phrase | Nhóm note theo arc nhạc | ✅ PhraseBoundaryDetector (Layer A) |
| Nhận ý nhạc | Legato? Brilliant? | ✅ PhraseIntentAnalyzer (Layer B) |
| Pattern recognition | "Đây là scale C major" → 1-2-3-1 | ✅ PatternLibrary (Fix 2) |
| Chord fingering | C-E-G LH → 5-3-1 reflexively | ✅ ChordHeuristic (Fix 3) |
| Cross-phrase | Ngón cuối phrase A → ngón đầu phrase B | ✅ CrossPhraseStitch (Layer D) |
| Climax shaping | Ngón mạnh tại đỉnh arc | ✅ PhraseScopedDP climax reward |
| LH vs RH | Ngón thuận ngược chiều | ✅ LH-aware ergonomics |
| Học từ data | Điều chỉnh theo phong cách | 🚧 Neural model (chưa) |
| High-level structure | Nhận ra sonata form | ❌ Ngoài scope hiện tại |

---

## 📦 Dependencies

```toml
# Core
numpy
# Optional: neural model
torch
torchcrf
# Optional: music21 for advanced parsing
music21
```

---

*Built with the philosophy: fingering is musical thinking, not just ergonomic optimization.*
