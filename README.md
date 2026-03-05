# 🎹 Automatic Piano Fingering

> **Tư duy như pianist thật sự** — đọc nhạc theo phrase, track tay theo mm, không tối ưu từng nốt đơn lẻ.

---

## 🧠 Triết lý thiết kế

Pianist thật sự **không** đọc từng nốt rồi tìm ngón. Họ:
1. **Nhìn cấu trúc tổng quát** — phrase, pattern, arc
2. **Xác định ý nhạc** — legato? brilliant? expressive?
3. **Giữ bàn tay cố định** (Lazy First Principle) — chỉ di chuyển khi bắt buộc
4. **Áp fingering theo nhóm** — scale → 1-2-3-1, chord → 1-3-5

Hệ thống mô phỏng đúng quy trình đó.

---

## 🏗️ Kiến trúc (Phase 2.9)

```
MusicXML
    │
    ▼
IO Layer — MusicXMLReader
  • Grand staff (staff1=RH, staff2=LH)
  • Slur, staccato, accent, fingering GT
  • Key signature → tonic_pc
    │
    ▼
RECOGNITION LAYERS
  Layer 0  ScoreProfile     — texture, key, markings
  Layer 1  VoiceSeparation  — melody stream (top note per onset)
  Layer 2  PhraseBoundary   — 9 signals, logit/sigmoid scoring
  Layer 3  GlobalSegment    — Viterbi DP: max Σ P(boundary)×Prior
    │
    ▼
FINGERING LAYERS
  Layer A  PatternLibrary   — 12-tone scale, Hanon, arpeggio, Alberti bass
  Layer B  IntentAnalyzer   — CANTABILE/BRILLIANT/LEGATO/STACCATO/EXPRESSIVE
  Layer C  PositionPlanner  — [Phase 2.8] sliding-window anchor planning
  Layer D  PhraseScopedDP   — Viterbi + biomechanics + HandState
  Layer E  CrossPhrase      — junction constraint, 3-note lookahead
  Layer F  FingeredAuditor  — 9-rule validation + auto-repair
    │
    ▼
List[int]  (finger 1–5 per note)
```

---

## 📁 Cấu trúc project

```
src/fingering/
├── models/
│   └── note_event.py           # NoteEvent dataclass
├── core/
│   └── keyboard.py             # Physical key positions (mm), anthropometric spans
├── io/
│   └── musicxml_reader.py      # MusicXML parser — grand staff + key sig
└── phrasing/
    ├── phrase.py               # Phrase, PhraseIntent, ArcType, HandResetType
    ├── boundary_detector.py    # Boundary signals + Viterbi segmentation
    ├── voice_separation.py     # Melody stream extraction
    ├── scale_fingering.py      # 12-tone scale database (all major/minor)
    ├── pattern_library.py      # Scale, Hanon, arpeggio, Alberti/Waltz bass
    ├── intent_analyzer.py      # Intent + tension curve + climax
    ├── position_planner.py     # [NEW] Sliding-window hand position planner
    ├── hand_position.py        # HandState (thumb_mm), HandPositionTracker
    ├── hand_reset.py           # [Phase 2.9] HandResetType classifier — rest/held note
    ├── phrase_dp.py            # Viterbi DP — biomechanics + HandState + anchor
    ├── chord_heuristic.py      # Chord fingering: 1-3-5 / 5-3-1
    ├── thumb_placement_planner.py  # Thumb injection for scale runs
    ├── cross_phrase.py         # CrossPhraseStitch (NONE/SOFT/FULL dispatch)
    ├── fingering_auditor.py    # Post-DP auditor + auto-repair
    ├── score_profile.py        # Global texture analysis
    └── pipeline.py             # PhraseAwareFingering orchestrator

tests/
└── phrasing/
    └── test_phrase_aware.py    # 51 unit tests (8 test classes)

scripts/
├── demo_grand_staff.py         # Grand staff demo + GT comparison
├── pig_eval.py                 # PIG dataset benchmark
├── audit_fingering.py          # Violation audit
├── error_analysis.py           # Error categorization
└── pig_eval_simple.py          # Benchmark for Physics-First engine
```

**New: Physics-First package**
```
src/fingering/simple/
├── __init__.py
├── fingering_dp.py             # SimpleFingering: Viterbi + 5-cost model (~250 lines)
└── pipeline.py                 # PianoFingering: thin wrapper
```

---

## 🧪 Tests & Benchmark

```bash
python -m pytest tests/ -q
# 55 passed ✅
```

### Match rate — PIG Piano Fingering Dataset v1.2

Benchmark: 20 files, Right Hand only, 4,550 notes.

| Version | Overall | Avg/File | Code |
|---|:---:|:---:|:---:|
| Greedy baseline | 28.9% | — | — |
| + Viterbi DP | 30.59% | — | — |
| + Biomechanics + HandState + Planner | 47.36% | 47.54% | ~2,000 lines |
| 🔹 **Physics-First (simple/)** | **48.13%** | **48.55%** | **~300 lines** |

> **Key insight:** Bồ toàn bộ phrase segmentation, intent analysis, pattern library — chỉ giữ vật lý đàn + lazy first + look-ahead — cho kết quả **tốt hơn** và code it hơn 7 lần.

> Inter-annotator agreement giữa pianist chuyên nghiệp ~60–70%. Rule-based ceiling ~50%.

### 🔬 Error Analysis — Physics-First Engine (2,360 wrong / 4,550 total)

| Error type | Count | % errors | Note |
|---|:---:|:---:|---|
| **OFF_BY_ONE** | **1,835** | **77.8%** | Biết đúng hand position, sai 1 ngón |
| OTHER | 261 | 11.1% | Sai xa, jump chức |
| THUMB_PRED | 220 | 9.3% | Dự đoán thumb quá nhiều |
| THUMB_MISS | 44 | 1.9% | Bỏ sót thumb |

**Top confusions (GT → Pred):**
```
GT=2 → Pred=1  (-1): 412x    GT=4 → Pred=3  (-1): 233x
GT=3 → Pred=4  (+1): 231x    GT=2 → Pred=3  (+1): 230x
GT=4 → Pred=5  (+1): 229x    GT=1 → Pred=2  (+1): 223x
```

**Bias:** Gần đối xứng — predict thấp (54.2%) vs cao (45.8%). OFF_BY_ONE chiếm 77.8% lỗi.

**Root cause:** `thumb_mm` của ngón 1–5 được xấp xỉ bằng `(f-1)×23.5mm` nhưng thực tế các ngón không đều nhau — khi người chơi giữ bàn tay kiểu thực, ngón 3 (giữa) dài hơn ngón 1 (cái) nhiều hơn 23.5mm. Cần per-finger anatomical offset thay vì uniform spacing.

---

## 🚀 Quick Start

```bash
pip install -e ".[dev]"
```

```python
# Physics-First Engine (simple)
from fingering.io.musicxml_reader import MusicXMLReader
from fingering.simple import PianoFingering

rh, lh = MusicXMLReader().parse_grand_staff("piece.musicxml")
pf = PianoFingering(bpm=120.0)
rh_fingers = pf.run(rh)   # [1, 2, 3, 1, 2, ...]  
lh_fingers = pf.run(lh)
```

```python
# Legacy Complex Engine (phrasing/)
from fingering.phrasing.pipeline import PhraseAwareFingering

paf = PhraseAwareFingering()
rh_fingers = paf.run(rh, companion_notes=lh)
```

```bash
# Demo với GT comparison
python scripts/demo_grand_staff.py test_file/FN7ALfpGxiI.musicxml

# Benchmark — Physics-First
python scripts/pig_eval_simple.py

# Benchmark — Complex engine
python scripts/pig_eval.py

# Audit violations
python scripts/audit_fingering.py test_file/FN7ALfpGxiI.musicxml
```

---

## 📊 Roadmap

### ✅ Đã hoàn thành — Phase 2.5–2.8

| Phase | Tính năng | Impact |
|---|---|---|
| 2.5 | Viterbi DP + Biomechanics + PatternLibrary + Auditor | 28.9% → 35.54% |
| 2.6 | Physical Keyboard Model (mm) + Lazy First Principle | → 38.00% |
| 2.7 | HandState — thumb_mm tracking, is_in_position() | → 44.00% (+6pp) |
| 2.8 | Position Planner — sliding-window anchor pre-pass | → 45.76% (+1.76pp) |
| 2.9 | **HandResetType** — tách Physical Reset khỏi Musical Phrase; CrossPhraseStitch 3-way dispatch (NONE/SOFT/FULL) | → **47.36%** (+1.6pp) |

**Phase 2.9 core insight:** `CrossPhraseStitch` trước đây chỉ có binary `starts_after_rest`. Bây giờ phân loại chi tiết dựa trên **rest gap tính bằng giây** (tempo-aware) và **held note duration** — tay có thể reset khi có rest >= 0.4s hoặc nốt dài >= 2 beats (không slur).

### 🚧 Phase 3 — Neural

Rule-based ceiling ~50%. Cần neural model để vượt.

| Priority | Việc cần làm | Mục tiêu |
|---|---|---|
| HIGH | **THUMB_MISS fix** — giảm thiên vị ngón cái | 47.8% → <30% |
| HIGH | **Tied notes** — skip đúng cách trong parser | Accuracy ↑ |
| HIGH | **Bi-LSTM/CRF** trained trên PIG 309 files | MR > 55% |
| HIGH | **Harmonic Rhythm Fallback** — ngăn mega-phrase | Segmentation ↑ |
| MED  | **LH hand-specific weights** — Alberti/Waltz patterns | LH accuracy ↑ |
| MED  | **MusicXML export** — ghi fingering annotation ngược lại | Usability |
| LOW  | **Trill/tremolo** — alternating finger patterns | Edge cases |

---

## 🔬 So sánh với pianist

| Cognitive step | Pianist | Hệ thống |
|---|---|---|
| Đọc cấu trúc | Key, meter, phrase | ✅ MusicXMLReader + ScoreProfile |
| Tối ưu toàn bản | Global path | ✅ Viterbi DP |
| Voice separation | Melody vs. accompaniment | ✅ VoiceSeparation |
| Ý nhạc | Legato? Brilliant? | ✅ PhraseIntentAnalyzer |
| Pattern recognition | C major → 1-2-3-1 | ✅ 12-tone PatternLibrary |
| Thumb-under / Finger-over | Scale crossings | ✅ Reward model |
| Sinh học ngón tay | Tendon coupling, tempo | ✅ Biomechanical costs |
| Giữ tầm tay | Lazy First Principle | ✅ IN_POSITION_REWARD |
| Biết từng ngón ở đâu | 5 ngón = 5 tọa độ mm | ✅ HandState (thumb_mm) |
| Plan vị trí trước | Nhìn 4–8 nốt tới | ✅ Position Planner |
| Kiểm tra lỗi | "Ngón này không thể" | ✅ FingeredAuditor |
| Học từ data | Phong cách cá nhân | 🚧 Phase 3 — Neural |

---

## 📦 Dependencies

```toml
numpy          # DP tables
# Optional
torch          # Phase 3 neural model
torchcrf       # CRF layer
music21        # Advanced score analysis
```

---

*Built with the philosophy: fingering is musical thinking, not just ergonomic optimization.*
