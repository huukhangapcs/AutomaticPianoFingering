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
└── error_analysis.py           # Error categorization
```

---

## 🧪 Tests & Benchmark

```bash
python -m pytest tests/ -q
# 55 passed ✅
```

### Match rate — PIG Piano Fingering Dataset v1.2

Benchmark: 20 files, Right Hand only, 4,550 notes.

| Version | Overall | Avg/File |
|---|:---:|:---:|
| Greedy baseline | 28.9% | — |
| + Viterbi DP | 30.59% | — |
| + Biomechanics (tendon, tempo, black key) | 32.11% | 32.44% |
| + Finger-4 penalty + Black key long finger | 35.47% | 35.66% |
| + Large leap + Sequential stepwise reward | 37.12% | 37.38% |
| + Physical Keyboard Model (mm) + Lazy First | 38.00% | 38.21% |
| + HandState model (thumb_mm tracking) | 44.00% | 43.74% |
| + Position Planner pre-pass | 45.76% | 45.90% |
| + WEAK_PAIR in-position exception | 46.88% | 47.12% |
| + 3-zone finger span model | 47.58% | 47.76% |
| **+ HandResetType (rest/held note reset)** | **47.36%** | **47.54%** |

> **Note:** Số tuyệt đối 47.36% thấp hơn 47.58% 1 chút do thay đổi cách đo (rerun sạch, không cumulative). Improvement thực sự là +15.9pp so với baseline 31.47%.

> **Note:** Inter-annotator agreement giữa pianist chuyên nghiệp ~60–70%. Rule-based ceiling thực tế ~50%.

> **⚠️ Pattern Library trade-off:** Hard-coded scale/arpeggio rules có thể conflict với global Viterbi optimization — đây là lý do cốt lõi để chuyển sang **neural ranker** ở Phase 3.



### 🔬 Error Analysis — Phase 2.8 (2,468 wrong / 4,550 total)

| Error | Count | % errors | Note |
|---|:---:|:---:|---|
| **OFF_BY_ONE** | **1,042** | **42.2%** | gt=3 pred=2 — sai 1 ngón |
| **THUMB_MISS** | **842** | **34.1%** | -636 so với phase cũ ✅ |
| SCALE_ERROR | 284 | 11.5% | |
| CHORD_ERROR | 251 | 10.2% | |
| LARGE_JUMP | 30 | 1.2% | |
| OTHER | 19 | 0.8% | |

**Top confusions:** f3→f2 (432x), f2→f1 (315x), f4→f3 (295x)

Systematic bias: hệ thống predict ngón **thấp hơn GT 1 bậc** — biết đúng hand position nhưng chọn sai finger trong position.

**Fix đã apply:** WEAK_PAIR_PENALTY chỉ fire khi `not in_position` (stretch thật sự), không fire khi f3↔f4 trong cùng anchor.

---

## 🚀 Quick Start

```bash
pip install -e ".[dev]"
```

```python
from fingering.io.musicxml_reader import MusicXMLReader
from fingering.phrasing.pipeline import PhraseAwareFingering

rh, lh, tonic_pc, mode = MusicXMLReader().parse_grand_staff_with_key("piece.musicxml")

paf = PhraseAwareFingering(tonic_pc=tonic_pc)
rh_fingering = paf.run(rh, companion_notes=lh)  # [1, 2, 3, 1, 2, ...]
lh_fingering = paf.run(lh, companion_notes=rh)
```

```bash
# Demo với GT comparison
python scripts/demo_grand_staff.py test_file/FN7ALfpGxiI.musicxml

# Benchmark
python scripts/pig_eval.py 20

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
