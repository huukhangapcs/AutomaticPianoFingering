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

## 🏗️ Kiến trúc (Phase 2.5 — Phrase Segmentation v2 + Biomechanics)

```
MusicXML File
     │
     ▼
┌─────────────────────────────────────────────────────────────┐
│  IO Layer: MusicXMLReader                                   │
│  • Grand staff support (staff 1=RH, staff 2=LH)            │
│  • Per-staff time tracking với <backup> elements            │
│  • Slur, staccato, accent, fingering GT parsing             │
│  • Key signature → tonic_pc (12 keys)                       │
└───────────────────┬─────────────────────────────────────────┘
                    │ List[NoteEvent]
                    ▼
┌─────────────────────────────────────────────────────────────┐
│  RECOGNITION LAYERS (top-down + bottom-up)                  │
│                                                             │
│  Layer 0: ScoreProfile — texture, key, markings (global)   │
│                                                             │
│  Layer 1: Voice Separation [v2]                             │
│    • Trích xuất Melody Stream (nốt cao nhất mỗi onset)     │
│    • Tránh nhiễu hợp âm đệm khi phân tích phrase           │
│                                                             │
│  Layer 2: PhraseBoundaryDetector [v2 — Logit/Sigmoid]       │
│    • 8 signals: slur, rest, cadence, agogic, melodic arc   │
│    • Melodic Leap Compensation (≥ P4 + stepwise resolve)    │
│    • Boundary probability P ∈ (0,1) — không còn threshold  │
│                                                             │
│  Layer 3: Global Segmentation — Viterbi DP [v2]             │
│    • Tối ưu toàn cục: max Σ P(boundary) × Prior(length)    │
│    • Phrase length prior: Gaussian μ=4, σ=2 measures        │
│    • Constraint: 2 ≤ length ≤ 12 measures                  │
│    • LH: segment by measure, RH: full phrase pipeline       │
│                                                             │
│  Layer 3b: MotifEngine + PhraseSelector (top-down)          │
│  Layer 3c: HarmonicSkeleton + PeriodDetector               │
└───────────────────┬─────────────────────────────────────────┘
                    │ List[Phrase]
                    ▼
┌─────────────────────────────────────────────────────────────┐
│  FINGERING LAYERS                                           │
│                                                             │
│  Layer A: Pattern Library [v2 — tone-specific]              │
│    • 12-tone scale fingering database (all major/minor)     │
│        C major: 1-2-3-1-2-3-4-5                             │
│        Bb major: 4-1-2-3-1-2-3-4 (Bb = black key start)   │
│        F# major: 2-3-4-1-2-3-4-5 (F# = black key root)    │
│    • Hanon 5-finger patterns (5-9 nốt liền bậc < 1 octave)│
│    • Arpeggio broken chord: 1-2-4 / 5-3-1                  │
│    • Partial scale (4-7 nốt)                                │
│                                                             │
│  Layer B: PhraseIntentAnalyzer                              │
│    • CANTABILE / BRILLIANT / LEGATO / STACCATO / EXPRESSIVE │
│    • Tension curve + climax note detection                  │
│                                                             │
│  Layer C: PhraseScopedDP — Viterbi [v2 + Biomechanics]     │
│    • Ergonomic cost: span, crossing, black keys             │
│    • Thumb-Under reward: -1.5 khi đúng ngữ cảnh scale      │
│    • Finger-Over reward: -1.2 khi đúng ngữ cảnh descending │
│    • Biomechanics — Tendon Coupling (3-4, 4-5) at tempo    │
│    • Biomechanics — Black key depth span correction        │
│    • Biomechanics — Tempo-aware max_span (Presto → compact)│
│    • Intent modifiers (LEGATO subst / BRILLIANT amplify)   │
│    • Climax note → strong fingers (2, 3)                    │
│    • Chord heuristic: RH [1,3,5], LH [5,3,1]              │
│                                                             │
│  Layer D: CrossPhraseStitch                                 │
│    • Junction constraint, preferred end finger look-ahead  │
│                                                             │
│  Layer E: FingeredAuditor [NEW]                             │
│    • 9 rules (HARD_VIOLATION + WARNING)                     │
│    • Auto-repair HARD violations via local greedy search    │
└───────────────────┬─────────────────────────────────────────┘
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
│   │   └── note_event.py           # NoteEvent data model + keyboard geometry
│   ├── core/
│   │   └── keyboard.py             # Span limits, biomechanics helpers [v2]
│   ├── io/
│   │   └── musicxml_reader.py      # MusicXML parser (grand staff + key sig)
│   └── phrasing/
│       ├── phrase.py               # Phrase dataclass, PhraseIntent, ArcType
│       ├── boundary_detector.py    # Boundary signals + Viterbi segmentation [v2]
│       ├── voice_separation.py     # Melody stream extraction [v2 NEW]
│       ├── scale_fingering.py      # 12-tone scale fingering database [NEW]
│       ├── pattern_library.py      # Tone-specific scale + Hanon + arpeggio [v2]
│       ├── intent_analyzer.py      # Intent + tension curve
│       ├── phrase_dp.py            # Viterbi DP + biomechanics costs [v2]
│       ├── chord_heuristic.py      # Chord fingering: 1-3-5 / 5-3-1
│       ├── cross_phrase.py         # Cross-phrase stitch (Layer D)
│       ├── fingering_auditor.py    # Post-DP validation + auto-repair [NEW]
│       ├── harmonic_skeleton.py    # Cadence detection
│       ├── motif_engine.py         # A-B-A form detection
│       ├── period_detector.py      # Antecedent/consequent labeling
│       ├── phrase_selector.py      # Merge top-down + bottom-up signals
│       ├── score_profile.py        # Global texture + key profile
│       └── pipeline.py             # Main orchestrator (PhraseAwareFingering)
├── tests/
│   └── phrasing/
│       └── test_phrase_aware.py    # 35 unit tests (7 test classes)
├── scripts/
│   ├── demo_musicxml.py            # Single-staff demo + GT comparison
│   ├── demo_grand_staff.py         # Grand staff demo (RH + LH separately)
│   ├── pig_eval.py                 # PIG dataset benchmark (match rate)
│   ├── audit_fingering.py          # Violation audit per phrase [NEW]
│   └── error_analysis.py           # Error categorization by type [NEW]
└── test_file/
    ├── cW8VLC9nnTo.musicxml        # Single-staff test (166 notes)
    └── FN7ALfpGxiI.musicxml        # Grand staff test (Für Elise, 199 measures)
```

---

## 🧪 Test Results

```bash
python -m pytest tests/ -q
# 35 passed in 0.07s ✅
```

### Match rate vs. PIG dataset (Ground Truth annotations)

Benchmark trên 20 file đầu tiên của **PIG Piano Fingering Dataset v1.2** (Right Hand only):

| Version | Overall Match Rate | Average File Acc |
|---|:---:|:---:|
| Greedy threshold (baseline) | 28.9% | — |
| Viterbi DP (phase 2) | 30.59% | — |
| + Voice Separation | 30.53% | 30.75% |
| + Tone-specific Scale + Hanon | 31.91% | 32.16% |
| + Thumb-Under/Finger-Over rewards | 31.91% | 32.16% |
| **+ Biomechanics (tendon + tempo)** | **32.11%** | **32.44%** |

> **Note:** GT annotations reflect personal stylistic preferences. The same piece fingered by different professional pianists differs 30–40%, so ~32% match against a single annotator's style is realistic for a rule-based system.

### 🔬 Error Analysis (3,089 wrong predictions)

| Error Category | Count | % of Total Errors |
|---|:---:|:---:|
| **THUMB_MISS** | 1,478 | **47.8%** |
| **OFF_BY_ONE** | 901 | **29.2%** |
| SCALE_ERROR | 363 | 11.8% |
| CHORD_ERROR | 251 | 8.1% |
| LARGE_JUMP | 61 | 2.0% |
| BLACK_KEY | 17 | 0.6% |
| OTHER | 18 | 0.6% |

**Top confusions:** f3→f2 (417x), f2→f1 (362x), f1→f2 (310x) — hệ thống đang thiên vị ngón 1-2.

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
rh, lh, tonic_pc, mode = MusicXMLReader().parse_grand_staff_with_key("piece.musicxml")

paf = PhraseAwareFingering(tonic_pc=tonic_pc)
rh_fingering = paf.run(rh, companion_notes=lh)  # [1, 2, 3, 1, 2, ...]
lh_fingering = paf.run(lh, companion_notes=rh)  # [5, 3, 1, 5, 3, ...]

# Or annotate in-place
paf.run_and_annotate(rh, companion_notes=lh)  # sets note.finger on each NoteEvent
```

### Demo & Diagnostic scripts

```bash
# Grand staff demo (RH + LH breakdown, phrase analysis)
python scripts/demo_grand_staff.py test_file/FN7ALfpGxiI.musicxml

# PIG dataset benchmark
python scripts/pig_eval.py 20

# Fingering violation audit
python scripts/audit_fingering.py test_file/FN7ALfpGxiI.musicxml

# Error category analysis on PIG
python scripts/error_analysis.py 20
```

---

## 📊 Hiện trạng & Roadmap

### ✅ Phase 2.5 — Hoàn thành

| Module | Mô tả |
|--------|-------|
| `voice_separation.py` | Melody stream extraction từ polyphonic RH |
| `boundary_detector.py` | Logit/Sigmoid probability + Viterbi global DP |
| `scale_fingering.py` | 12-tone scale fingering database (all major + minor) |
| `pattern_library.py` | Tone-specific scale, Hanon 5-finger patterns |
| `phrase_dp.py` | Thumb-Under/Finger-Over rewards, biomechanics |
| `keyboard.py` | Tendon coupling, black key depth, tempo-aware span |
| `fingering_auditor.py` | 9-rule post-DP auditor + auto-repair |
| `error_analysis.py` | Error categorizer (9 types + confusion matrix) |
| Tests | 35 unit tests pass |

### 🚧 Phase 3 — Tiếp theo (Neural)

| Priority | Việc cần làm |
|----------|-------------|
| HIGH | **THUMB_MISS fix** — giảm thiên vị ngón cái trong DP cost model |
| HIGH | **MusicXML parser: tied notes** — skip tied notes đúng cách |
| HIGH | **Neural baseline** — BI-LSTM/CRF trained on full PIG dataset (309 files) |
| MED  | **Octave/interval fingering** — 1-5 octaves, 1-3/1-4 thirds |
| MED  | **Score export** — ghi fingering annotation ngược lại vào MusicXML |
| LOW  | **Trill/tremolo detection** — alternating finger patterns |

---

## 🔬 So sánh với pianist thật sự

| Cognitive step | Pianist | Hệ thống này |
|---------------|---------|--------------:|
| Nhìn tổng quát | Scan toàn bản, key/meter | ✅ MusicXML parse |
| Đọc theo phrase (global) | Tối ưu toàn bản | ✅ Viterbi DP segmentation |
| Voice separation | Phân biệt giai điệu / đệm | ✅ Melody stream extraction |
| Nhận ý nhạc | Legato? Brilliant? | ✅ PhraseIntentAnalyzer |
| Pattern recognition | "C major scale" → 1-2-3-1 | ✅ 12-tone PatternLibrary |
| Luồn ngón (thumb-under) | Ngón cái chui qua | ✅ Reward model |
| Vắt ngón (finger-over) | Ngón 3 vắt qua ngón cái | ✅ Reward model |
| Sinh học ngón tay | Gân 3-4 liên kết, tempo | ✅ Biomechanical costs |
| Chord fingering | C-E-G LH → 5-3-1 | ✅ ChordHeuristic |
| Cross-phrase planning | Ngón cuối A → đầu B | ✅ CrossPhraseStitch |
| Kiểm tra lỗi | "Ngón này không thể" | ✅ FingeredAuditor |
| Học từ data | Điều chỉnh phong cách | 🚧 Phase 3 — Neural Model |

---

## 📦 Dependencies

```toml
# Core
numpy

# Optional: neural model (Phase 3)
torch
torchcrf

# Optional: advanced parsing
music21
```

---

*Built with the philosophy: fingering is musical thinking, not just ergonomic optimization.*
