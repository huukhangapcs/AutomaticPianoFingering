"""
Standalone audit script: run the FingeredAuditor on a MusicXML file
and print a violation report per phrase.

Usage:
    python3 scripts/audit_fingering.py test_file/FN7ALfpGxiI.musicxml
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from fingering.io.musicxml_reader import MusicXMLReader
from fingering.phrasing.pipeline import PhraseAwareFingering
from fingering.phrasing.fingering_auditor import FingeredAuditor, Severity

def main(path: str):
    print(f"\n{'='*60}")
    print(f"FINGERING AUDIT: {os.path.basename(path)}")
    print(f"{'='*60}")

    reader = MusicXMLReader()
    rh_notes, lh_notes, key_pc, _mode = reader.parse_grand_staff_with_key(path)

    paf = PhraseAwareFingering(tonic_pc=key_pc)
    auditor = FingeredAuditor()

    # --- RIGHT HAND ---
    print("\n[ RIGHT HAND ]\n")
    rh_phrases = paf.get_phrases(rh_notes, companion_notes=lh_notes)
    all_rh_hard = 0
    all_rh_warn = 0

    for p in rh_phrases:
        if not p.notes:
            continue
        # Re-run DP without auditor to get raw output for comparison
        from fingering.phrasing.phrase_dp import PhraseScopedDP
        from fingering.phrasing.pattern_library import apply_pattern_constraints
        from fingering.phrasing.chord_heuristic import build_forced_constraints

        forced = {}
        forced = apply_pattern_constraints(p.notes, 'right', forced)
        forced = build_forced_constraints(p.notes, 'right', forced)

        dp_solver = PhraseScopedDP()
        raw_fingering = dp_solver.solve(p)

        report = auditor.audit(p.notes, raw_fingering, hand='right')
        repaired = auditor.repair(p.notes, raw_fingering, hand='right')
        report_after = auditor.audit(p.notes, repaired, hand='right')

        all_rh_hard += len(report.hard_violations)
        all_rh_warn += len(report.warnings)

        if report.issues:
            print(f"  Phrase {p.id:3d}  m{p.notes[0].measure}→m{p.notes[-1].measure}"
                  f"  [{len(p.notes):3d} notes]")
            for issue in report.issues:
                icon = "✖" if issue.severity == Severity.HARD_VIOLATION else "⚠"
                print(f"    {icon} {issue}")
            if not report_after.is_clean:
                print(f"    → Repair reduced to: {len(report_after.hard_violations)} HARD, "
                      f"{len(report_after.warnings)} WARN")
            print()

    print(f"\n  TOTAL RH: {all_rh_hard} HARD violations, {all_rh_warn} warnings")

    # --- LEFT HAND ---
    print("\n[ LEFT HAND ]\n")
    lh_phrases = paf.get_phrases(lh_notes, companion_notes=rh_notes)
    all_lh_hard = 0
    all_lh_warn = 0

    for p in lh_phrases:
        if not p.notes:
            continue
        raw_fingering = PhraseScopedDP().solve(p)
        report = auditor.audit(p.notes, raw_fingering, hand='left')
        all_lh_hard += len(report.hard_violations)
        all_lh_warn += len(report.warnings)
        if report.hard_violations:
            print(f"  Phrase {p.id:3d}  m{p.notes[0].measure}→m{p.notes[-1].measure}")
            for issue in report.hard_violations:
                print(f"    ✖ {issue}")
            print()

    print(f"  TOTAL LH: {all_lh_hard} HARD violations, {all_lh_warn} warnings")
    print(f"\n{'='*60}\n")


if __name__ == '__main__':
    path = sys.argv[1] if len(sys.argv) > 1 else 'test_file/FN7ALfpGxiI.musicxml'
    main(path)
