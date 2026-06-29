#!/usr/bin/env python3
from pathlib import Path
from datetime import datetime

ROOT = Path("results/bus_real_data/99_FINAL_RESULTS_FOR_REPORT")

def read(path):
    p = Path(path)
    if not p.exists():
        return f"[MISSING] {p}\n"
    return p.read_text(errors="replace")

def section(title, body):
    line = "=" * 100
    return f"\n{line}\n{title}\n{line}\n{body.rstrip()}\n"

def main():
    ROOT.mkdir(parents=True, exist_ok=True)

    parts = []
    parts.append("FINAL REPORT — CAMERA RIG CALIBRATION\n")
    parts.append(f"generated_at: {datetime.now().isoformat(timespec='seconds')}\n")

    parts.append("""
Evaluation hierarchy:

PRIMARY:
- Pairwise static camera-to-camera extrinsic errors for AP01, AP02, AP03.
- No GT map alignment is used in the primary metric.
- This is the main report metric.

SECONDARY:
- Ref14/world-frame static camera-map vs GT after evaluation-only SE(3) alignment.
- This evaluates global camera-map shape/consistency.
- GT is used only after method estimation.

DIAGNOSTICS:
- AP02 graph-BA summary and optional GT-aligned full marker-map evaluation.
- AP03 COLMAP reconstruction coverage.
- AP03 marker-size-only scale stability.

Input rule:
- GT/SDF is used only for evaluation.
- AP01/AP02/AP03 method inputs are images, intrinsics, detections/feature tracks, marker IDs/size as applicable.
- AP03 does not use a known SDF marker map as method input.
""".strip() + "\n")

    parts.append(section(
        "PRIMARY — PAIRWISE CAMERA-TO-CAMERA EXTRINSIC ERRORS",
        read(ROOT / "BASELINE_FINAL_CLEAN_COMPARISON.txt")
    ))

    parts.append(section(
        "SECONDARY — REF14/WORLD CAMERA-MAP VS GT, EVALUATION-ONLY SE(3) ALIGNMENT",
        read(ROOT / "SECONDARY_REF14_WORLD_CAMERA_MAP_EVALUATION.txt")
    ))

    parts.append(section(
        "DIAGNOSTIC — AP02 WITH-MOVING BA SUMMARY",
        read("results/bus_real_data/02_ref_marker_graph_ba/07_graph_ba/with_moving/ba_summary.txt")
    ))

    ap02_full_txt = Path("results/bus_real_data/02_ref_marker_graph_ba/08_final_results/AP02_FINAL_GT_ALIGNED_FULL_MAP_EVALUATION.txt")
    if ap02_full_txt.exists():
        parts.append(section(
            "DIAGNOSTIC — AP02 GT-ALIGNED FULL MARKER-MAP EVALUATION",
            read(ap02_full_txt)
        ))

    parts.append(section(
        "DIAGNOSTIC — AP03 COLMAP RECONSTRUCTION",
        read(ROOT / "DIAGNOSTIC_AP03_COLMAP_RECONSTRUCTION.txt")
    ))

    parts.append(section(
        "DIAGNOSTIC — AP03 MARKER-SIZE-ONLY SCALE",
        read(ROOT / "DIAGNOSTIC_AP03_MARKER_SIZE_SCALE.txt")
    ))

    files = [
        "BASELINE_FINAL_CLEAN_COMPARISON.txt",
        "PRIMARY_PAIRWISE_SUMMARY.csv",
        "PRIMARY_PAIRWISE_DETAIL.csv",
        "SECONDARY_REF14_WORLD_CAMERA_MAP_SUMMARY.csv",
        "SECONDARY_REF14_WORLD_CAMERA_MAP_DETAIL.csv",
        "SECONDARY_REF14_WORLD_CAMERA_MAP_EVALUATION.txt",
        "DIAGNOSTIC_AP03_COLMAP_RECONSTRUCTION.txt",
        "DIAGNOSTIC_AP03_MARKER_SIZE_SCALE.txt",
        "DIAGNOSTIC_AP03_MARKER_SIZE_SCALE.json",
        "FINAL_REPORT.txt",
        "README.txt",
        "MANIFEST.json",
    ]

    parts.append(section(
        "FILE INDEX",
        "\n".join(f"- {f}" for f in files)
    ))

    (ROOT / "FINAL_REPORT.txt").write_text("\n".join(parts) + "\n")

    (ROOT / "README.txt").write_text("""FINAL_RESULTS README
====================

Start here:
1. BASELINE_FINAL_CLEAN_COMPARISON.txt
   Primary pairwise static camera-to-camera extrinsic comparison.

2. SECONDARY_REF14_WORLD_CAMERA_MAP_EVALUATION.txt
   Secondary camera-map shape evaluation after evaluation-only SE(3) alignment.

3. FINAL_REPORT.txt
   Combined readable report.

Primary metric is pairwise camera-to-camera extrinsics. Secondary/diagnostics are not the main ranking metric.
""")

    print("[OK] refreshed", ROOT / "FINAL_REPORT.txt")
    print("[OK] refreshed", ROOT / "README.txt")

if __name__ == "__main__":
    main()
