#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import csv
import json
import shutil
import sys
from datetime import datetime

sys.path.insert(0, str(Path("run/bus_real_data/ablation/_shared").resolve()))
from final_eval_policy import (
    CAM_MAP, CAM_ORDER, METHOD_POLICY,
    parse_clean_final_table, row_to_camera_rows,
    write_csv, read_csv, sha16, fmt, mean_float, validate_four_camera_csv,
)

RES_FINAL = Path("results/bus_real_data/ablation/moving_cam/res/final_results/MOVING_CAM_RES_CLEAN_FINAL_COMPARISON.txt")
OUT = Path("results/bus_real_data/99_FINAL_RESULTS_FOR_REPORT")
BASELINE_RESOLUTION = "1280x720_baseline"

AP03_SINGLE_SRC = Path("results/bus_real_data/ablation/moving_cam/res/04_ap03_results/res_1280x720_baseline/03_targetless_colmap_aruco_scale/07_final_results/AP03_FINAL_SINGLE_REF14_RESULT.csv")
AP03_MULTI_SRC = Path("results/bus_real_data/ablation/moving_cam/res/04_ap03_results/res_1280x720_baseline/03_targetless_colmap_aruco_scale/07_final_results/AP03_FINAL_MULTI_ARUCO_RESULT.csv")

def fail(msg: str) -> None:
    print(f"[FAIL] {msg}")
    raise SystemExit(1)

def require_row(rows: list[dict], method: str) -> dict:
    matches = [
        r for r in rows
        if r.get("resolution") == BASELINE_RESOLUTION and r.get("method") == method
    ]
    if not matches:
        fail(f"missing source-gated row for {BASELINE_RESOLUTION} / {method}")
    row = matches[0]
    if row.get("status") != "OK":
        fail(f"{method} baseline status is not OK: {row.get('status')}")
    return row

def write_method_report(method_dir: Path, method: str, row: dict, extra_note: str = "") -> None:
    camera_rows = row_to_camera_rows(row, method)
    fields = [
        "method", "status", "entity_type", "entity_id",
        "translation_error_cm", "rotation_error_deg",
        "source_policy", "note",
    ]

    canonical = METHOD_POLICY[method]["canonical"]
    csv_path = method_dir / f"{canonical}_FINAL_RESULT.csv"
    txt_path = method_dir / f"{canonical}_FINAL_RESULT.txt"

    write_csv(csv_path, camera_rows, fields)

    text = []
    text.append(f"{canonical} FINAL RESULT — SOURCE-GATED BASELINE")
    text.append("=" * len(text[0]))
    text.append("")
    text.append(f"Baseline input: {BASELINE_RESOLUTION}")
    text.append(f"Status: {row['status']}")
    text.append(f"Mean translation error [cm]: {row['mean_t_cm']}")
    text.append(f"Mean rotation error [deg]: {row['mean_r_deg']}")
    text.append("")
    text.append("Source policy:")
    text.append(METHOD_POLICY[method]["source_policy"])
    text.append("")
    if extra_note:
        text.append("Note:")
        text.append(extra_note)
        text.append("")
    text.append("Per-camera:")
    for r in camera_rows:
        text.append(f"- {r['entity_id']}: {r['translation_error_cm']} cm / {r['rotation_error_deg']} deg")
    text.append("")
    text.append(f"Source clean comparison TXT: {RES_FINAL}")
    text.append(f"CSV: {csv_path}")
    txt_path.write_text("\n".join(text) + "\n", encoding="utf-8")

def copy_ap03_source(src: Path, dst: Path, method: str) -> None:
    if not src.exists():
        fail(f"missing AP03 source: {src}")

    ok, cams = validate_four_camera_csv(src)
    if not ok:
        fail(f"{method} source does not contain all four cameras: {cams}")

    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)

def write_ap03_split_summary(ap03_dir: Path, single_row: dict, multi_row: dict) -> None:
    single_dst = ap03_dir / "AP03_FINAL_SINGLE_REF14_RESULT.csv"
    multi_dst = ap03_dir / "AP03_FINAL_MULTI_ARUCO_RESULT.csv"

    copy_ap03_source(AP03_SINGLE_SRC, single_dst, "AP03-SINGLE-REF14")
    copy_ap03_source(AP03_MULTI_SRC, multi_dst, "AP03-MULTI-ARUCO")

    rows = []
    for method, row in [
        ("AP03-SINGLE-REF14", single_row),
        ("AP03-MULTI-ARUCO", multi_row),
    ]:
        rows.append({
            "method": method,
            "status": row["status"],
            "mean_t_cm": row["mean_t_cm"],
            "mean_r_deg": row["mean_r_deg"],
            "cam0_t": row["cam0_t"],
            "cam0_r": row["cam0_r"],
            "cam1_t": row["cam1_t"],
            "cam1_r": row["cam1_r"],
            "cam3_t": row["cam3_t"],
            "cam3_r": row["cam3_r"],
            "cam5_t": row["cam5_t"],
            "cam5_r": row["cam5_r"],
            "source_policy": METHOD_POLICY[method]["source_policy"],
            "source_csv_sha16": sha16(single_dst if method == "AP03-SINGLE-REF14" else multi_dst),
            "note": row.get("note", ""),
        })

    fields = [
        "method", "status", "mean_t_cm", "mean_r_deg",
        "cam0_t", "cam0_r", "cam1_t", "cam1_r",
        "cam3_t", "cam3_r", "cam5_t", "cam5_r",
        "source_policy", "source_csv_sha16", "note",
    ]
    write_csv(ap03_dir / "AP03_FINAL_RESULT.csv", rows, fields)

    txt = []
    txt.append("AP03 FINAL RESULT — SOURCE-GATED SINGLE/MULTI SPLIT")
    txt.append("===================================================")
    txt.append("")
    txt.append(f"Baseline input: {BASELINE_RESOLUTION}")
    txt.append("")
    txt.append("Method policy:")
    txt.append("- AP03-SINGLE-REF14: targetless COLMAP + single Ref14 metric registration; GT evaluation-only.")
    txt.append("- AP03-MULTI-ARUCO: targetless COLMAP + Multi-ArUco metric registration; GT evaluation-only.")
    txt.append("")
    txt.append("Results:")
    for r in rows:
        txt.append(f"- {r['method']}: {r['mean_t_cm']} cm / {r['mean_r_deg']} deg, status={r['status']}")
    txt.append("")
    txt.append("Per-camera source CSV files:")
    txt.append(f"- {single_dst}")
    txt.append(f"- {multi_dst}")
    txt.append("")
    txt.append("GT camera poses are not used for AP03 fitting; GT is used only for final error evaluation.")
    (ap03_dir / "AP03_FINAL_RESULT.txt").write_text("\n".join(txt) + "\n", encoding="utf-8")

def write_baseline_clean_comparison(rows_by_method: dict[str, dict]) -> None:
    rows = []
    for method in ["AP01", "AP02", "AP03-SINGLE-REF14", "AP03-MULTI-ARUCO"]:
        r = rows_by_method[method]
        rows.append({
            "resolution": "baseline_1280x720",
            "method": method,
            "status": r["status"],
            "mean_t_cm": r["mean_t_cm"],
            "mean_r_deg": r["mean_r_deg"],
            "cam0_t": r["cam0_t"],
            "cam0_r": r["cam0_r"],
            "cam1_t": r["cam1_t"],
            "cam1_r": r["cam1_r"],
            "cam3_t": r["cam3_t"],
            "cam3_r": r["cam3_r"],
            "cam5_t": r["cam5_t"],
            "cam5_r": r["cam5_r"],
            "note": r.get("note", ""),
        })

    spec = {
        "title": "BASELINE FINAL CLEAN COMPARISON — SOURCE-GATED",
        "scope": "Camera-to-GT comparison for the validated 1280x720 baseline input. Translation error in cm. Rotation error in degrees.",
        "source_policy": [
            METHOD_POLICY["AP01"]["source_policy"],
            METHOD_POLICY["AP02"]["source_policy"],
            METHOD_POLICY["AP03-SINGLE-REF14"]["source_policy"],
            METHOD_POLICY["AP03-MULTI-ARUCO"]["source_policy"],
            "This baseline report is generated from the validated moving_cam/res 1280x720 source-gated final layer, not from legacy global pipeline reports.",
        ],
        "rows": rows,
        "interpretation": [
            "AP01 is a relative camera-map result after evaluation-only SE(3) alignment; it is not an absolute world-pose estimate.",
            "AP02 uses official/full-map SE(3) evaluation with marker14 held out from alignment.",
            "AP03 is split into Single-Ref14 and Multi-ArUco metric registration.",
            "AP03-MULTI-ARUCO is the robust AP03 baseline result.",
        ],
        "workflow": [
            "Run or snapshot method outputs.",
            "Apply source gating and final evaluator policy.",
            "Write 99_FINAL_RESULTS_FOR_REPORT only from accepted final rows.",
            "Do not treat legacy intermediate reports as final metrics.",
        ],
    }

    spec_path = OUT / "_BASELINE_FINAL_CLEAN_COMPARISON_SPEC.json"
    spec_path.write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")

    # Use existing shared writer.
    import subprocess
    subprocess.check_call([
        "python3",
        "run/bus_real_data/ablation/_shared/write_clean_ablation_comparison.py",
        "--spec", str(spec_path),
        "--out", str(OUT / "BASELINE_FINAL_CLEAN_COMPARISON.txt"),
    ])

def write_manifest(rows_by_method: dict[str, dict], backup_dir: Path | None) -> None:
    manifest = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "baseline_input": BASELINE_RESOLUTION,
        "source_clean_comparison_txt": str(RES_FINAL),
        "previous_99_final_backup": str(backup_dir) if backup_dir else None,
        "method_policy": METHOD_POLICY,
        "source_rows": rows_by_method,
        "source_files": {
            "ap03_single_src": str(AP03_SINGLE_SRC),
            "ap03_single_sha16": sha16(AP03_SINGLE_SRC),
            "ap03_multi_src": str(AP03_MULTI_SRC),
            "ap03_multi_sha16": sha16(AP03_MULTI_SRC),
        },
    }
    (OUT / "_SOURCE_GATING_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

def main() -> None:
    if not RES_FINAL.exists():
        fail(f"missing RES source-gated final TXT: {RES_FINAL}")

    rows = parse_clean_final_table(RES_FINAL)
    rows_by_method = {
        "AP01": require_row(rows, "AP01"),
        "AP02": require_row(rows, "AP02"),
        "AP03-SINGLE-REF14": require_row(rows, "AP03-SINGLE-REF14"),
        "AP03-MULTI-ARUCO": require_row(rows, "AP03-MULTI-ARUCO"),
    }

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = None
    if OUT.exists():
        backup_dir = Path(f"results/bus_real_data/_backup_99_FINAL_RESULTS_FOR_REPORT_before_source_gated_{stamp}")
        shutil.copytree(OUT, backup_dir)
        print(f"[BACKUP] {OUT} -> {backup_dir}")

    OUT.mkdir(parents=True, exist_ok=True)

    # Clean report folders only, keep backup outside.
    for name in ["AP01", "AP02", "AP03"]:
        d = OUT / name
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)

    write_method_report(
        OUT / "AP01",
        "AP01",
        rows_by_method["AP01"],
        "This replaces legacy AP01 Ref14/cam3-anchor global output with the accepted relative camera-map SE(3) evaluation.",
    )

    write_method_report(
        OUT / "AP02",
        "AP02",
        rows_by_method["AP02"],
        "This uses the accepted official/full-map SE(3) baseline row from the RES source-gated final layer.",
    )

    write_ap03_split_summary(
        OUT / "AP03",
        rows_by_method["AP03-SINGLE-REF14"],
        rows_by_method["AP03-MULTI-ARUCO"],
    )

    write_baseline_clean_comparison(rows_by_method)
    write_manifest(rows_by_method, backup_dir)

    readme = []
    readme.append("99_FINAL_RESULTS_FOR_REPORT — SOURCE-GATED")
    readme.append("==========================================")
    readme.append("")
    readme.append("This folder is generated by:")
    readme.append("run/bus_real_data/baseline_sanity/write_99_final_results_for_report.py")
    readme.append("")
    readme.append("Accepted baseline input:")
    readme.append(BASELINE_RESOLUTION)
    readme.append("")
    readme.append("Do not use legacy method-internal output files as final metrics.")
    readme.append("Use BASELINE_FINAL_CLEAN_COMPARISON.txt for the baseline method comparison.")
    readme.append("")
    readme.append("AP01:")
    readme.append(METHOD_POLICY["AP01"]["source_policy"])
    readme.append("")
    readme.append("AP02:")
    readme.append(METHOD_POLICY["AP02"]["source_policy"])
    readme.append("")
    readme.append("AP03:")
    readme.append("- split into AP03-SINGLE-REF14 and AP03-MULTI-ARUCO")
    readme.append("- COLMAP targetless; ArUco post-reconstruction only; GT evaluation-only")
    (OUT / "README.txt").write_text("\n".join(readme) + "\n", encoding="utf-8")

    print("[OK] wrote source-gated 99_FINAL_RESULTS_FOR_REPORT")
    print(f"[OK] main report: {OUT / 'BASELINE_FINAL_CLEAN_COMPARISON.txt'}")
    print(f"[OK] manifest:    {OUT / '_SOURCE_GATING_MANIFEST.json'}")

if __name__ == "__main__":
    main()
