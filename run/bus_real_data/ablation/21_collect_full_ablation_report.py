#!/usr/bin/env python3
import csv
import sys
from pathlib import Path

if len(sys.argv) != 3:
    print("usage: python3 21_collect_full_ablation_report.py <ablation_root> <name>")
    sys.exit(1)

ROOT = Path(sys.argv[1])
NAME = sys.argv[2]
OUT = ROOT / "ABLATION_SUMMARY"
OUT.mkdir(parents=True, exist_ok=True)

PRIMARY_DETAIL_CANDIDATES = [
    "PRIMARY_PAIRWISE_DETAIL.csv",
    "BASELINE_FINAL_PAIRWISE_DETAIL.csv",
]
PRIMARY_SUMMARY_CANDIDATES = [
    "PRIMARY_PAIRWISE_SUMMARY.csv",
    "BASELINE_FINAL_PAIRWISE_SUMMARY.csv",
]
SECONDARY_SUMMARY_CANDIDATES = [
    "SECONDARY_REF14_WORLD_CAMERA_MAP_SUMMARY.csv",
]
SECONDARY_DETAIL_CANDIDATES = [
    "SECONDARY_REF14_WORLD_CAMERA_MAP_DETAIL.csv",
]

def read_csv_first(final: Path, names):
    for n in names:
        p = final / n
        if p.exists():
            with p.open(newline="", errors="replace") as f:
                return p, list(csv.DictReader(f))
    return None, []

def read_status(final: Path):
    p = final / "RUN_STATUS.txt"
    d = {}
    if p.exists():
        for line in p.read_text(errors="replace").splitlines():
            if "=" in line:
                k, v = line.split("=", 1)
                d[k.strip()] = v.strip()
    return d

def write_aggregate_csv(path, rows):
    if not rows:
        path.write_text("")
        return
    fields = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

def slim_primary_row(r):
    keys = [
        "method", "pair",
        "from_camera", "to_camera",
        "translation_error_cm", "rotation_error_deg",
        "baseline_error_cm", "direction_error_deg",
        "status",
    ]
    return " | ".join(f"{k}={r.get(k,'')}" for k in keys if k in r)

def slim_secondary_row(r):
    keys = [
        "method",
        "mean_translation_error_cm", "mean_rotation_error_deg",
        "median_translation_error_cm", "median_rotation_error_deg",
        "max_translation_error_cm", "max_rotation_error_deg",
        "status",
    ]
    existing = [k for k in keys if k in r]
    if existing:
        return " | ".join(f"{k}={r.get(k,'')}" for k in existing)
    return " | ".join(f"{k}={v}" for k, v in r.items())

def main():
    all_status = []
    all_primary_detail = []
    all_primary_summary = []
    all_secondary_summary = []
    all_secondary_detail = []

    report = []
    report.append(f"FULL ABLATION REPORT — {NAME}")
    report.append("=" * 100)
    report.append(f"root: {ROOT}")
    report.append("")

    variants = sorted(
        p for p in ROOT.iterdir()
        if p.is_dir() and p.name != "ABLATION_SUMMARY" and (p / "FINAL_RESULTS").exists()
    )

    for var in variants:
        final = var / "FINAL_RESULTS"
        status = read_status(final)

        status_row = {"variant": var.name, **status}
        all_status.append(status_row)

        p_sum_path, p_sum = read_csv_first(final, PRIMARY_SUMMARY_CANDIDATES)
        p_det_path, p_det = read_csv_first(final, PRIMARY_DETAIL_CANDIDATES)
        s_sum_path, s_sum = read_csv_first(final, SECONDARY_SUMMARY_CANDIDATES)
        s_det_path, s_det = read_csv_first(final, SECONDARY_DETAIL_CANDIDATES)

        for r in p_sum:
            all_primary_summary.append({"variant": var.name, **r})
        for r in p_det:
            all_primary_detail.append({"variant": var.name, **r})
        for r in s_sum:
            all_secondary_summary.append({"variant": var.name, **r})
        for r in s_det:
            all_secondary_detail.append({"variant": var.name, **r})

        report.append("")
        report.append("#" * 100)
        report.append(f"VARIANT: {var.name}")
        report.append("#" * 100)
        report.append("")

        report.append("RUN_STATUS")
        report.append("-" * 100)
        if status:
            for k, v in status.items():
                report.append(f"{k}={v}")
        else:
            report.append("[MISSING] RUN_STATUS.txt")
        report.append("")

        report.append("PRIMARY PAIRWISE SUMMARY")
        report.append("-" * 100)
        report.append(f"source: {p_sum_path.name if p_sum_path else 'MISSING'}")
        if p_sum:
            for r in p_sum:
                report.append(" | ".join(f"{k}={v}" for k, v in r.items()))
        else:
            report.append("[MISSING/EMPTY]")
        report.append("")

        report.append("PRIMARY CAM-TO-CAM DETAIL")
        report.append("-" * 100)
        report.append(f"source: {p_det_path.name if p_det_path else 'MISSING'}")
        if p_det:
            for r in p_det:
                report.append(slim_primary_row(r))
        else:
            report.append("[MISSING/EMPTY]")
        report.append("")

        report.append("SECONDARY REF14/WORLD SUMMARY")
        report.append("-" * 100)
        report.append(f"source: {s_sum_path.name if s_sum_path else 'MISSING'}")
        if s_sum:
            for r in s_sum:
                report.append(slim_secondary_row(r))
        else:
            report.append("[MISSING/EMPTY]")
        report.append("")

        report.append("SECONDARY REF14/WORLD DETAIL")
        report.append("-" * 100)
        report.append(f"source: {s_det_path.name if s_det_path else 'MISSING'}")
        if s_det:
            for r in s_det:
                report.append(" | ".join(f"{k}={v}" for k, v in r.items()))
        else:
            report.append("[MISSING/EMPTY]")
        report.append("")

    write_aggregate_csv(OUT / "RUN_STATUS_ALL_VARIANTS.csv", all_status)
    write_aggregate_csv(OUT / "PRIMARY_PAIRWISE_SUMMARY_ALL_VARIANTS.csv", all_primary_summary)
    write_aggregate_csv(OUT / "PRIMARY_PAIRWISE_DETAIL_ALL_VARIANTS.csv", all_primary_detail)
    write_aggregate_csv(OUT / "SECONDARY_REF14_WORLD_SUMMARY_ALL_VARIANTS.csv", all_secondary_summary)
    write_aggregate_csv(OUT / "SECONDARY_REF14_WORLD_DETAIL_ALL_VARIANTS.csv", all_secondary_detail)

    txt = OUT / "FULL_ABLATION_REPORT.txt"
    txt.write_text("\n".join(report) + "\n")

    print("[OK] wrote:")
    print(txt)
    for p in sorted(OUT.glob("*ALL_VARIANTS.csv")):
        print(p)

if __name__ == "__main__":
    main()
