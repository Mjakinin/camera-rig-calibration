#!/usr/bin/env python3
from pathlib import Path
import csv
import os
import re

STUDY_ROOT = Path(os.environ.get("STUDY_ROOT", "results/bus_real_data/ablation/moving_cam/res"))
FINAL = STUDY_ROOT / "final_results"

OUT_CSV = FINAL / "GT_MAP_VS_EST_MAP_CAMERA_TABLE.csv"
OUT_TXT = FINAL / "GT_MAP_VS_EST_MAP_CAMERA_TABLE.txt"
OUT_MEAN_CSV = FINAL / "GT_MAP_VS_EST_MAP_MEAN_TABLE.csv"
OUT_MEAN_TXT = FINAL / "GT_MAP_VS_EST_MAP_MEAN_TABLE.txt"

STATUS_CSV = next(iter(FINAL.glob("*pipeline_status_summary.csv")), None)

CAM_RE = re.compile(
    r"^\s*(cam_edge_\d+)\s*\|\s*"
    r"([0-9.+\-eE]+)\s*\|\s*"
    r"([0-9.+\-eE]+)\s*\|\s*"
    r"([0-9.+\-eE]+)\s*\|\s*"
    r"([0-9.+\-eE]+)\s*\|\s*"
    r"([0-9.+\-eE]+)\s*\|\s*"
    r"(\([^)]+\))\s*\|\s*"
    r"(\([^)]+\))"
)

MEAN_RE = re.compile(r"mean[_ ]translation[_ ]error.*?([0-9.+\-eE]+)", re.IGNORECASE)
ROT_RE = re.compile(r"mean[_ ]rotation[_ ]error.*?([0-9.+\-eE]+)", re.IGNORECASE)

def variant_from_report(p: Path):
    return p.name.replace("_FINAL_RESULT.txt", "")

def current_approach_from_line(line, current):
    if "AP01" in line:
        return "AP01"
    if "AP02" in line:
        return "AP02"
    if "AP03" in line:
        return "AP03"
    return current

def read_status():
    rows = {}
    if not STATUS_CSV or not STATUS_CSV.exists():
        return rows

    with STATUS_CSV.open(newline="", errors="replace") as f:
        for r in csv.DictReader(f):
            rows[(r.get("variant", ""), r.get("approach", ""))] = r
    return rows

def extract_camera_rows(report: Path, status_rows):
    variant = variant_from_report(report)
    lines = report.read_text(errors="replace").splitlines()

    rows = []
    current_approach = None

    for line in lines:
        current_approach = current_approach_from_line(line, current_approach)

        m = CAM_RE.search(line)
        if not m:
            continue

        approach = current_approach or "UNKNOWN"
        status = status_rows.get((variant, approach), {}).get("status", "")

        rows.append({
            "variant": variant,
            "approach": approach,
            "status": status,
            "camera": m.group(1),
            "t_err_cm": m.group(2),
            "r_err_deg": m.group(3),
            "dX_cm": m.group(4),
            "dY_cm": m.group(5),
            "dZ_cm": m.group(6),
            "estimated_xyz_m": m.group(7),
            "gt_xyz_m": m.group(8),
        })

    # Deduplicate same row.
    seen = set()
    dedup = []
    for r in rows:
        key = tuple(r.items())
        if key in seen:
            continue
        seen.add(key)
        dedup.append(r)

    return dedup

def load_mean_rows(status_rows):
    out = []
    for (variant, approach), r in sorted(status_rows.items()):
        out.append({
            "variant": variant,
            "approach": approach,
            "status": r.get("status", ""),
            "mean_t_cm": r.get("mean_t_cm", ""),
            "mean_r_deg": r.get("mean_r_deg", ""),
            "registered_static": r.get("registered_static", ""),
            "registered_moving": r.get("registered_moving", ""),
            "ref14_obs": r.get("ref14_obs", ""),
            "corner_fit_cm": r.get("corner_fit_cm", ""),
        })
    return out

def write_csv(path, rows):
    if not rows:
        path.write_text("[NO ROWS]\n")
        return
    with path.open("w", newline="", errors="replace") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

def write_txt_table(path, rows, title):
    with path.open("w", errors="replace") as f:
        f.write(title + "\n")
        f.write("=" * len(title) + "\n\n")

        if not rows:
            f.write("[NO ROWS FOUND]\n")
            return

        cols = list(rows[0].keys())
        widths = {}
        for c in cols:
            widths[c] = max(len(c), *(len(str(r.get(c, ""))) for r in rows))

        f.write(" | ".join(c.ljust(widths[c]) for c in cols) + "\n")
        f.write("-+-".join("-" * widths[c] for c in cols) + "\n")

        for r in rows:
            f.write(" | ".join(str(r.get(c, "")).ljust(widths[c]) for c in cols) + "\n")

def main():
    status_rows = read_status()

    reports = sorted([
        p for p in FINAL.glob("*_FINAL_RESULT.txt")
        if not p.name.startswith("ALL_")
    ])

    camera_rows = []
    for report in reports:
        camera_rows.extend(extract_camera_rows(report, status_rows))

    mean_rows = load_mean_rows(status_rows)

    # Sort nicely.
    order = {
        "res_320x180_extreme": 0,
        "res_640x360": 1,
        "res_960x540": 2,
        "res_1280x720_baseline": 3,
        "res_1920x1080": 4,
    }

    camera_rows.sort(key=lambda r: (
        order.get(r["variant"], 999),
        r["variant"],
        r["approach"],
        r["camera"],
    ))

    mean_rows.sort(key=lambda r: (
        order.get(r["variant"], 999),
        r["variant"],
        r["approach"],
    ))

    write_csv(OUT_CSV, camera_rows)
    write_txt_table(OUT_TXT, camera_rows, "GT MAP VS EST MAP — CAMERA TABLE")

    write_csv(OUT_MEAN_CSV, mean_rows)
    write_txt_table(OUT_MEAN_TXT, mean_rows, "GT MAP VS EST MAP — MEAN TABLE")

    print("[OK] wrote:")
    print(" ", OUT_TXV if False else OUT_TXT)
    print(" ", OUT_CSV)
    print(" ", OUT_MEAN_TXT)
    print(" ", OUT_MEAN_CSV)

if __name__ == "__main__":
    main()
