#!/usr/bin/env python3

import csv
from pathlib import Path

ROOT = Path("results/bus_real_data/ablation/moving_cam/res/01_shared_observations")
OUT = Path("results/bus_real_data/ablation/moving_cam/res/99_summary")
OUT.mkdir(parents=True, exist_ok=True)

CANDIDATE_FILES = [
    "shared_all_aruco_observations.csv",
    "shared_moving_aruco_observations.csv",
    "shared_static_aruco_observations.csv",
]

def count_csv(path: Path):
    if not path.exists():
        return 0
    with path.open(newline="") as fp:
        return sum(1 for _ in csv.DictReader(fp))

def unique_values(path: Path, field: str):
    if not path.exists():
        return set()
    vals = set()
    with path.open(newline="") as fp:
        for r in csv.DictReader(fp):
            if field in r and r[field] != "":
                vals.add(r[field])
    return vals

rows = []

for variant_dir in sorted(p for p in ROOT.iterdir() if p.is_dir() and not p.name.startswith("_")):
    variant = variant_dir.name

    all_csv = variant_dir / "shared_all_aruco_observations.csv"
    moving_csv = variant_dir / "shared_moving_aruco_observations.csv"
    static_csv = variant_dir / "shared_static_aruco_observations.csv"

    n_all = count_csv(all_csv)
    n_moving = count_csv(moving_csv)
    n_static = count_csv(static_csv)

    marker_fields = ["marker_id", "aruco_id", "id"]
    image_fields = ["image_name", "frame_id", "source_image"]

    markers = set()
    images = set()

    for f in [all_csv, moving_csv, static_csv]:
        if not f.exists():
            continue
        with f.open(newline="") as fp:
            reader = csv.DictReader(fp)
            for r in reader:
                for mf in marker_fields:
                    if mf in r and r[mf] != "":
                        markers.add(r[mf])
                        break
                for imf in image_fields:
                    if imf in r and r[imf] != "":
                        images.add(r[imf])
                        break

    rows.append({
        "variant": variant,
        "all_observations": n_all,
        "moving_observations": n_moving,
        "static_observations": n_static,
        "unique_markers": len(markers),
        "unique_images_with_observations": len(images),
    })

fields = [
    "variant",
    "all_observations",
    "moving_observations",
    "static_observations",
    "unique_markers",
    "unique_images_with_observations",
]

out_csv = OUT / "moving_cam_res_shared_observations_summary.csv"
with out_csv.open("w", newline="") as fp:
    w = csv.DictWriter(fp, fieldnames=fields)
    w.writeheader()
    w.writerows(rows)

def table(headers, rows):
    widths = [len(h) for h in headers]
    data = []
    for r in rows:
        row = [str(r[h]) for h in headers]
        data.append(row)
        for i, c in enumerate(row):
            widths[i] = max(widths[i], len(c))
    lines = []
    lines.append(" | ".join(headers[i].ljust(widths[i]) for i in range(len(headers))))
    lines.append(" | ".join("-" * widths[i] for i in range(len(headers))))
    for row in data:
        lines.append(" | ".join(row[i].ljust(widths[i]) for i in range(len(headers))))
    return "\n".join(lines)

txt = "MOVING CAMERA RESOLUTION — SHARED ARUCO OBSERVATIONS SUMMARY\n"
txt += "=============================================================\n\n"
txt += table(fields, rows)
txt += "\n"

out_txt = OUT / "moving_cam_res_shared_observations_summary.txt"
out_txt.write_text(txt)

print(txt)
print("[OK] wrote:")
print(" ", out_csv)
print(" ", out_txt)
