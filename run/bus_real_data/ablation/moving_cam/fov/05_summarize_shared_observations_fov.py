#!/usr/bin/env python3
from pathlib import Path
import csv

ROOT = Path("results/bus_real_data/ablation/moving_cam/fov/01_shared_observations")
OUT = Path("results/bus_real_data/ablation/moving_cam/fov/99_summary")
OUT.mkdir(parents=True, exist_ok=True)

VARIANTS = [
    "fov_40deg",
    "fov_50deg",
    "fov_60deg",
    "fov_69deg_baseline",
    "fov_80deg",
    "fov_90deg",
    "fov_100deg",
    "fov_110deg",
    "fov_120deg",
    "fov_140deg_extreme",
]

def read_rows(p: Path):
    if not p.exists():
        return []
    with p.open(newline="", errors="replace") as f:
        return list(csv.DictReader(f))

def first_existing_key(row, keys):
    for k in keys:
        if k in row:
            return k
    return None

def unique_count(rows, keys):
    vals = set()
    for r in rows:
        k = first_existing_key(r, keys)
        if k and r.get(k):
            vals.add(r[k])
    return len(vals)

def main():
    rows_out = []

    for v in VARIANTS:
        d = ROOT / v
        all_rows = read_rows(d / "shared_all_aruco_observations.csv")
        moving_rows = read_rows(d / "shared_moving_aruco_observations.csv")
        static_rows = read_rows(d / "shared_static_aruco_observations.csv")

        rows_out.append({
            "variant": v,
            "all_observations": len(all_rows),
            "moving_observations": len(moving_rows),
            "static_observations": len(static_rows),
            "unique_markers": unique_count(all_rows, ["marker_id", "aruco_id", "id"]),
            "unique_images_with_observations": unique_count(all_rows, ["image_name", "image", "filename", "frame"]),
        })

    csv_path = OUT / "moving_cam_fov_shared_observations_summary.csv"
    txt_path = OUT / "moving_cam_fov_shared_observations_summary.txt"

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        writer.writeheader()
        writer.writerows(rows_out)

    lines = []
    lines.append("MOVING CAMERA FOV — SHARED ARUCO OBSERVATIONS SUMMARY")
    lines.append("=====================================================")
    lines.append("")
    headers = list(rows_out[0].keys())
    widths = {h: max(len(h), *(len(str(r[h])) for r in rows_out)) for h in headers}
    lines.append(" | ".join(h.ljust(widths[h]) for h in headers))
    lines.append(" | ".join("-" * widths[h] for h in headers))
    for r in rows_out:
        lines.append(" | ".join(str(r[h]).ljust(widths[h]) for h in headers))

    txt_path.write_text("\n".join(lines) + "\n")

    print("\n".join(lines))
    print("\n[OK] wrote:")
    print(f"  {csv_path}")
    print(f"  {txt_path}")

if __name__ == "__main__":
    main()
