#!/usr/bin/env python3

import csv
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

BUS_RUN = Path(__file__).resolve().parents[1]
if str(BUS_RUN) not in sys.path:
    sys.path.insert(0, str(BUS_RUN))

from _shared.common.constants import STATIC_CAMERAS
from ap03_scale_common import load_best_colmap_model


AP03_ROOT = Path("results/bus_real_data/03_targetless_colmap_aruco_scale")
OUT_ROOT = AP03_ROOT / "09_pure_targetless_relative_rig"
FINAL_ROOT = AP03_ROOT / "07_final_results"
FINAL_REPORT_ROOT = Path("results/bus_real_data/99_FINAL_RESULTS_FOR_REPORT/AP03")

ROOT_CAMERA = "cam_edge_3"


def source_id_from_image_name(name: str) -> str:
    if name.startswith("static_") and name.endswith(".png"):
        return name[len("static_"):-len(".png")]
    return name


def f(x, n=3):
    return f"{float(x):.{n}f}"


def fmt_xyz(p):
    return f"({f(p[0])}, {f(p[1])}, {f(p[2])})"


def norm(p):
    return float(np.linalg.norm(np.asarray(p, dtype=float)))


def md_table(headers, rows):
    widths = [len(str(h)) for h in headers]
    for row in rows:
        for i, c in enumerate(row):
            widths[i] = max(widths[i], len(str(c)))

    out = []
    out.append(" | ".join(str(h).ljust(widths[i]) for i, h in enumerate(headers)))
    out.append(" | ".join("-" * widths[i] for i in range(len(headers))))
    for row in rows:
        out.append(" | ".join(str(c).ljust(widths[i]) for i, c in enumerate(row)))
    return "\n".join(out)


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def append_section(report_path: Path, marker: str, section: str):
    if not report_path.exists():
        return

    text = report_path.read_text()
    if marker in text:
        text = text.split(marker)[0].rstrip()

    report_path.write_text(text.rstrip() + "\n\n" + marker + "\n" + section.strip() + "\n")


def load_static_camera_centers():
    best_model, model_dir, cameras, images, points3d = load_best_colmap_model()

    centers = {}
    image_names = {}

    for image_name, payload in images.items():
        if not image_name.startswith("static_"):
            continue

        cam = source_id_from_image_name(image_name)
        if cam not in STATIC_CAMERAS:
            continue

        T_col_cam = payload["T_col_cam"]
        centers[cam] = np.asarray(T_col_cam[:3, 3], dtype=float)
        image_names[cam] = image_name

    missing = [cam for cam in STATIC_CAMERAS if cam not in centers]
    if missing:
        raise RuntimeError(f"Missing static cameras in COLMAP model: {missing}")

    if ROOT_CAMERA not in centers:
        raise RuntimeError(f"Root camera {ROOT_CAMERA} not found in COLMAP static cameras.")

    return best_model, model_dir, centers, image_names, images, points3d


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    FINAL_ROOT.mkdir(parents=True, exist_ok=True)
    FINAL_REPORT_ROOT.mkdir(parents=True, exist_ok=True)

    best_model, model_dir, centers, image_names, images, points3d = load_static_camera_centers()

    # Raw pairwise distances in arbitrary COLMAP units.
    raw_pair_dists = []
    for a, b in combinations(STATIC_CAMERAS, 2):
        d = norm(centers[a] - centers[b])
        raw_pair_dists.append(d)

    median_pair_dist = float(np.median(raw_pair_dists))
    if median_pair_dist <= 1e-12:
        raise RuntimeError("Invalid median pair distance.")

    # Scale-less normalization:
    # root camera becomes origin, median static-camera pair distance becomes 1.0.
    root_center = centers[ROOT_CAMERA]

    camera_rows = []
    for cam in STATIC_CAMERAS:
        p_raw_root = centers[cam] - root_center
        p_norm = p_raw_root / median_pair_dist

        camera_rows.append({
            "cam": cam,
            "root_camera": ROOT_CAMERA,
            "normalization": "median_static_pair_distance_equals_1",
            "x_norm": p_norm[0],
            "y_norm": p_norm[1],
            "z_norm": p_norm[2],
            "xyz_norm": fmt_xyz(p_norm),
            "dist_to_root_norm": norm(p_norm),
            "raw_colmap_x": centers[cam][0],
            "raw_colmap_y": centers[cam][1],
            "raw_colmap_z": centers[cam][2],
            "image_name": image_names.get(cam, ""),
            "real_life_output": "yes_scale_less_unitless",
            "metric_units": "no",
        })

    pair_rows = []
    for a, b in combinations(STATIC_CAMERAS, 2):
        d_raw = norm(centers[a] - centers[b])
        d_norm = d_raw / median_pair_dist

        pair_rows.append({
            "cam_a": a,
            "cam_b": b,
            "normalization": "median_static_pair_distance_equals_1",
            "raw_colmap_distance": d_raw,
            "normalized_distance": d_norm,
            "metric_units": "no",
            "real_life_output": "yes_scale_less_unitless",
        })

    ratio_rows = []
    for r1, r2 in combinations(pair_rows, 2):
        pair1 = f"{r1['cam_a']}-{r1['cam_b']}"
        pair2 = f"{r2['cam_a']}-{r2['cam_b']}"

        ratio = float(r1["raw_colmap_distance"]) / float(r2["raw_colmap_distance"])

        ratio_rows.append({
            "ratio_id": f"{pair1}_over_{pair2}",
            "numerator_pair": pair1,
            "denominator_pair": pair2,
            "distance_ratio": ratio,
            "metric_units": "no",
            "real_life_output": "yes_scale_less_unitless",
        })

    camera_fields = [
        "cam",
        "root_camera",
        "normalization",
        "x_norm",
        "y_norm",
        "z_norm",
        "xyz_norm",
        "dist_to_root_norm",
        "raw_colmap_x",
        "raw_colmap_y",
        "raw_colmap_z",
        "image_name",
        "real_life_output",
        "metric_units",
    ]

    pair_fields = [
        "cam_a",
        "cam_b",
        "normalization",
        "raw_colmap_distance",
        "normalized_distance",
        "metric_units",
        "real_life_output",
    ]

    ratio_fields = [
        "ratio_id",
        "numerator_pair",
        "denominator_pair",
        "distance_ratio",
        "metric_units",
        "real_life_output",
    ]

    for root in [OUT_ROOT, FINAL_ROOT, FINAL_REPORT_ROOT]:
        write_csv(root / "AP03_PURE_TARGETLESS_RELATIVE_CAMERA_POSITIONS.csv", camera_rows, camera_fields)
        write_csv(root / "AP03_PURE_TARGETLESS_NORMALIZED_PAIRWISE_DISTANCES.csv", pair_rows, pair_fields)
        write_csv(root / "AP03_PURE_TARGETLESS_DISTANCE_RATIOS.csv", ratio_rows, ratio_fields)

    cam_table = []
    for r in camera_rows:
        cam_table.append([
            r["cam"],
            r["xyz_norm"],
            f(r["dist_to_root_norm"]),
        ])

    pair_table = []
    for r in pair_rows:
        pair_table.append([
            f"{r['cam_a']}-{r['cam_b']}",
            f(r["normalized_distance"]),
        ])

    ratio_table = []
    for r in ratio_rows[:10]:
        ratio_table.append([
            r["numerator_pair"],
            r["denominator_pair"],
            f(r["distance_ratio"]),
        ])

    report = f"""AP03-0 PURE TARGETLESS SCALE-LESS RELATIVE RIG OUTPUT
====================================================

Purpose:
This is the pure targetless AP03 output without ArUco, without known marker size, and without any metric reference.
It uses only raw COLMAP static-camera centers.

Important:
- This output is real-life applicable without ArUco.
- It is scale-less and unitless.
- It does not contain meters.
- It does not estimate World -> Ref14.
- It does not estimate metric camera distances.
- It represents the static camera rig geometry only up to an unknown global scale, rotation, and translation.

Coordinate convention:
- root camera: {ROOT_CAMERA}
- {ROOT_CAMERA} is set to the local origin.
- scale normalization: median static-camera pair distance = 1.0
- axes are inherited from the arbitrary COLMAP frame.

COLMAP model:
- best model: {best_model}
- model dir: {model_dir}
- registered images: {len(images)}
- sparse 3D points: {len(points3d)}
- static cameras: {len(STATIC_CAMERAS)}

Scale-less camera positions relative to {ROOT_CAMERA}:
{md_table(["cam", "xyz_norm", "dist_to_root_norm"], cam_table)}

Normalized pairwise static-camera distances:
{md_table(["pair", "normalized_distance"], pair_table)}

Distance ratios:
{md_table(["num_pair", "den_pair", "ratio"], ratio_table)}

Interpretation:
This is the deployable pure-targetless result if no metric reference is available.
It tells us the shape of the static camera rig, but not its real size in meters.
To convert this into metric camera positions, we still need one external metric reference, such as a known marker size, known marker layout, known camera-camera baseline, CAD dimension, depth sensor, or another measured distance.
"""

    for root in [OUT_ROOT, FINAL_ROOT, FINAL_REPORT_ROOT]:
        (root / "AP03_PURE_TARGETLESS_RELATIVE_RIG.txt").write_text(report)

    marker = "AP03-0 PURE TARGETLESS SCALE-LESS RELATIVE RIG OUTPUT"
    append_section(FINAL_ROOT / "AP03_FINAL_RESULT.txt", marker, report)
    append_section(FINAL_REPORT_ROOT / "AP03_FINAL_RESULT.txt", marker, report)

    print(report)
    print()
    print("[OK] wrote pure targetless scale-less AP03 outputs:")
    print(" ", FINAL_REPORT_ROOT / "AP03_PURE_TARGETLESS_RELATIVE_RIG.txt")
    print(" ", FINAL_REPORT_ROOT / "AP03_PURE_TARGETLESS_RELATIVE_CAMERA_POSITIONS.csv")
    print(" ", FINAL_REPORT_ROOT / "AP03_PURE_TARGETLESS_NORMALIZED_PAIRWISE_DISTANCES.csv")
    print(" ", FINAL_REPORT_ROOT / "AP03_PURE_TARGETLESS_DISTANCE_RATIOS.csv")
    print("[OK] appended section to AP03_FINAL_RESULT.txt")


if __name__ == "__main__":
    main()
