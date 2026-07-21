#!/usr/bin/env python3

import csv
import sys
from pathlib import Path

import numpy as np

BUS_RUN = Path(__file__).resolve().parents[1]
if str(BUS_RUN) not in sys.path:
    sys.path.insert(0, str(BUS_RUN))

from _shared.common.constants import STATIC_CAMERAS
from ap03_scale_common import load_best_colmap_model


AP03_ROOT = Path("results/bus_real_data/03_targetless_colmap_aruco_scale")
OUT_ROOT = AP03_ROOT / "09_pure_targetless_unitless_rig"
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


def vec_norm(p):
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
        raise RuntimeError(f"Root camera {ROOT_CAMERA} not found.")

    return best_model, model_dir, centers, image_names, images, points3d


def strip_old_ap03_scaleless_sections(report_path: Path):
    if not report_path.exists():
        return

    text = report_path.read_text()

    markers = [
        "\n\nAP03 SCALE-LESS COLMAP GEOMETRY EVALUATION\n",
        "\n\nAP03-0 PURE TARGETLESS SCALE-LESS RELATIVE RIG OUTPUT\n",
        "\n\nAP03-0 PURE TARGETLESS SCALE-LESS CAMERA RIG OUTPUT\n",
    ]

    cut_positions = [text.find(m) for m in markers if text.find(m) != -1]
    if cut_positions:
        text = text[:min(cut_positions)].rstrip()

    report_path.write_text(text.rstrip() + "\n")


def append_section(report_path: Path, section: str):
    if not report_path.exists():
        return
    strip_old_ap03_scaleless_sections(report_path)
    text = report_path.read_text().rstrip()
    report_path.write_text(text + "\n\n" + section.strip() + "\n")


def remove_old_pair_ratio_files():
    old_patterns = [
        "AP03_SCALELESS_*",
        "AP03_PURE_TARGETLESS_NORMALIZED_PAIRWISE_DISTANCES.csv",
        "AP03_PURE_TARGETLESS_DISTANCE_RATIOS.csv",
        "AP03_PURE_TARGETLESS_RELATIVE_RIG.txt",
        "AP03_PURE_TARGETLESS_RELATIVE_CAMERA_POSITIONS.csv",
    ]

    roots = [OUT_ROOT, FINAL_ROOT, FINAL_REPORT_ROOT]

    # also remove previous scale-less folder if it exists
    old_roots = [
        AP03_ROOT / "08_scaleless_geometry_evaluation",
        AP03_ROOT / "09_pure_targetless_relative_rig",
    ]
    roots.extend(old_roots)

    for root in roots:
        if not root.exists():
            continue
        for pattern in old_patterns:
            for p in root.glob(pattern):
                if p.is_file():
                    p.unlink()


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    FINAL_ROOT.mkdir(parents=True, exist_ok=True)
    FINAL_REPORT_ROOT.mkdir(parents=True, exist_ok=True)

    remove_old_pair_ratio_files()

    best_model, model_dir, centers, image_names, images, points3d = load_static_camera_centers()

    root_center = centers[ROOT_CAMERA]

    raw_rel = {}
    for cam in STATIC_CAMERAS:
        raw_rel[cam] = centers[cam] - root_center

    # Pure scale-less normalization:
    # root camera is origin.
    # The farthest static camera from the root has distance 1.0.
    # This avoids pairwise/ratio output and gives a clean unitless rig shape.
    non_root_distances = [
        vec_norm(raw_rel[cam])
        for cam in STATIC_CAMERAS
        if cam != ROOT_CAMERA
    ]

    normalization_scale = max(non_root_distances)
    if normalization_scale <= 1e-12:
        raise RuntimeError("Invalid normalization scale.")

    rows = []
    for cam in STATIC_CAMERAS:
        p_raw = raw_rel[cam]
        p_norm = p_raw / normalization_scale

        rows.append({
            "cam": cam,
            "root_camera": ROOT_CAMERA,
            "raw_rel_xyz_colmap_units": fmt_xyz(p_raw),
            "raw_dist_to_root_colmap_units": vec_norm(p_raw),
            "unitless_xyz_norm": fmt_xyz(p_norm),
            "unitless_dist_to_root_norm": vec_norm(p_norm),
            "normalization": "root_camera_origin_and_farthest_static_camera_distance_equals_1",
            "metric_units": "no",
            "uses_aruco": "no",
            "uses_gt_for_output": "no",
            "image_name": image_names.get(cam, ""),
        })

    fields = [
        "cam",
        "root_camera",
        "raw_rel_xyz_colmap_units",
        "raw_dist_to_root_colmap_units",
        "unitless_xyz_norm",
        "unitless_dist_to_root_norm",
        "normalization",
        "metric_units",
        "uses_aruco",
        "uses_gt_for_output",
        "image_name",
    ]

    for root in [OUT_ROOT, FINAL_ROOT, FINAL_REPORT_ROOT]:
        write_csv(root / "AP03_PURE_TARGETLESS_UNITLESS_CAMERA_RIG.csv", rows, fields)

    table_rows = []
    for r in rows:
        table_rows.append([
            r["cam"],
            r["raw_rel_xyz_colmap_units"],
            f(r["raw_dist_to_root_colmap_units"]),
            r["unitless_xyz_norm"],
            f(r["unitless_dist_to_root_norm"]),
        ])

    report = f"""AP03-0 PURE TARGETLESS SCALE-LESS CAMERA RIG OUTPUT
=================================================

Purpose:
This is the pure targetless AP03 output without ArUco, without known marker size, without known marker layout, and without any metric reference.
It uses only raw COLMAP static-camera poses.

Important:
- This is real-life applicable without ArUco.
- It is scale-less and unitless.
- It does not contain meters.
- It does not estimate World -> Ref14.
- It does not estimate metric camera distances.
- It gives the static camera rig shape only up to an unknown global scale, rotation, and translation.

Coordinate convention:
- root camera: {ROOT_CAMERA}
- {ROOT_CAMERA} is set to the local origin.
- axes are inherited from the arbitrary COLMAP frame.
- normalization: farthest static camera from {ROOT_CAMERA} has distance 1.0.
- raw COLMAP units are kept as additional unitless output.

COLMAP model:
- best model: {best_model}
- model dir: {model_dir}
- registered images: {len(images)}
- sparse 3D points: {len(points3d)}
- static cameras: {len(STATIC_CAMERAS)}

Pure targetless camera rig output:
{md_table(["cam", "raw_rel_xyz", "raw_dist", "norm_xyz", "norm_dist"], table_rows)}

Interpretation:
This is the clean pure-targetless output if no metric reference is available.
It tells us where the static cameras are relative to each other in an arbitrary COLMAP coordinate system.
The normalized positions make the rig shape easier to compare or visualize, but they are still not metric.
To obtain meters, AP03 still needs an external metric reference such as known marker size, known marker layout, a measured camera-camera baseline, CAD dimension, depth, or another metric source.
"""

    for root in [OUT_ROOT, FINAL_ROOT, FINAL_REPORT_ROOT]:
        (root / "AP03_PURE_TARGETLESS_UNITLESS_CAMERA_RIG.txt").write_text(report)

    append_section(FINAL_ROOT / "AP03_FINAL_RESULT.txt", report)
    append_section(FINAL_REPORT_ROOT / "AP03_FINAL_RESULT.txt", report)

    print(report)
    print()
    print("[OK] wrote clean pure targetless unitless AP03 output:")
    print(" ", FINAL_REPORT_ROOT / "AP03_PURE_TARGETLESS_UNITLESS_CAMERA_RIG.txt")
    print(" ", FINAL_REPORT_ROOT / "AP03_PURE_TARGETLESS_UNITLESS_CAMERA_RIG.csv")
    print("[OK] removed old pair/ratio scale-less files from final AP03 folders")
    print("[OK] appended clean AP03-0 section to AP03_FINAL_RESULT.txt")


if __name__ == "__main__":
    main()
