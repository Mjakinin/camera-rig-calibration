#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import xml.etree.ElementTree as ET
from itertools import combinations
from pathlib import Path

import numpy as np

CAMS = ["cam_edge_0", "cam_edge_1", "cam_edge_3", "cam_edge_5"]
METHODS = ["AP01", "AP02", "AP03"]


def args():
    p = argparse.ArgumentParser(
        description="Write the GT-free AP01/AP02/AP03 real-data report."
    )
    p.add_argument("--dataset", required=True)
    p.add_argument("--results-root", required=True)
    p.add_argument(
        "--simulation-world-sdf",
        default="src/calib_lab/bus_real_data/worlds/"
        "bus_real_data_moving_camera.sdf",
    )
    return p.parse_args()


def load_json(path, default=None):
    path = Path(path)
    if not path.is_file():
        return {} if default is None else default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {} if default is None else default


def load_csv(path):
    path = Path(path)
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def save_csv(path, rows, fields):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def positions(path):
    out = {}
    for row in load_csv(path):
        cam = row.get("entity_id")
        if cam not in CAMS:
            continue
        try:
            out[cam] = np.array(
                [float(row["x_m"]), float(row["y_m"]), float(row["z_m"])]
            )
        except (KeyError, TypeError, ValueError):
            pass
    return out


def sdf_positions(path):
    path = Path(path)
    if not path.is_file():
        return {}
    out = {}
    for model in ET.parse(path).getroot().iter("model"):
        name = model.attrib.get("name")
        pose = model.find("pose")
        if name in CAMS and pose is not None and pose.text:
            values = [float(v) for v in pose.text.split()]
            if len(values) >= 3:
                out[name] = np.array(values[:3])
    return out


def key(a, b):
    return f"{a}--{b}"


def distances(pos):
    return {
        key(a, b): float(np.linalg.norm(pos[a] - pos[b]))
        for a, b in combinations(CAMS, 2)
        if a in pos and b in pos
    }


def fmt(value, digits=6):
    try:
        value = float(value)
        return f"{value:.{digits}f}" if math.isfinite(value) else "NA"
    except (TypeError, ValueError):
        return "NA"


def method_files(root):
    return {
        "AP01": {
            "root": root / "02_ap01_real",
            "pose": root / "02_ap01_real/03_static_extrinsics/"
            "AP01_STATIC_CAMERA_POSES_CAM3_REFERENCE.csv",
            "diag": root / "02_ap01_real/03_static_extrinsics/"
            "AP01_DIAGNOSTICS.json",
        },
        "AP02": {
            "root": root / "03_ap02_real",
            "pose": root / "03_ap02_real/07_graph_ba/with_moving/"
            "optimized_static_camera_poses_ref_marker.csv",
            "diag": root / "03_ap02_real/08_final_results/"
            "AP02_DIAGNOSTICS.json",
        },
        "AP03": {
            "root": root / "04_ap03_real",
            "pose": root / "04_ap03_real/07_final_results/"
            "AP03_MARKER_SIZE_SCALE_ONLY_STATIC_CAMERA_POSES.csv",
            "diag": root / "04_ap03_real/07_final_results/"
            "AP03_DIAGNOSTICS.json",
        },
    }


def main():
    a = args()
    dataset = Path(a.dataset).resolve()
    root = Path(a.results_root).resolve()
    world = Path(a.simulation_world_sdf).resolve()
    final = root / "99_FINAL_RESULTS"
    final.mkdir(parents=True, exist_ok=True)

    files = method_files(root)
    status = {}
    pos = {}
    dist = {}
    diag = {}
    for method in METHODS:
        status[method] = load_json(
            files[method]["root"] / "METHOD_STATUS.json",
            {"status": "NOT_RUN", "success": False},
        )
        pos[method] = positions(files[method]["pose"])
        dist[method] = distances(pos[method])
        diag[method] = load_json(files[method]["diag"], {})

    nominal = distances(sdf_positions(world))
    nominal_path = final / "simulation_gt_reference_distances.json"
    nominal_path.write_text(
        json.dumps(
            {
                "source_world_sdf": str(world),
                "warning": (
                    "Nominal Gazebo camera-center distances only; "
                    "not real-world ground truth."
                ),
                "distances_m": nominal,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    measured_path = final / "measured_reference_distances.json"
    if not measured_path.is_file():
        measured_path.write_text(
            json.dumps(
                {key(a, b): None for a, b in combinations(CAMS, 2)},
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    measured = load_json(measured_path, {})

    rows = []
    for first, second in combinations(CAMS, 2):
        pair = key(first, second)
        try:
            real_ref = (
                float(measured.get(pair))
                if measured.get(pair) is not None
                else None
            )
        except (TypeError, ValueError):
            real_ref = None
        nominal_ref = nominal.get(pair)
        row = {
            "camera_a": first,
            "camera_b": second,
            "simulation_gt_reference_m": nominal_ref,
            "measured_real_reference_m": real_ref,
        }
        for method in METHODS:
            prefix = method.lower()
            value = dist[method].get(pair)
            row[f"{prefix}_distance_m"] = value
            row[f"{prefix}_vs_simulation_gt_cm"] = (
                abs(value - nominal_ref) * 100
                if value is not None and nominal_ref is not None
                else None
            )
            row[f"{prefix}_vs_measured_real_cm"] = (
                abs(value - real_ref) * 100
                if value is not None and real_ref is not None
                else None
            )
        rows.append(row)

    fields = [
        "camera_a", "camera_b",
        "ap01_distance_m", "ap02_distance_m", "ap03_distance_m",
        "simulation_gt_reference_m",
        "ap01_vs_simulation_gt_cm", "ap02_vs_simulation_gt_cm",
        "ap03_vs_simulation_gt_cm",
        "measured_real_reference_m",
        "ap01_vs_measured_real_cm", "ap02_vs_measured_real_cm",
        "ap03_vs_measured_real_cm",
    ]
    pair_csv = final / "REAL_DATA_PAIRWISE_DISTANCES.csv"
    save_csv(pair_csv, rows, fields)

    status_rows = []
    for method in METHODS:
        s = status[method]
        status_rows.append(
            {
                "method": method,
                "status": s.get("status", "UNKNOWN"),
                "success": s.get("success", False),
                "available_static_cameras": ";".join(sorted(pos[method])),
                "camera_count": len(pos[method]),
                "runtime_seconds": s.get(
                    "runtime_seconds", diag[method].get("runtime_seconds")
                ),
                "error": s.get("error", ""),
                "pose_file": str(files[method]["pose"]),
                "diagnostics_file": str(files[method]["diag"]),
            }
        )
    status_csv = final / "REAL_DATA_METHOD_STATUS.csv"
    save_csv(
        status_csv,
        status_rows,
        [
            "method", "status", "success", "available_static_cameras",
            "camera_count", "runtime_seconds", "error", "pose_file",
            "diagnostics_file",
        ],
    )

    moving = len(list((dataset / "raw_images/moving").glob("frame_*.png")))
    lines = [
        "REAL-DATA CAMERA-RIG CALIBRATION: AP01 / AP02 / AP03",
        "=" * 150,
        "",
        f"Dataset: {dataset}",
        "Primary evaluation mode: GT-free.",
        "Nominal Gazebo distances are context only, not real-world errors.",
        f"Moving frames: {moving}",
        "",
        "METHOD STATUS",
        "-" * 150,
        f"{'Method':8s}{'Status':28s}{'Cameras':>10s}{'Runtime [s]':>18s}  Error",
    ]
    for method in METHODS:
        s = status[method]
        error = str(s.get("error", ""))
        lines.append(
            f"{method:8s}{str(s.get('status', 'NOT_RUN')):28s}"
            f"{len(pos[method]):>4d}/4      "
            f"{fmt(s.get('runtime_seconds'), 2):>15s}  {error[:65]}"
        )

    lines += [
        "",
        "PAIRWISE CAMERA DISTANCES WITH NOMINAL-LAYOUT COMPARISON",
        "-" * 150,
        f"{'Camera pair':29s}{'AP01 [m]':>12s}{'AP02 [m]':>12s}"
        f"{'AP03 [m]':>12s}{'Nominal [m]':>13s}"
        f"{'|AP01-Nom| [cm]':>18s}{'|AP02-Nom| [cm]':>18s}"
        f"{'|AP03-Nom| [cm]':>18s}",
    ]
    for row in rows:
        pair = f"{row['camera_a']} - {row['camera_b']}"
        lines.append(
            f"{pair:29s}{fmt(row['ap01_distance_m']):>12s}"
            f"{fmt(row['ap02_distance_m']):>12s}"
            f"{fmt(row['ap03_distance_m']):>12s}"
            f"{fmt(row['simulation_gt_reference_m']):>13s}"
            f"{fmt(row['ap01_vs_simulation_gt_cm'], 3):>18s}"
            f"{fmt(row['ap02_vs_simulation_gt_cm'], 3):>18s}"
            f"{fmt(row['ap03_vs_simulation_gt_cm'], 3):>18s}"
        )

    lines += ["", "INDEPENDENT REAL-WORLD REFERENCES", "-" * 150]
    if any(r["measured_real_reference_m"] is not None for r in rows):
        for row in rows:
            pair = f"{row['camera_a']} - {row['camera_b']}"
            lines.append(
                f"{pair:29s} measured={fmt(row['measured_real_reference_m'])} m"
                f" AP01={fmt(row['ap01_vs_measured_real_cm'], 3)} cm"
                f" AP02={fmt(row['ap02_vs_measured_real_cm'], 3)} cm"
                f" AP03={fmt(row['ap03_vs_measured_real_cm'], 3)} cm"
            )
    else:
        lines.append(
            "No independent measurements entered; real accuracy remains unknown."
        )

    lines += ["", "METHOD DIAGNOSTICS", "-" * 150]
    for method in METHODS:
        lines += [method, json.dumps(diag[method], indent=2, sort_keys=True), ""]

    lines += [
        "INTERPRETATION",
        "-" * 150,
        "- Pairwise distances evaluate translation and metric scale, not orientation.",
        "- Nominal-layout differences are plausibility context, not real accuracy.",
        "- Real accuracy needs independent measurements or trusted transforms.",
        "- Reprojection error alone does not prove correct global geometry.",
        "",
        f"Pairwise CSV: {pair_csv}",
        f"Method-status CSV: {status_csv}",
        f"Nominal-reference JSON: {nominal_path}",
        f"Measured-reference template: {measured_path}",
        "",
    ]
    report = final / "REAL_DATA_ALL_METHODS.txt"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(report.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
