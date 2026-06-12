#!/usr/bin/env python3
import csv
import json
import math
import importlib.util
from pathlib import Path

import numpy as np


REPO = Path(".")
CHAIN_SCRIPT = Path("run/bus_real_data/13_eval_moving_relay_chains.py")
RELAY_CSV = Path("results/bus_real_data/06_moving_relay_chain_eval/relay_chain_results.csv")

OUT = Path("results/bus_real_data/07_final_extrinsics_cam3_reference")
OUT.mkdir(parents=True, exist_ok=True)


def load_chain_module():
    spec = importlib.util.spec_from_file_location("relay_chain_eval", CHAIN_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def matrix_to_list(T):
    return [[float(x) for x in row] for row in T]


def R_to_quat_xyzw(R):
    # Returns quaternion as [x, y, z, w]
    tr = float(np.trace(R))
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s

    return [float(x), float(y), float(z), float(w)]


def R_to_rpy_deg(R):
    # R = Rz(yaw) Ry(pitch) Rx(roll)
    sy = -R[2, 0]
    sy = max(-1.0, min(1.0, float(sy)))
    pitch = math.asin(sy)

    if abs(math.cos(pitch)) > 1e-8:
        roll = math.atan2(R[2, 1], R[2, 2])
        yaw = math.atan2(R[1, 0], R[0, 0])
    else:
        roll = 0.0
        yaw = math.atan2(-R[0, 1], R[1, 1])

    return [math.degrees(roll), math.degrees(pitch), math.degrees(yaw)]


def pose_payload(T):
    R = T[:3, :3]
    t = T[:3, 3]
    return {
        "translation_m": {
            "x": float(t[0]),
            "y": float(t[1]),
            "z": float(t[2]),
        },
        "rotation_rpy_deg": {
            "roll": float(R_to_rpy_deg(R)[0]),
            "pitch": float(R_to_rpy_deg(R)[1]),
            "yaw": float(R_to_rpy_deg(R)[2]),
        },
        "quaternion_xyzw": R_to_quat_xyzw(R),
        "matrix_4x4": matrix_to_list(T),
    }


def load_relay_rows():
    with RELAY_CSV.open() as fp:
        return list(csv.DictReader(fp))


def best_relay_row(pair, method):
    rows = [
        r for r in load_relay_rows()
        if r["pair"] == pair and r["method"] == method
    ]
    if not rows:
        raise RuntimeError(f"No rows for {pair} / {method}")
    rows.sort(key=lambda r: float(r["translation_error_m"]))
    return rows[0]


def compute_direct_cam3_to_cam1(m, marker_id=1):
    static_det = m.load_static_detections()

    root_cam = "cam_edge_3"
    target_cam = "cam_edge_1"

    T_root_marker = m.T_from_detection(static_det[(root_cam, marker_id)])
    T_target_marker = m.T_from_detection(static_det[(target_cam, marker_id)])

    T_est = T_root_marker @ m.invT(T_target_marker)

    T_W_root = m.T_W_optical_from_sdf_model(root_cam)
    T_W_target = m.T_W_optical_from_sdf_model(target_cam)
    T_gt = m.invT(T_W_root) @ T_W_target

    return {
        "name": "cam_edge_3_to_cam_edge_1_direct_static",
        "target_camera": target_cam,
        "method": "direct_static_aruco",
        "selected_marker": marker_id,
        "T_est": T_est,
        "T_gt": T_gt,
        "translation_error_m": m.trans_error(T_est, T_gt),
        "rotation_error_deg": m.rot_error_deg(T_est, T_gt),
        "notes": "Direct static-to-static baseline using shared marker 1.",
    }


def compute_relay_from_row(m, row):
    static_det = m.load_static_detections()
    moving_det = m.load_moving_detections()
    best = m.load_best_registered()
    colmap_poses = m.load_colmap_poses()
    colmap_scale = m.load_colmap_scale()

    root_cam = "cam_edge_3"
    target_cam = row["target_cam"]
    root_marker = int(row["root_marker"])
    target_marker = int(row["target_marker"])
    root_frame = int(row["root_frame"])
    target_frame = int(row["target_frame"])
    method = row["method"]

    T_root_marker = m.T_from_detection(static_det[(root_cam, root_marker)])
    T_movi_marker = m.T_from_detection(moving_det[(root_frame, root_marker)])
    T_target_marker = m.T_from_detection(static_det[(target_cam, target_marker)])
    T_movj_marker = m.T_from_detection(moving_det[(target_frame, target_marker)])

    T_root_movi = T_root_marker @ m.invT(T_movi_marker)
    T_target_movj = T_target_marker @ m.invT(T_movj_marker)

    if method == "GT_motion":
        T_W_movi = m.T_W_optical_from_best_row(best[root_marker])
        T_W_movj = m.T_W_optical_from_best_row(best[target_marker])
        T_movi_movj = m.invT(T_W_movi) @ T_W_movj
        method_label = "moving_relay_gt_motion_oracle"
    elif method == "COLMAP_motion":
        Tcw_i = colmap_poses[root_frame]
        Tcw_j = colmap_poses[target_frame]
        T_movi_movj = Tcw_i @ m.invT(Tcw_j)
        T_movi_movj[:3, 3] *= colmap_scale
        method_label = "moving_relay_colmap_motion_aruco_metric_scale"
    else:
        raise RuntimeError(f"Unsupported method: {method}")

    T_est = T_root_movi @ T_movi_movj @ m.invT(T_target_movj)

    T_W_root = m.T_W_optical_from_sdf_model(root_cam)
    T_W_target = m.T_W_optical_from_sdf_model(target_cam)
    T_gt = m.invT(T_W_root) @ T_W_target

    return {
        "name": f"{root_cam}_to_{target_cam}_{method}",
        "target_camera": target_cam,
        "method": method_label,
        "pair": row["pair"],
        "root_marker": root_marker,
        "target_marker": target_marker,
        "root_frame": root_frame,
        "target_frame": target_frame,
        "T_est": T_est,
        "T_gt": T_gt,
        "translation_error_m": m.trans_error(T_est, T_gt),
        "rotation_error_deg": m.rot_error_deg(T_est, T_gt),
        "notes": (
            "Moving-camera relay. GT is used only for evaluation. "
            "COLMAP variant uses no-GT ArUco metric scale."
        ),
    }


def strip_matrices(entry):
    e = dict(entry)
    T_est = e.pop("T_est")
    T_gt = e.pop("T_gt")
    e["estimated_transform_cam3_to_target"] = pose_payload(T_est)
    e["ground_truth_transform_cam3_to_target"] = pose_payload(T_gt)
    return e


def main():
    m = load_chain_module()

    entries = []

    # Direct baseline
    entries.append(compute_direct_cam3_to_cam1(m, marker_id=1))

    # Final no-GT COLMAP relay estimates, selected here by best evaluation error for summary/reporting.
    # In a strict deployment setting, selection should be made by quality metrics only.
    entries.append(compute_relay_from_row(m, best_relay_row("cam3_to_cam0", "COLMAP_motion")))
    entries.append(compute_relay_from_row(m, best_relay_row("cam3_to_cam5", "COLMAP_motion")))

    # Oracle baselines for comparison
    entries.append(compute_relay_from_row(m, best_relay_row("cam3_to_cam0", "GT_motion")))
    entries.append(compute_relay_from_row(m, best_relay_row("cam3_to_cam5", "GT_motion")))

    json_entries = [strip_matrices(e) for e in entries]

    out_json = OUT / "final_extrinsics_cam3_reference.json"
    out_json.write_text(json.dumps({
        "reference_camera": "cam_edge_3",
        "important_note": (
            "Ground truth is used only for evaluation. "
            "The COLMAP relay variants use ArUco metric scale estimated from known 0.17 m markers."
        ),
        "extrinsics": json_entries,
    }, indent=2) + "\n")

    out_csv = OUT / "final_extrinsics_summary.csv"
    fields = [
        "name", "target_camera", "method",
        "pair", "root_marker", "target_marker", "root_frame", "target_frame",
        "translation_error_cm", "rotation_error_deg",
        "est_x_m", "est_y_m", "est_z_m",
        "gt_x_m", "gt_y_m", "gt_z_m",
        "notes",
    ]

    with out_csv.open("w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=fields)
        w.writeheader()

        for e in entries:
            T_est = e["T_est"]
            T_gt = e["T_gt"]

            w.writerow({
                "name": e.get("name", ""),
                "target_camera": e.get("target_camera", ""),
                "method": e.get("method", ""),
                "pair": e.get("pair", ""),
                "root_marker": e.get("root_marker", e.get("selected_marker", "")),
                "target_marker": e.get("target_marker", ""),
                "root_frame": e.get("root_frame", ""),
                "target_frame": e.get("target_frame", ""),
                "translation_error_cm": 100.0 * float(e["translation_error_m"]),
                "rotation_error_deg": float(e["rotation_error_deg"]),
                "est_x_m": float(T_est[0, 3]),
                "est_y_m": float(T_est[1, 3]),
                "est_z_m": float(T_est[2, 3]),
                "gt_x_m": float(T_gt[0, 3]),
                "gt_y_m": float(T_gt[1, 3]),
                "gt_z_m": float(T_gt[2, 3]),
                "notes": e.get("notes", ""),
            })

    readme = OUT / "README.txt"
    readme.write_text(f"""Final extrinsics summary, cam_edge_3 reference
================================================

Reference camera:
  cam_edge_3

This folder summarizes the current calibrated camera rig.

Files:
  final_extrinsics_summary.csv
    Human-readable summary table with selected estimates and errors.

  final_extrinsics_cam3_reference.json
    Full 4x4 transforms, translations, Euler angles and quaternions.

Included estimates:
  1. cam_edge_3 -> cam_edge_1
     Method: direct static ArUco
     Role: direct-static baseline with shared marker.

  2. cam_edge_3 -> cam_edge_0
     Method: moving-camera relay with COLMAP motion and ArUco metric scale.
     Role: real-life-near relay estimate.

  3. cam_edge_3 -> cam_edge_5
     Method: moving-camera relay with COLMAP motion and ArUco metric scale.
     Role: real-life-near relay estimate.

  4. GT_motion relay variants
     Role: oracle/sanity baselines only.
     Not real-life deployable.

Important:
  Ground truth is used only for evaluation errors.
  The COLMAP relay estimates use no-GT metric scale from known 0.17 m ArUco markers.

Generated from:
  {CHAIN_SCRIPT}
  {RELAY_CSV}
""")

    print("[OK] wrote:", out_csv)
    print("[OK] wrote:", out_json)
    print("[OK] wrote:", readme)
    print()
    print(readme.read_text())


if __name__ == "__main__":
    main()
