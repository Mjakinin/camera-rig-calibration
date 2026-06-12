#!/usr/bin/env python3
import csv
import json
import math
from pathlib import Path

import numpy as np


IN_JSON = Path("results/bus_real_data/07_final_extrinsics_cam3_reference/final_extrinsics_cam3_reference.json")
OUT_DIR = Path("results/bus_real_data/07_final_extrinsics_cam3_reference")

OUT_OVERVIEW_TXT = OUT_DIR / "FINAL_CAMERA_RIG_OVERVIEW.txt"
OUT_OVERVIEW_MD = OUT_DIR / "FINAL_CAMERA_RIG_OVERVIEW.md"
OUT_PAIRWISE_CSV = OUT_DIR / "pairwise_extrinsics_summary.csv"


def mat(entry, key):
    return np.array(entry[key]["matrix_4x4"], dtype=float)


def invT(T):
    R = T[:3, :3]
    t = T[:3, 3]
    Ti = np.eye(4)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti


def rot_error_deg(T_est, T_gt):
    R = T_est[:3, :3].T @ T_gt[:3, :3]
    x = (np.trace(R) - 1.0) / 2.0
    x = max(-1.0, min(1.0, float(x)))
    return math.degrees(math.acos(x))


def trans_error_m(T_est, T_gt):
    return float(np.linalg.norm(T_est[:3, 3] - T_gt[:3, 3]))


def rpy_deg_from_R(R):
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


def row_from_T(T):
    t = T[:3, 3]
    r, p, y = rpy_deg_from_R(T[:3, :3])
    return {
        "x_m": float(t[0]),
        "y_m": float(t[1]),
        "z_m": float(t[2]),
        "roll_deg": float(r),
        "pitch_deg": float(p),
        "yaw_deg": float(y),
    }


def fmt_xyz(d):
    return f"({d['x_m']:+.3f}, {d['y_m']:+.3f}, {d['z_m']:+.3f}) m"


def fmt_rpy(d):
    return f"({d['roll_deg']:+.2f}, {d['pitch_deg']:+.2f}, {d['yaw_deg']:+.2f}) deg"


def short_method(m):
    if m == "direct_static_aruco":
        return "Direct ArUco"
    if m == "moving_relay_colmap_motion_aruco_metric_scale":
        return "Relay COLMAP + ArUco scale"
    if m == "moving_relay_gt_motion_oracle":
        return "Relay GT-motion oracle"
    return m


def target_from_entry(e):
    return e["target_camera"]


def is_final_estimate(e):
    return e["method"] in {
        "direct_static_aruco",
        "moving_relay_colmap_motion_aruco_metric_scale",
    }


def main():
    data = json.loads(IN_JSON.read_text())
    entries = data["extrinsics"]

    final_entries = [e for e in entries if is_final_estimate(e)]
    oracle_entries = [e for e in entries if e["method"] == "moving_relay_gt_motion_oracle"]

    lines = []
    lines.append("FINAL CAMERA RIG EXTRINSICS")
    lines.append("===========================")
    lines.append("")
    lines.append("Reference camera: cam_edge_3")
    lines.append("")
    lines.append("Ground truth is used only for error evaluation.")
    lines.append("COLMAP relay uses no-GT metric scale from known 0.17 m ArUco markers.")
    lines.append("")
    lines.append("FINAL ESTIMATES")
    lines.append("---------------")

    for e in final_entries:
        T_est = mat(e, "estimated_transform_cam3_to_target")
        T_gt = mat(e, "ground_truth_transform_cam3_to_target")

        est = row_from_T(T_est)
        gt = row_from_T(T_gt)

        target = target_from_entry(e)
        method = short_method(e["method"])

        marker_info = ""
        if "selected_marker" in e:
            marker_info = f"marker={e['selected_marker']}"
        elif "root_marker" in e and "target_marker" in e:
            marker_info = (
                f"root_marker={e['root_marker']}@frame={e.get('root_frame', '')}, "
                f"target_marker={e['target_marker']}@frame={e.get('target_frame', '')}"
            )

        lines.append("")
        lines.append(f"cam_edge_3 -> {target}")
        lines.append(f"  method:       {method}")
        lines.append(f"  chain:        {marker_info}")
        lines.append(f"  estimated t:  {fmt_xyz(est)}")
        lines.append(f"  estimated r:  {fmt_rpy(est)}")
        lines.append(f"  GT t:         {fmt_xyz(gt)}")
        lines.append(f"  GT r:         {fmt_rpy(gt)}")
        lines.append(f"  error:        {100.0 * trans_error_m(T_est, T_gt):.2f} cm, {rot_error_deg(T_est, T_gt):.2f} deg")

    lines.append("")
    lines.append("ORACLE / SANITY BASELINES")
    lines.append("-------------------------")
    lines.append("These use GT moving-camera motion and are not real-life deployable.")

    for e in oracle_entries:
        T_est = mat(e, "estimated_transform_cam3_to_target")
        T_gt = mat(e, "ground_truth_transform_cam3_to_target")
        target = target_from_entry(e)

        marker_info = (
            f"root_marker={e.get('root_marker', '')}@frame={e.get('root_frame', '')}, "
            f"target_marker={e.get('target_marker', '')}@frame={e.get('target_frame', '')}"
        )

        lines.append("")
        lines.append(f"cam_edge_3 -> {target}")
        lines.append(f"  method:       {short_method(e['method'])}")
        lines.append(f"  chain:        {marker_info}")
        lines.append(f"  error:        {100.0 * trans_error_m(T_est, T_gt):.2f} cm, {rot_error_deg(T_est, T_gt):.2f} deg")

    # Pairwise transforms from final estimates only.
    T_est_ref = {"cam_edge_3": np.eye(4)}
    T_gt_ref = {"cam_edge_3": np.eye(4)}

    for e in final_entries:
        cam = target_from_entry(e)
        T_est_ref[cam] = mat(e, "estimated_transform_cam3_to_target")
        T_gt_ref[cam] = mat(e, "ground_truth_transform_cam3_to_target")

    cams = sorted(T_est_ref.keys())

    pair_fields = [
        "from_camera", "to_camera",
        "estimated_x_m", "estimated_y_m", "estimated_z_m",
        "estimated_roll_deg", "estimated_pitch_deg", "estimated_yaw_deg",
        "gt_x_m", "gt_y_m", "gt_z_m",
        "gt_roll_deg", "gt_pitch_deg", "gt_yaw_deg",
        "translation_error_cm", "rotation_error_deg",
    ]

    with OUT_PAIRWISE_CSV.open("w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=pair_fields)
        w.writeheader()

        for a in cams:
            for b in cams:
                if a == b:
                    continue

                T_est_ab = invT(T_est_ref[a]) @ T_est_ref[b]
                T_gt_ab = invT(T_gt_ref[a]) @ T_gt_ref[b]

                est = row_from_T(T_est_ab)
                gt = row_from_T(T_gt_ab)

                w.writerow({
                    "from_camera": a,
                    "to_camera": b,
                    "estimated_x_m": est["x_m"],
                    "estimated_y_m": est["y_m"],
                    "estimated_z_m": est["z_m"],
                    "estimated_roll_deg": est["roll_deg"],
                    "estimated_pitch_deg": est["pitch_deg"],
                    "estimated_yaw_deg": est["yaw_deg"],
                    "gt_x_m": gt["x_m"],
                    "gt_y_m": gt["y_m"],
                    "gt_z_m": gt["z_m"],
                    "gt_roll_deg": gt["roll_deg"],
                    "gt_pitch_deg": gt["pitch_deg"],
                    "gt_yaw_deg": gt["yaw_deg"],
                    "translation_error_cm": 100.0 * trans_error_m(T_est_ab, T_gt_ab),
                    "rotation_error_deg": rot_error_deg(T_est_ab, T_gt_ab),
                })

    lines.append("")
    lines.append("ALL CAMERA-TO-CAMERA PAIRS")
    lines.append("--------------------------")
    lines.append(f"Full pairwise camera-to-camera transforms are written to:")
    lines.append(f"  {OUT_PAIRWISE_CSV}")
    lines.append("")
    lines.append("Files:")
    lines.append(f"  {OUT_OVERVIEW_TXT}")
    lines.append(f"  {OUT_OVERVIEW_MD}")
    lines.append(f"  {OUT_PAIRWISE_CSV}")

    txt = "\n".join(lines) + "\n"
    OUT_OVERVIEW_TXT.write_text(txt)
    OUT_OVERVIEW_MD.write_text("```text\n" + txt + "```\n")

    print(txt)


if __name__ == "__main__":
    main()
