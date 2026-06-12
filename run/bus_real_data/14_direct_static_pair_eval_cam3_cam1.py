#!/usr/bin/env python3

import argparse
import csv
import math
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT_CAM = "cam_edge_3"
TARGET_CAM = "cam_edge_1"


def clamp(x, lo=-1.0, hi=1.0):
    return max(lo, min(hi, x))


def rvec_to_rotmat(rvec):
    rvec = np.asarray(rvec, dtype=np.float64).reshape(3)
    theta = float(np.linalg.norm(rvec))

    if theta < 1e-12:
        return np.eye(3)

    k = rvec / theta
    K = np.array([
        [0.0, -k[2], k[1]],
        [k[2], 0.0, -k[0]],
        [-k[1], k[0], 0.0],
    ])

    return np.eye(3) + math.sin(theta) * K + (1.0 - math.cos(theta)) * (K @ K)


def rpy_to_rotmat(roll, pitch, yaw):
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)

    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float64)
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float64)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float64)

    return Rz @ Ry @ Rx


def rotmat_to_rpy(R):
    pitch = math.atan2(-R[2, 0], math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2))
    roll = math.atan2(R[2, 1], R[2, 2])
    yaw = math.atan2(R[1, 0], R[0, 0])
    return roll, pitch, yaw


def make_T(R, t):
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(t, dtype=np.float64).reshape(3)
    return T


def inv_T(T):
    R = T[:3, :3]
    t = T[:3, 3]

    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ t
    return out


def transform_error(T_est, T_gt):
    dT = inv_T(T_gt) @ T_est

    trans_err = float(np.linalg.norm(dT[:3, 3]))
    rot_arg = (np.trace(dT[:3, :3]) - 1.0) / 2.0
    rot_err = math.degrees(math.acos(clamp(rot_arg)))

    return trans_err, rot_err


def parse_world_camera_poses(world_sdf):
    p = Path(world_sdf)
    if not p.exists():
        raise RuntimeError(f"World SDF not found: {p}")

    tree = ET.parse(p)
    root = tree.getroot()

    poses = {}

    for model in root.iter("model"):
        name = model.attrib.get("name", "")

        if "cam_edge_" not in name:
            continue

        pose_el = model.find("pose")
        if pose_el is None or not pose_el.text:
            continue

        vals = [float(x) for x in pose_el.text.split()]
        if len(vals) < 6:
            continue

        x, y, z, roll, pitch, yaw = vals[:6]

        cam_name = name
        if "cam_edge_" in name:
            cam_name = name[name.index("cam_edge_"):]

        poses[cam_name] = make_T(
            rpy_to_rotmat(roll, pitch, yaw),
            np.array([x, y, z], dtype=np.float64),
        )

    return poses


def load_static_detections(path):
    p = Path(path)
    if not p.exists():
        raise RuntimeError(f"Missing detections CSV: {p}")

    rows = []

    with p.open() as f:
        reader = csv.DictReader(f)

        for r in reader:
            if str(r.get("pnp_success", "True")).lower() in ["false", "0", "no"]:
                continue

            cam = r["camera"]
            marker_id = int(r["marker_id"])

            rvec = np.array([
                float(r["rvec_x"]),
                float(r["rvec_y"]),
                float(r["rvec_z"]),
            ], dtype=np.float64)

            tvec = np.array([
                float(r["tvec_x_m"]),
                float(r["tvec_y_m"]),
                float(r["tvec_z_m"]),
            ], dtype=np.float64)

            T_cam_marker = make_T(rvec_to_rotmat(rvec), tvec)

            rows.append({
                "camera": cam,
                "marker_id": marker_id,
                "T_cam_marker": T_cam_marker,
                "center_u": float(r["center_u"]),
                "center_v": float(r["center_v"]),
                "distance_m": float(r["distance_m"]),
            })

    return rows


def build_best_detection_by_camera_marker(rows):
    # Static image has one observation per camera/marker normally.
    # If there are duplicates, choose closer/larger-looking observation heuristically by distance.
    best = {}

    for r in rows:
        key = (r["camera"], r["marker_id"])

        if key not in best:
            best[key] = r
        else:
            if r["distance_m"] < best[key]["distance_m"]:
                best[key] = r

    return best


def candidate_gt_conventions(T_W_link):
    # This is the same correction we used in the world generator.
    R_opt_to_link = rpy_to_rotmat(0.0, -math.pi / 2.0, math.pi / 2.0)
    T_opt_to_link = make_T(R_opt_to_link, np.zeros(3))

    return {
        "raw_sdf_model_pose_as_camera_frame": T_W_link,
        "sdf_link_times_inverse_optical_to_link": T_W_link @ inv_T(T_opt_to_link),
        "sdf_link_times_optical_to_link": T_W_link @ T_opt_to_link,
    }


def median_transform_medoid(estimates):
    # Robust simple aggregate: choose the estimate with lowest median pairwise SE(3) distance.
    if not estimates:
        return None, None

    best_idx = None
    best_score = None

    for i, Ti in enumerate(estimates):
        scores = []

        for j, Tj in enumerate(estimates):
            if i == j:
                continue

            t, r = transform_error(Ti, Tj)
            scores.append(t + 0.02 * r)

        score = float(np.median(scores)) if scores else 0.0

        if best_score is None or score < best_score:
            best_score = score
            best_idx = i

    return estimates[best_idx], best_idx


def pose_fields(prefix, T):
    roll, pitch, yaw = rotmat_to_rpy(T[:3, :3])
    return {
        f"{prefix}_x": T[0, 3],
        f"{prefix}_y": T[1, 3],
        f"{prefix}_z": T[2, 3],
        f"{prefix}_roll": roll,
        f"{prefix}_pitch": pitch,
        f"{prefix}_yaw": yaw,
        f"{prefix}_roll_deg": math.degrees(roll),
        f"{prefix}_pitch_deg": math.degrees(pitch),
        f"{prefix}_yaw_deg": math.degrees(yaw),
    }


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()

        for r in rows:
            rr = {}
            for k, v in r.items():
                if isinstance(v, float):
                    rr[k] = f"{v:.9f}"
                else:
                    rr[k] = v
            w.writerow(rr)

    print("[OK] wrote:", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--detections", default="results/bus_real_data/02_a4_marker_detection_static/detections.csv")
    ap.add_argument("--world-sdf", default="src/calib_lab/bus_real_data/worlds/bus_real_data_a4_markers.sdf")
    ap.add_argument("--out", default="results/bus_real_data/05_direct_static_pair_cam3_cam1")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    detections = load_static_detections(args.detections)
    best = build_best_detection_by_camera_marker(detections)

    world_poses = parse_world_camera_poses(args.world_sdf)

    if ROOT_CAM not in world_poses:
        raise RuntimeError(f"Missing {ROOT_CAM} in world SDF parsed poses. Found: {sorted(world_poses.keys())}")
    if TARGET_CAM not in world_poses:
        raise RuntimeError(f"Missing {TARGET_CAM} in world SDF parsed poses. Found: {sorted(world_poses.keys())}")

    root_markers = set(m for cam, m in best.keys() if cam == ROOT_CAM)
    target_markers = set(m for cam, m in best.keys() if cam == TARGET_CAM)
    common_markers = sorted(root_markers & target_markers)

    if not common_markers:
        raise RuntimeError(f"No common markers between {ROOT_CAM} and {TARGET_CAM}")

    print("[INFO] common markers:", common_markers)

    T_W_root_link = world_poses[ROOT_CAM]
    T_W_target_link = world_poses[TARGET_CAM]

    # Build estimates from shared markers.
    estimates = []

    for marker_id in common_markers:
        T_root_marker = best[(ROOT_CAM, marker_id)]["T_cam_marker"]
        T_target_marker = best[(TARGET_CAM, marker_id)]["T_cam_marker"]

        # OpenCV PnP gives T_camera_marker.
        # Same marker gives T_root_target = T_root_marker * inv(T_target_marker).
        T_root_target_est = T_root_marker @ inv_T(T_target_marker)

        estimates.append({
            "marker_id": marker_id,
            "T_root_target_est": T_root_target_est,
            "root_distance_m": best[(ROOT_CAM, marker_id)]["distance_m"],
            "target_distance_m": best[(TARGET_CAM, marker_id)]["distance_m"],
        })

    # Test GT camera frame conventions.
    convention_summaries = []

    for conv_name in candidate_gt_conventions(T_W_root_link):
        T_W_root = candidate_gt_conventions(T_W_root_link)[conv_name]
        T_W_target = candidate_gt_conventions(T_W_target_link)[conv_name]
        T_gt_root_target = inv_T(T_W_root) @ T_W_target

        trans_errors = []
        rot_errors = []

        for e in estimates:
            te, re = transform_error(e["T_root_target_est"], T_gt_root_target)
            trans_errors.append(te)
            rot_errors.append(re)

        convention_summaries.append({
            "convention": conv_name,
            "median_translation_error_m": float(np.median(trans_errors)),
            "mean_translation_error_m": float(np.mean(trans_errors)),
            "min_translation_error_m": float(np.min(trans_errors)),
            "max_translation_error_m": float(np.max(trans_errors)),
            "median_rotation_error_deg": float(np.median(rot_errors)),
            "mean_rotation_error_deg": float(np.mean(rot_errors)),
            "min_rotation_error_deg": float(np.min(rot_errors)),
            "max_rotation_error_deg": float(np.max(rot_errors)),
        })

    convention_summaries = sorted(
        convention_summaries,
        key=lambda r: r["median_translation_error_m"] + 0.02 * r["median_rotation_error_deg"],
    )

    best_convention = convention_summaries[0]["convention"]

    T_W_root_best = candidate_gt_conventions(T_W_root_link)[best_convention]
    T_W_target_best = candidate_gt_conventions(T_W_target_link)[best_convention]
    T_gt_root_target_best = inv_T(T_W_root_best) @ T_W_target_best

    rows = []

    for e in estimates:
        T_est = e["T_root_target_est"]
        te, re = transform_error(T_est, T_gt_root_target_best)

        row = {
            "marker_id": e["marker_id"],
            "root_camera": ROOT_CAM,
            "target_camera": TARGET_CAM,
            "root_distance_m": e["root_distance_m"],
            "target_distance_m": e["target_distance_m"],
            "translation_error_m": te,
            "translation_error_cm": te * 100.0,
            "rotation_error_deg": re,
        }

        row.update(pose_fields("estimated_target_in_root", T_est))
        row.update(pose_fields("gt_target_in_root", T_gt_root_target_best))

        rows.append(row)

    rows_by_best = sorted(rows, key=lambda r: r["translation_error_m"] + 0.02 * r["rotation_error_deg"])
    rows_by_worst = sorted(rows, key=lambda r: r["translation_error_m"] + 0.02 * r["rotation_error_deg"], reverse=True)

    fields = [
        "marker_id",
        "root_camera",
        "target_camera",
        "root_distance_m",
        "target_distance_m",
        "translation_error_m",
        "translation_error_cm",
        "rotation_error_deg",
        "estimated_target_in_root_x",
        "gt_target_in_root_x",
        "estimated_target_in_root_y",
        "gt_target_in_root_y",
        "estimated_target_in_root_z",
        "gt_target_in_root_z",
        "estimated_target_in_root_roll_deg",
        "gt_target_in_root_roll_deg",
        "estimated_target_in_root_pitch_deg",
        "gt_target_in_root_pitch_deg",
        "estimated_target_in_root_yaw_deg",
        "gt_target_in_root_yaw_deg",
        "estimated_target_in_root_roll",
        "gt_target_in_root_roll",
        "estimated_target_in_root_pitch",
        "gt_target_in_root_pitch",
        "estimated_target_in_root_yaw",
        "gt_target_in_root_yaw",
    ]

    write_csv(out_dir / "01_gt_frame_convention_test.csv", convention_summaries, list(convention_summaries[0].keys()))
    write_csv(out_dir / "02_all_shared_marker_estimates_best_convention.csv", rows_by_best, fields)

    # Aggregate medoid estimate from all markers.
    medoid_T, medoid_idx = median_transform_medoid([e["T_root_target_est"] for e in estimates])
    medoid_marker = estimates[medoid_idx]["marker_id"]
    med_te, med_re = transform_error(medoid_T, T_gt_root_target_best)

    medoid_row = {
        "aggregate_type": "medoid_lowest_pairwise_disagreement",
        "chosen_marker_id": medoid_marker,
        "translation_error_m": med_te,
        "translation_error_cm": med_te * 100.0,
        "rotation_error_deg": med_re,
    }
    medoid_row.update(pose_fields("estimated_target_in_root", medoid_T))
    medoid_row.update(pose_fields("gt_target_in_root", T_gt_root_target_best))

    write_csv(out_dir / "03_aggregate_medoid_estimate.csv", [medoid_row], list(medoid_row.keys()))

    best_row = rows_by_best[0]
    worst_row = rows_by_worst[0]

    summary = []
    summary.append(f"Direct static-to-static evaluation: {ROOT_CAM} -> {TARGET_CAM}")
    summary.append("=" * 68)
    summary.append("")
    summary.append(f"Common markers: {common_markers}")
    summary.append(f"Number of marker estimates: {len(rows)}")
    summary.append("")
    summary.append("GT frame convention test:")
    for c in convention_summaries:
        summary.append(
            f"- {c['convention']}: "
            f"median_t={c['median_translation_error_m']:.4f} m, "
            f"median_r={c['median_rotation_error_deg']:.3f} deg, "
            f"mean_t={c['mean_translation_error_m']:.4f} m, "
            f"mean_r={c['mean_rotation_error_deg']:.3f} deg"
        )

    summary.append("")
    summary.append(f"Selected best GT convention: {best_convention}")
    summary.append("")
    summary.append("Best individual shared-marker estimate:")
    summary.append(f"- marker_id: {best_row['marker_id']}")
    summary.append(f"- translation_error: {best_row['translation_error_m']:.6f} m = {best_row['translation_error_cm']:.3f} cm")
    summary.append(f"- rotation_error: {best_row['rotation_error_deg']:.6f} deg")
    summary.append(
        f"- estimated target position in {ROOT_CAM}: "
        f"x={best_row['estimated_target_in_root_x']:.4f}, "
        f"y={best_row['estimated_target_in_root_y']:.4f}, "
        f"z={best_row['estimated_target_in_root_z']:.4f}"
    )
    summary.append(
        f"- GT target position in {ROOT_CAM}: "
        f"x={best_row['gt_target_in_root_x']:.4f}, "
        f"y={best_row['gt_target_in_root_y']:.4f}, "
        f"z={best_row['gt_target_in_root_z']:.4f}"
    )
    summary.append("")
    summary.append("Worst individual shared-marker estimate:")
    summary.append(f"- marker_id: {worst_row['marker_id']}")
    summary.append(f"- translation_error: {worst_row['translation_error_m']:.6f} m = {worst_row['translation_error_cm']:.3f} cm")
    summary.append(f"- rotation_error: {worst_row['rotation_error_deg']:.6f} deg")
    summary.append(
        f"- estimated target position in {ROOT_CAM}: "
        f"x={worst_row['estimated_target_in_root_x']:.4f}, "
        f"y={worst_row['estimated_target_in_root_y']:.4f}, "
        f"z={worst_row['estimated_target_in_root_z']:.4f}"
    )
    summary.append(
        f"- GT target position in {ROOT_CAM}: "
        f"x={worst_row['gt_target_in_root_x']:.4f}, "
        f"y={worst_row['gt_target_in_root_y']:.4f}, "
        f"z={worst_row['gt_target_in_root_z']:.4f}"
    )
    summary.append("")
    summary.append("Aggregate medoid estimate over all shared markers:")
    summary.append(f"- chosen medoid marker_id: {medoid_marker}")
    summary.append(f"- translation_error: {med_te:.6f} m = {med_te * 100.0:.3f} cm")
    summary.append(f"- rotation_error: {med_re:.6f} deg")
    summary.append(
        f"- estimated target position in {ROOT_CAM}: "
        f"x={medoid_row['estimated_target_in_root_x']:.4f}, "
        f"y={medoid_row['estimated_target_in_root_y']:.4f}, "
        f"z={medoid_row['estimated_target_in_root_z']:.4f}"
    )
    summary.append(
        f"- GT target position in {ROOT_CAM}: "
        f"x={medoid_row['gt_target_in_root_x']:.4f}, "
        f"y={medoid_row['gt_target_in_root_y']:.4f}, "
        f"z={medoid_row['gt_target_in_root_z']:.4f}"
    )
    summary.append("")
    summary.append("Interpretation:")
    summary.append("- This direct static-static test is the sanity check for OpenCV PnP and GT frame convention.")
    summary.append("- If errors are still meter-scale after the best convention, the PnP/SDF frame convention or marker coordinate convention is still wrong.")
    summary.append("- If errors are cm-scale / few degrees, we can continue to GT moving relay chains.")

    summary_path = out_dir / "summary.txt"
    summary_path.write_text("\n".join(summary) + "\n")
    print("[OK] wrote:", summary_path)

    print("")
    print("\n".join(summary))


if __name__ == "__main__":
    main()
