#!/usr/bin/env python3

import argparse
import csv
import json
import math
import xml.etree.ElementTree as ET
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
    T[:3, :3] = np.asarray(R, dtype=np.float64)
    T[:3, 3] = np.asarray(t, dtype=np.float64).reshape(3)
    return T


def inv_T(T):
    R = T[:3, :3]
    t = T[:3, 3]
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ t
    return out


def transform_error(T_est, T_ref):
    dT = inv_T(T_ref) @ T_est
    trans_err = float(np.linalg.norm(dT[:3, 3]))
    rot_arg = (np.trace(dT[:3, :3]) - 1.0) / 2.0
    rot_err = math.degrees(math.acos(clamp(rot_arg)))
    return trans_err, rot_err


def rotmat_to_quat(R):
    R = np.asarray(R, dtype=np.float64)
    tr = float(np.trace(R))

    if tr > 0.0:
        s = math.sqrt(tr + 1.0) * 2.0
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    else:
        if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = math.sqrt(max(0.0, 1.0 + R[0, 0] - R[1, 1] - R[2, 2])) * 2.0
            qw = (R[2, 1] - R[1, 2]) / s
            qx = 0.25 * s
            qy = (R[0, 1] + R[1, 0]) / s
            qz = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = math.sqrt(max(0.0, 1.0 + R[1, 1] - R[0, 0] - R[2, 2])) * 2.0
            qw = (R[0, 2] - R[2, 0]) / s
            qx = (R[0, 1] + R[1, 0]) / s
            qy = 0.25 * s
            qz = (R[1, 2] + R[2, 1]) / s
        else:
            s = math.sqrt(max(0.0, 1.0 + R[2, 2] - R[0, 0] - R[1, 1])) * 2.0
            qw = (R[1, 0] - R[0, 1]) / s
            qx = (R[0, 2] + R[2, 0]) / s
            qy = (R[1, 2] + R[2, 1]) / s
            qz = 0.25 * s

    q = np.array([qw, qx, qy, qz], dtype=np.float64)
    q /= max(1e-15, np.linalg.norm(q))
    return q


def quat_to_rotmat(q):
    q = np.asarray(q, dtype=np.float64)
    q /= max(1e-15, np.linalg.norm(q))
    qw, qx, qy, qz = q

    return np.array([
        [1 - 2*qy*qy - 2*qz*qz,     2*qx*qy - 2*qz*qw,     2*qx*qz + 2*qy*qw],
        [    2*qx*qy + 2*qz*qw, 1 - 2*qx*qx - 2*qz*qz,     2*qy*qz - 2*qx*qw],
        [    2*qx*qz - 2*qy*qw,     2*qy*qz + 2*qx*qw, 1 - 2*qx*qx - 2*qy*qy],
    ], dtype=np.float64)


def weighted_rotation_mean(rotations, weights):
    qs = [rotmat_to_quat(R) for R in rotations]
    ref = qs[0]
    A = np.zeros((4, 4), dtype=np.float64)

    for q, w in zip(qs, weights):
        if float(np.dot(q, ref)) < 0.0:
            q = -q
        A += float(w) * np.outer(q, q)

    vals, vecs = np.linalg.eigh(A)
    q_mean = vecs[:, int(np.argmax(vals))]
    if q_mean[0] < 0:
        q_mean = -q_mean
    return quat_to_rotmat(q_mean)


def weighted_transform_mean(transforms, weights):
    weights = np.asarray(weights, dtype=np.float64)
    weights = weights / max(1e-15, float(np.sum(weights)))

    t = np.zeros(3, dtype=np.float64)
    for T, w in zip(transforms, weights):
        t += float(w) * T[:3, 3]

    R = weighted_rotation_mean([T[:3, :3] for T in transforms], weights)

    return make_T(R, t)


def polygon_area(points):
    pts = np.asarray(points, dtype=np.float64)
    x = pts[:, 0]
    y = pts[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def safe_float(row, key, default=float("nan")):
    try:
        return float(row[key])
    except Exception:
        return default


def marker_area_px(row):
    keys = [
        ("corner0_u", "corner0_v"),
        ("corner1_u", "corner1_v"),
        ("corner2_u", "corner2_v"),
        ("corner3_u", "corner3_v"),
    ]

    if all(k1 in row and k2 in row for k1, k2 in keys):
        pts = [[safe_float(row, u), safe_float(row, v)] for u, v in keys]
        if all(math.isfinite(x) and math.isfinite(y) for x, y in pts):
            return polygon_area(pts)

    return float("nan")


def center_norm(row, width=1280.0, height=720.0):
    cu = safe_float(row, "center_u")
    cv = safe_float(row, "center_v")

    if not math.isfinite(cu) or not math.isfinite(cv):
        return float("nan")

    cx = width / 2.0
    cy = height / 2.0
    half_diag = math.sqrt(cx * cx + cy * cy)
    return math.sqrt((cu - cx) ** 2 + (cv - cy) ** 2) / half_diag


def detection_quality(row):
    dist = safe_float(row, "distance_m")
    area = marker_area_px(row)
    cn = center_norm(row)

    if not math.isfinite(dist) or dist <= 0.0:
        dist = 99.0
    if not math.isfinite(area) or area <= 0.0:
        area = 1.0
    if not math.isfinite(cn):
        cn = 1.0

    return area / ((dist * dist) * (1.0 + cn))


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
        cam_name = name[name.index("cam_edge_"):] if "cam_edge_" in name else name

        poses[cam_name] = make_T(
            rpy_to_rotmat(roll, pitch, yaw),
            np.array([x, y, z], dtype=np.float64),
        )

    return poses


def candidate_gt_conventions(T_W_link):
    R_opt_to_link = rpy_to_rotmat(0.0, -math.pi / 2.0, math.pi / 2.0)
    T_opt_to_link = make_T(R_opt_to_link, np.zeros(3))
    return {
        "raw_sdf_model_pose_as_camera_frame": T_W_link,
        "sdf_link_times_inverse_optical_to_link": T_W_link @ inv_T(T_opt_to_link),
        "sdf_link_times_optical_to_link": T_W_link @ T_opt_to_link,
    }


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

            rr = dict(r)
            rr["camera"] = r["camera"]
            rr["marker_id"] = int(r["marker_id"])
            rr["T_cam_marker"] = T_cam_marker
            rr["distance_m"] = float(r["distance_m"])
            rr["area_px"] = marker_area_px(r)
            rr["center_norm"] = center_norm(r)
            rr["quality"] = detection_quality(r)
            rows.append(rr)

    return rows


def best_detection_by_camera_marker(rows):
    best = {}
    for r in rows:
        key = (r["camera"], r["marker_id"])
        if key not in best or r["quality"] > best[key]["quality"]:
            best[key] = r
    return best


def se3_medoid(candidates):
    if not candidates:
        raise RuntimeError("No candidates")

    best_idx = None
    best_score = None

    for i, ci in enumerate(candidates):
        ds = []
        Ti = ci["T_est"]
        for j, cj in enumerate(candidates):
            if i == j:
                continue
            t, r = transform_error(Ti, cj["T_est"])
            ds.append(t + 0.02 * r)

        score = float(np.median(ds)) if ds else 0.0

        if best_score is None or score < best_score:
            best_score = score
            best_idx = i

    return best_idx, best_score


def robust_inliers_by_medoid(candidates, medoid_idx, min_t_thresh=0.30, min_r_thresh_deg=5.0):
    T_med = candidates[medoid_idx]["T_est"]

    tdev = []
    rdev = []

    for c in candidates:
        t, r = transform_error(c["T_est"], T_med)
        c["deviation_to_medoid_m"] = t
        c["deviation_to_medoid_cm"] = 100.0 * t
        c["deviation_to_medoid_rot_deg"] = r
        tdev.append(t)
        rdev.append(r)

    tdev = np.asarray(tdev, dtype=np.float64)
    rdev = np.asarray(rdev, dtype=np.float64)

    t_med = float(np.median(tdev))
    r_med = float(np.median(rdev))

    t_mad = 1.4826 * float(np.median(np.abs(tdev - t_med)))
    r_mad = 1.4826 * float(np.median(np.abs(rdev - r_med)))

    t_thresh = max(min_t_thresh, t_med + 3.0 * t_mad)
    r_thresh = max(min_r_thresh_deg, r_med + 3.0 * r_mad)

    inliers = []
    outliers = []

    for c in candidates:
        is_inlier = (
            c["deviation_to_medoid_m"] <= t_thresh
            and c["deviation_to_medoid_rot_deg"] <= r_thresh
        )
        c["inlier"] = bool(is_inlier)
        c["inlier_rule"] = (
            f"dev_t <= {t_thresh:.6f} m and dev_r <= {r_thresh:.6f} deg; "
            f"thresholds from median+3*MAD with floors {min_t_thresh}m/{min_r_thresh_deg}deg"
        )
        if is_inlier:
            inliers.append(c)
        else:
            outliers.append(c)

    stats = {
        "t_dev_median_m": t_med,
        "t_dev_mad_scaled_m": t_mad,
        "t_inlier_threshold_m": t_thresh,
        "r_dev_median_deg": r_med,
        "r_dev_mad_scaled_deg": r_mad,
        "r_inlier_threshold_deg": r_thresh,
        "num_inliers": len(inliers),
        "num_outliers": len(outliers),
    }

    return inliers, outliers, stats


def pose_fields(prefix, T):
    roll, pitch, yaw = rotmat_to_rpy(T[:3, :3])
    q = rotmat_to_quat(T[:3, :3])
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
        f"{prefix}_qw": q[0],
        f"{prefix}_qx": q[1],
        f"{prefix}_qy": q[2],
        f"{prefix}_qz": q[3],
    }


def serializable_row(row):
    out = {}
    for k, v in row.items():
        if k.startswith("T_"):
            continue
        if isinstance(v, (np.floating, float)):
            if math.isfinite(float(v)):
                out[k] = f"{float(v):.9f}"
            else:
                out[k] = ""
        elif isinstance(v, (np.integer, int)):
            out[k] = int(v)
        elif isinstance(v, bool):
            out[k] = str(v)
        else:
            out[k] = v
    return out


def write_csv(path, rows, fields=None):
    path.parent.mkdir(parents=True, exist_ok=True)

    serial = [serializable_row(r) for r in rows]

    if fields is None:
        fields = []
        for r in serial:
            for k in r.keys():
                if k not in fields:
                    fields.append(k)

    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in serial:
            w.writerow(r)

    print("[OK] wrote:", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--detections", default="results/bus_real_data/01_marker_direct_relay_multimarker_multichain/01_static_a4_marker_detection/detections.csv")
    ap.add_argument("--world-sdf", default="src/calib_lab/bus_real_data/worlds/bus_real_data_moving_camera.sdf")
    ap.add_argument("--out", default="results/bus_real_data/01_marker_direct_relay_multimarker_multichain/05_direct_static_cam3_cam1_multimarker")
    ap.add_argument("--root-camera", default=ROOT_CAM)
    ap.add_argument("--target-camera", default=TARGET_CAM)
    args = ap.parse_args()

    root_cam = args.root_camera
    target_cam = args.target_camera
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    detections = load_static_detections(args.detections)
    best = best_detection_by_camera_marker(detections)

    root_markers = set(m for cam, m in best.keys() if cam == root_cam)
    target_markers = set(m for cam, m in best.keys() if cam == target_cam)
    common_markers = sorted(root_markers & target_markers)

    if not common_markers:
        raise RuntimeError(f"No common markers between {root_cam} and {target_cam}")

    world_poses = parse_world_camera_poses(args.world_sdf)
    if root_cam not in world_poses or target_cam not in world_poses:
        raise RuntimeError(f"Missing GT camera pose. Found: {sorted(world_poses)}")

    candidates = []

    for marker_id in common_markers:
        root_det = best[(root_cam, marker_id)]
        target_det = best[(target_cam, marker_id)]

        T_root_marker = root_det["T_cam_marker"]
        T_target_marker = target_det["T_cam_marker"]

        # Same physical marker observed by both cameras:
        # T_root_target = T_root_marker * inv(T_target_marker)
        T_root_target = T_root_marker @ inv_T(T_target_marker)

        quality = math.sqrt(max(1e-12, root_det["quality"] * target_det["quality"]))

        c = {
            "marker_id": marker_id,
            "root_camera": root_cam,
            "target_camera": target_cam,
            "root_distance_m": root_det["distance_m"],
            "target_distance_m": target_det["distance_m"],
            "root_area_px": root_det["area_px"],
            "target_area_px": target_det["area_px"],
            "root_center_norm": root_det["center_norm"],
            "target_center_norm": target_det["center_norm"],
            "root_quality": root_det["quality"],
            "target_quality": target_det["quality"],
            "combined_quality": quality,
            "T_est": T_root_target,
        }
        c.update(pose_fields("estimated_target_in_root", T_root_target))
        candidates.append(c)

    # GT only for evaluation and convention choice.
    T_W_root_link = world_poses[root_cam]
    T_W_target_link = world_poses[target_cam]

    convention_rows = []
    for conv_name in candidate_gt_conventions(T_W_root_link):
        T_W_root = candidate_gt_conventions(T_W_root_link)[conv_name]
        T_W_target = candidate_gt_conventions(T_W_target_link)[conv_name]
        T_gt = inv_T(T_W_root) @ T_W_target

        tes, res = [], []
        for c in candidates:
            te, re = transform_error(c["T_est"], T_gt)
            tes.append(te)
            res.append(re)

        convention_rows.append({
            "convention": conv_name,
            "median_translation_error_m": float(np.median(tes)),
            "mean_translation_error_m": float(np.mean(tes)),
            "min_translation_error_m": float(np.min(tes)),
            "max_translation_error_m": float(np.max(tes)),
            "median_rotation_error_deg": float(np.median(res)),
            "mean_rotation_error_deg": float(np.mean(res)),
            "min_rotation_error_deg": float(np.min(res)),
            "max_rotation_error_deg": float(np.max(res)),
        })

    convention_rows = sorted(
        convention_rows,
        key=lambda r: r["median_translation_error_m"] + 0.02 * r["median_rotation_error_deg"],
    )

    best_conv = convention_rows[0]["convention"]
    T_W_root = candidate_gt_conventions(T_W_root_link)[best_conv]
    T_W_target = candidate_gt_conventions(T_W_target_link)[best_conv]
    T_gt_best = inv_T(T_W_root) @ T_W_target

    for c in candidates:
        te, re = transform_error(c["T_est"], T_gt_best)
        c["translation_error_m"] = te
        c["translation_error_cm"] = 100.0 * te
        c["rotation_error_deg"] = re
        c.update(pose_fields("gt_target_in_root", T_gt_best))

    medoid_idx, medoid_score = se3_medoid(candidates)
    inliers, outliers, inlier_stats = robust_inliers_by_medoid(candidates, medoid_idx)

    if not inliers:
        raise RuntimeError("No inliers after robust filtering")

    T_medoid = candidates[medoid_idx]["T_est"]
    med_te, med_re = transform_error(T_medoid, T_gt_best)

    weights = np.array([max(1e-12, c["combined_quality"]) for c in inliers], dtype=np.float64)
    T_weighted = weighted_transform_mean([c["T_est"] for c in inliers], weights)
    wt_te, wt_re = transform_error(T_weighted, T_gt_best)

    T_unweighted = weighted_transform_mean(
        [c["T_est"] for c in inliers],
        np.ones(len(inliers), dtype=np.float64),
    )
    uw_te, uw_re = transform_error(T_unweighted, T_gt_best)

    aggregate_rows = []

    aggregate_rows.append({
        "aggregate_type": "se3_medoid_no_gt_selection",
        "selected_marker_id": candidates[medoid_idx]["marker_id"],
        "num_candidates": len(candidates),
        "num_inliers": len(inliers),
        "num_outliers": len(outliers),
        "medoid_pairwise_score": medoid_score,
        "translation_error_m": med_te,
        "translation_error_cm": 100.0 * med_te,
        "rotation_error_deg": med_re,
        **pose_fields("estimated_target_in_root", T_medoid),
        **pose_fields("gt_target_in_root", T_gt_best),
    })

    aggregate_rows.append({
        "aggregate_type": "weighted_mean_of_mad_inliers_no_gt_selection",
        "selected_marker_id": "",
        "num_candidates": len(candidates),
        "num_inliers": len(inliers),
        "num_outliers": len(outliers),
        "medoid_pairwise_score": medoid_score,
        "translation_error_m": wt_te,
        "translation_error_cm": 100.0 * wt_te,
        "rotation_error_deg": wt_re,
        **pose_fields("estimated_target_in_root", T_weighted),
        **pose_fields("gt_target_in_root", T_gt_best),
    })

    aggregate_rows.append({
        "aggregate_type": "unweighted_mean_of_mad_inliers_no_gt_selection",
        "selected_marker_id": "",
        "num_candidates": len(candidates),
        "num_inliers": len(inliers),
        "num_outliers": len(outliers),
        "medoid_pairwise_score": medoid_score,
        "translation_error_m": uw_te,
        "translation_error_cm": 100.0 * uw_te,
        "rotation_error_deg": uw_re,
        **pose_fields("estimated_target_in_root", T_unweighted),
        **pose_fields("gt_target_in_root", T_gt_best),
    })

    candidates_sorted = sorted(candidates, key=lambda c: c["translation_error_m"] + 0.02 * c["rotation_error_deg"])
    inliers_sorted = sorted(inliers, key=lambda c: c["combined_quality"], reverse=True)
    outliers_sorted = sorted(outliers, key=lambda c: c["deviation_to_medoid_m"], reverse=True)

    write_csv(out_dir / "01_gt_frame_convention_test.csv", convention_rows)
    write_csv(out_dir / "02_all_common_marker_candidates.csv", candidates_sorted)
    write_csv(out_dir / "03_mad_inlier_candidates.csv", inliers_sorted)
    write_csv(out_dir / "04_mad_outlier_candidates.csv", outliers_sorted)
    write_csv(out_dir / "05_multimarker_aggregate_estimates.csv", aggregate_rows)
    write_csv(out_dir / "06_inlier_filter_stats.csv", [inlier_stats])

    result_json = {
        "root_camera": root_cam,
        "target_camera": target_cam,
        "common_markers": common_markers,
        "selected_gt_convention_for_evaluation_only": best_conv,
        "selection_policy": {
            "uses_gt_for_selection": False,
            "candidate_source": "shared ArUco detections between root and target static cameras",
            "inlier_rule": "SE(3) medoid followed by median+3*MAD filtering on translation and rotation deviation",
            "final_estimates": [
                "se3_medoid_no_gt_selection",
                "weighted_mean_of_mad_inliers_no_gt_selection",
                "unweighted_mean_of_mad_inliers_no_gt_selection",
            ],
        },
        "inlier_stats": inlier_stats,
        "aggregate_errors": [
            {
                "aggregate_type": r["aggregate_type"],
                "translation_error_m": r["translation_error_m"],
                "rotation_error_deg": r["rotation_error_deg"],
                "num_candidates": r["num_candidates"],
                "num_inliers": r["num_inliers"],
                "num_outliers": r["num_outliers"],
            }
            for r in aggregate_rows
        ],
    }
    (out_dir / "multimarker_direct_result.json").write_text(json.dumps(result_json, indent=2))
    print("[OK] wrote:", out_dir / "multimarker_direct_result.json")

    best_individual = candidates_sorted[0]
    weighted_row = [r for r in aggregate_rows if r["aggregate_type"] == "weighted_mean_of_mad_inliers_no_gt_selection"][0]
    medoid_row = [r for r in aggregate_rows if r["aggregate_type"] == "se3_medoid_no_gt_selection"][0]

    summary = []
    summary.append(f"Direct static-to-static multi-marker evaluation: {root_cam} -> {target_cam}")
    summary.append("=" * 78)
    summary.append("")
    summary.append(f"Common markers: {common_markers}")
    summary.append(f"Number of candidates: {len(candidates)}")
    summary.append(f"Number of inliers: {len(inliers)}")
    summary.append(f"Number of outliers: {len(outliers)}")
    summary.append("")
    summary.append("No-GT selection policy:")
    summary.append("- Candidates are generated from shared marker PnP observations.")
    summary.append("- Medoid and MAD inlier filtering use only candidate consistency, not GT.")
    summary.append("- GT is used only for final evaluation.")
    summary.append("")
    summary.append(f"Selected GT convention for evaluation only: {best_conv}")
    summary.append("")
    summary.append("Best individual marker by GT evaluation only:")
    summary.append(f"- marker_id: {best_individual['marker_id']}")
    summary.append(f"- translation_error: {best_individual['translation_error_m']:.6f} m = {best_individual['translation_error_cm']:.3f} cm")
    summary.append(f"- rotation_error: {best_individual['rotation_error_deg']:.6f} deg")
    summary.append("")
    summary.append("SE(3) medoid aggregate, no GT selection:")
    summary.append(f"- medoid marker_id: {medoid_row['selected_marker_id']}")
    summary.append(f"- translation_error: {medoid_row['translation_error_m']:.6f} m = {medoid_row['translation_error_cm']:.3f} cm")
    summary.append(f"- rotation_error: {medoid_row['rotation_error_deg']:.6f} deg")
    summary.append("")
    summary.append("Weighted mean of MAD inliers, no GT selection:")
    summary.append(f"- translation_error: {weighted_row['translation_error_m']:.6f} m = {weighted_row['translation_error_cm']:.3f} cm")
    summary.append(f"- rotation_error: {weighted_row['rotation_error_deg']:.6f} deg")
    summary.append("")
    summary.append("Inlier filter:")
    summary.append(f"- t threshold: {inlier_stats['t_inlier_threshold_m']:.6f} m")
    summary.append(f"- r threshold: {inlier_stats['r_inlier_threshold_deg']:.6f} deg")
    summary.append("")
    summary.append("Outliers:")
    if outliers_sorted:
        for c in outliers_sorted:
            summary.append(
                f"- marker {c['marker_id']}: "
                f"dev_t={c['deviation_to_medoid_cm']:.2f} cm, "
                f"dev_r={c['deviation_to_medoid_rot_deg']:.2f} deg"
            )
    else:
        summary.append("- none")

    (out_dir / "summary.txt").write_text("\n".join(summary) + "\n")
    print("[OK] wrote:", out_dir / "summary.txt")
    print()
    print("\n".join(summary))


if __name__ == "__main__":
    main()
