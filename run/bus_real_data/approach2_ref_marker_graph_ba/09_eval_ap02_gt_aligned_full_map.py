#!/usr/bin/env python3

import csv
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np


WORLD_SDF = Path("src/calib_lab/bus_real_data/worlds/bus_real_data_moving_camera.sdf")

AP02_FINAL = Path("results/bus_real_data/02_ref_marker_graph_ba/08_final_results")
AP02_CAMERA_CSV = AP02_FINAL / "ap02_with_moving_static_camera_poses_ref_marker.csv"
AP02_MARKER_CSV = AP02_FINAL / "ap02_with_moving_marker_poses_ref_marker.csv"

OUT_CSV = AP02_FINAL / "AP02_FINAL_GT_ALIGNED_FULL_MAP_EVALUATION.csv"
OUT_TXT = AP02_FINAL / "AP02_FINAL_GT_ALIGNED_FULL_MAP_EVALUATION.txt"
OUT_META = AP02_FINAL / "AP02_FINAL_GT_ALIGNED_FULL_MAP_EVALUATION_metadata.json"

STATIC_CAMERAS = ["cam_edge_0", "cam_edge_1", "cam_edge_3", "cam_edge_5"]
REF_MARKER_ID = 14
REF_MARKER_ENTITY = "aruco_ref_floor_14"

# Gazebo marker model frame:
# - board plane is X/Z
# - board normal is Y
#
# OpenCV ArUco marker frame used by solvePnP:
# - x: marker right
# - y: marker down / vertical image direction
# - z: marker normal
#
# This rotation maps OpenCV marker coordinates into the SDF marker model frame.
R_MODEL_FROM_OPENCV_MARKER = np.array([
    [1.0, 0.0,  0.0],
    [0.0, 0.0, -1.0],
    [0.0, 1.0,  0.0],
], dtype=np.float64)


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_csv(path: Path):
    if not path.exists():
        raise RuntimeError(f"Missing CSV: {path}")
    with path.open(newline="") as fp:
        return list(csv.DictReader(fp))


def write_csv(path: Path, rows, fields):
    ensure_dir(path.parent)
    with path.open("w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def clamp(x, lo=-1.0, hi=1.0):
    return max(lo, min(hi, x))


def rpy_to_R(roll, pitch, yaw):
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)

    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float64)
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float64)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float64)
    return Rz @ Ry @ Rx


def rvec_to_R(rvec):
    rvec = np.asarray(rvec, dtype=np.float64).reshape(3)
    theta = float(np.linalg.norm(rvec))
    if theta < 1e-12:
        return np.eye(3)

    k = rvec / theta
    K = np.array([
        [0.0, -k[2], k[1]],
        [k[2], 0.0, -k[0]],
        [-k[1], k[0], 0.0],
    ], dtype=np.float64)
    return np.eye(3) + math.sin(theta) * K + (1.0 - math.cos(theta)) * (K @ K)


def R_to_rvec(R):
    R = np.asarray(R, dtype=np.float64)
    arg = clamp((float(np.trace(R)) - 1.0) / 2.0)
    theta = math.acos(arg)

    if theta < 1e-12:
        return np.zeros(3, dtype=np.float64)

    denom = 2.0 * math.sin(theta)
    axis = np.array([
        (R[2, 1] - R[1, 2]) / denom,
        (R[0, 2] - R[2, 0]) / denom,
        (R[1, 0] - R[0, 1]) / denom,
    ], dtype=np.float64)
    return axis * theta


def R_to_rpy_deg(R):
    pitch = math.atan2(-R[2, 0], math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2))
    roll = math.atan2(R[2, 1], R[2, 2])
    yaw = math.atan2(R[1, 0], R[0, 0])
    return [math.degrees(roll), math.degrees(pitch), math.degrees(yaw)]


def make_T(R, t):
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = np.asarray(R, dtype=np.float64)
    T[:3, 3] = np.asarray(t, dtype=np.float64).reshape(3)
    return T


def invT(T):
    R = T[:3, :3]
    t = T[:3, 3]
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ t
    return out


def trans_error_cm(T_est, T_gt):
    dT = invT(T_gt) @ T_est
    return 100.0 * float(np.linalg.norm(dT[:3, 3]))


def rot_error_deg(T_est, T_gt):
    dT = invT(T_gt) @ T_est
    arg = clamp((float(np.trace(dT[:3, :3])) - 1.0) / 2.0)
    return float(math.degrees(math.acos(arg)))


def parse_pose_text(text):
    vals = [float(x) for x in text.split()]
    if len(vals) < 6:
        raise RuntimeError(f"Invalid SDF pose: {text}")
    x, y, z, roll, pitch, yaw = vals[:6]
    return make_T(rpy_to_R(roll, pitch, yaw), np.array([x, y, z], dtype=np.float64)), vals[:6]


def parse_world_poses():
    if not WORLD_SDF.exists():
        raise RuntimeError(f"Missing world SDF: {WORLD_SDF}")

    tree = ET.parse(WORLD_SDF)
    root = tree.getroot()

    poses = {}

    for model in root.iter("model"):
        name = model.attrib.get("name", "").strip()
        pose_el = model.find("pose")
        if not name or pose_el is None or not pose_el.text:
            continue
        T, vals = parse_pose_text(pose_el.text)
        poses[name] = {"T_W_model": T, "pose_vals": vals, "source": "model"}

    for inc in root.iter("include"):
        name_el = inc.find("name")
        pose_el = inc.find("pose")
        if name_el is None or pose_el is None or not name_el.text or not pose_el.text:
            continue
        name = name_el.text.strip()
        T, vals = parse_pose_text(pose_el.text)
        poses[name] = {"T_W_model": T, "pose_vals": vals, "source": "include"}

    return poses


def get_R_opt_to_link():
    return rpy_to_R(0.0, -math.pi / 2.0, math.pi / 2.0)


def sdf_model_pose_to_optical(T_W_model):
    T_opt_to_link = make_T(get_R_opt_to_link(), np.zeros(3))
    return T_W_model @ invT(T_opt_to_link)


def marker_name(marker_id: int) -> str:
    if int(marker_id) == REF_MARKER_ID:
        return REF_MARKER_ENTITY
    return f"marker_{int(marker_id):03d}"


def sdf_marker_model_to_opencv_frame(T_W_model):
    T_model_cvmarker = make_T(R_MODEL_FROM_OPENCV_MARKER, np.zeros(3))
    return T_W_model @ T_model_cvmarker


def load_gt_world_entities():
    poses = parse_world_poses()

    gt = {}

    for cam in STATIC_CAMERAS:
        if cam not in poses:
            raise RuntimeError(f"Could not find camera {cam} in {WORLD_SDF}")
        gt[("static_camera", cam)] = sdf_model_pose_to_optical(poses[cam]["T_W_model"])

    for mid in range(0, 15):
        name = marker_name(mid)
        if name not in poses:
            raise RuntimeError(f"Could not find marker {name} in {WORLD_SDF}")
        gt[("aruco_marker", mid)] = sdf_marker_model_to_opencv_frame(poses[name]["T_W_model"])

    return gt


def T_from_ap02_row(row):
    rvec = np.array([
        float(row["rvec_x"]),
        float(row["rvec_y"]),
        float(row["rvec_z"]),
    ], dtype=np.float64)

    t = np.array([
        float(row["x_m"]),
        float(row["y_m"]),
        float(row["z_m"]),
    ], dtype=np.float64)

    return make_T(rvec_to_R(rvec), t)


def load_ap02_est_local_entities():
    est = {}

    for r in read_csv(AP02_CAMERA_CSV):
        cam = r["entity_id"]
        est[("static_camera", cam)] = T_from_ap02_row(r)

    for r in read_csv(AP02_MARKER_CSV):
        mid = int(float(r["entity_id"]))
        est[("aruco_marker", mid)] = T_from_ap02_row(r)

    return est


def estimate_se3_from_points(src_pts, dst_pts):
    """
    Estimate dst ~= R @ src + t using Kabsch/Umeyama without scale.
    src_pts: estimated local AP02 positions
    dst_pts: GT world positions
    """
    src = np.asarray(src_pts, dtype=np.float64)
    dst = np.asarray(dst_pts, dtype=np.float64)

    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 3:
        raise RuntimeError(f"Invalid alignment point shapes: {src.shape}, {dst.shape}")
    if src.shape[0] < 3:
        raise RuntimeError("Need at least 3 entities for SE(3) alignment")

    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)

    src_c = src - src_mean
    dst_c = dst - dst_mean

    H = src_c.T @ dst_c
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1.0
        R = Vt.T @ U.T

    t = dst_mean - R @ src_mean
    return make_T(R, t), S


def pose_columns(prefix, T):
    rpy = R_to_rpy_deg(T[:3, :3])
    rvec = R_to_rvec(T[:3, :3])
    return {
        f"{prefix}_x_m": T[0, 3],
        f"{prefix}_y_m": T[1, 3],
        f"{prefix}_z_m": T[2, 3],
        f"{prefix}_roll_deg": rpy[0],
        f"{prefix}_pitch_deg": rpy[1],
        f"{prefix}_yaw_deg": rpy[2],
        f"{prefix}_rvec_x": rvec[0],
        f"{prefix}_rvec_y": rvec[1],
        f"{prefix}_rvec_z": rvec[2],
    }


def delta_columns(T_est_aligned, T_gt):
    dt = T_est_aligned[:3, 3] - T_gt[:3, 3]
    return {
        "delta_x_cm": 100.0 * dt[0],
        "delta_y_cm": 100.0 * dt[1],
        "delta_z_cm": 100.0 * dt[2],
    }


def mean(xs):
    xs = [float(x) for x in xs]
    return sum(xs) / len(xs) if xs else float("nan")


def median(xs):
    xs = sorted([float(x) for x in xs])
    if not xs:
        return float("nan")
    n = len(xs)
    if n % 2:
        return xs[n // 2]
    return 0.5 * (xs[n // 2 - 1] + xs[n // 2])


def fieldnames():
    return [
        "approach",
        "evaluation",
        "entity_type",
        "entity_id",
        "marker_id",
        "used_for_alignment",
        "alignment_frame",
        "translation_error_cm",
        "rotation_error_deg",
        "delta_x_cm",
        "delta_y_cm",
        "delta_z_cm",
        "est_local_x_m",
        "est_local_y_m",
        "est_local_z_m",
        "est_local_roll_deg",
        "est_local_pitch_deg",
        "est_local_yaw_deg",
        "est_local_rvec_x",
        "est_local_rvec_y",
        "est_local_rvec_z",
        "est_gt_aligned_x_m",
        "est_gt_aligned_y_m",
        "est_gt_aligned_z_m",
        "est_gt_aligned_roll_deg",
        "est_gt_aligned_pitch_deg",
        "est_gt_aligned_yaw_deg",
        "est_gt_aligned_rvec_x",
        "est_gt_aligned_rvec_y",
        "est_gt_aligned_rvec_z",
        "gt_world_x_m",
        "gt_world_y_m",
        "gt_world_z_m",
        "gt_world_roll_deg",
        "gt_world_pitch_deg",
        "gt_world_yaw_deg",
        "gt_world_rvec_x",
        "gt_world_rvec_y",
        "gt_world_rvec_z",
    ]


def main():
    ensure_dir(AP02_FINAL)

    gt = load_gt_world_entities()
    est = load_ap02_est_local_entities()

    # Use all static cameras and all non-reference markers for alignment.
    # Marker 14 is explicitly held out, so its final row is not a forced zero.
    alignment_keys = []
    for cam in STATIC_CAMERAS:
        alignment_keys.append(("static_camera", cam))
    for mid in range(0, 15):
        if mid == REF_MARKER_ID:
            continue
        alignment_keys.append(("aruco_marker", mid))

    src_pts = []
    dst_pts = []
    used_keys = []

    for key in alignment_keys:
        if key not in est:
            raise RuntimeError(f"Missing AP02 estimated entity for alignment: {key}")
        if key not in gt:
            raise RuntimeError(f"Missing GT entity for alignment: {key}")
        src_pts.append(est[key][:3, 3])
        dst_pts.append(gt[key][:3, 3])
        used_keys.append(key)

    T_gt_from_est_local, singular_values = estimate_se3_from_points(src_pts, dst_pts)

    rows = []

    ordered_eval_keys = []
    for cam in STATIC_CAMERAS:
        ordered_eval_keys.append(("static_camera", cam))
    for mid in range(0, 15):
        ordered_eval_keys.append(("aruco_marker", mid))

    for key in ordered_eval_keys:
        if key not in est:
            raise RuntimeError(f"Missing AP02 estimated entity for evaluation: {key}")
        if key not in gt:
            raise RuntimeError(f"Missing GT entity for evaluation: {key}")

        entity_type, entity_id_raw = key

        T_est_local = est[key]
        T_est_aligned = T_gt_from_est_local @ T_est_local
        T_gt = gt[key]

        if entity_type == "aruco_marker":
            mid = int(entity_id_raw)
            entity_id = marker_name(mid)
            marker_id = mid
        else:
            entity_id = str(entity_id_raw)
            marker_id = ""

        row = {
            "approach": "AP02_ref_marker_graph_ba",
            "evaluation": "gt_aligned_full_map",
            "entity_type": entity_type,
            "entity_id": entity_id,
            "marker_id": marker_id,
            "used_for_alignment": "yes" if key in used_keys else "no",
            "alignment_frame": "GT_world_best_fit_SE3_using_static_cameras_and_markers_0_to_13",
            "translation_error_cm": trans_error_cm(T_est_aligned, T_gt),
            "rotation_error_deg": rot_error_deg(T_est_aligned, T_gt),
        }
        row.update(delta_columns(T_est_aligned, T_gt))
        row.update(pose_columns("est_local", T_est_local))
        row.update(pose_columns("est_gt_aligned", T_est_aligned))
        row.update(pose_columns("gt_world", T_gt))
        rows.append(row)

    write_csv(OUT_CSV, rows, fieldnames())

    camera_rows = [r for r in rows if r["entity_type"] == "static_camera"]
    marker_rows = [r for r in rows if r["entity_type"] == "aruco_marker"]
    marker14_rows = [r for r in rows if r["entity_id"] == REF_MARKER_ENTITY]
    if len(marker14_rows) != 1:
        raise RuntimeError(f"Expected exactly one marker14 row, got {len(marker14_rows)}")
    marker14 = marker14_rows[0]

    meta = {
        "approach": "AP02_ref_marker_graph_ba",
        "evaluation": "gt_aligned_full_map",
        "world_sdf": str(WORLD_SDF),
        "ap02_camera_input": str(AP02_CAMERA_CSV),
        "ap02_marker_input": str(AP02_MARKER_CSV),
        "output_csv": str(OUT_CSV),
        "output_txt": str(OUT_TXT),
        "alignment": {
            "type": "best_fit_SE3_from_entity_centers_without_scale",
            "gt_used_only_after_optimization": True,
            "used_entities_count": len(used_keys),
            "used_static_cameras": STATIC_CAMERAS,
            "used_marker_ids": [mid for mid in range(0, 15) if mid != REF_MARKER_ID],
            "held_out_marker_id": REF_MARKER_ID,
            "held_out_marker_entity": REF_MARKER_ENTITY,
            "singular_values": [float(x) for x in singular_values],
            "T_gt_from_est_local": T_gt_from_est_local.tolist(),
        },
        "important_note": (
            "GT is used only after AP02 optimization for evaluation alignment. "
            "Marker 14 is held out from the alignment, so its error is a real residual "
            "and not a coordinate-frame zero."
        ),
    }
    OUT_META.write_text(json.dumps(meta, indent=2) + "\n")

    cam_t = [r["translation_error_cm"] for r in camera_rows]
    cam_r = [r["rotation_error_deg"] for r in camera_rows]
    marker_t = [r["translation_error_cm"] for r in marker_rows]
    marker_r = [r["rotation_error_deg"] for r in marker_rows]

    lines = [
        "AP02 GT-Aligned Full-Map Evaluation",
        "====================================",
        "",
        "What this evaluates:",
        "- AP02 estimated map is aligned to the Gazebo GT map with a best-fit SE(3) transform.",
        "- The alignment uses static cameras and markers 0..13.",
        "- Marker 14 / aruco_ref_floor_14 is held out from the alignment.",
        "- Therefore marker 14 receives a real residual/error instead of a forced coordinate-frame zero.",
        "- GT is used only after optimization for evaluation, not during AP02 estimation.",
        "",
        "Alignment:",
        f"- used entities: {len(used_keys)}",
        "- used static cameras: " + ", ".join(STATIC_CAMERAS),
        "- used marker ids: 0..13",
        f"- held-out marker: {REF_MARKER_ID} / {REF_MARKER_ENTITY}",
        "",
        "Camera pose errors:",
        f"- count: {len(camera_rows)}",
        f"- mean translation error [cm]: {mean(cam_t):.6f}",
        f"- median translation error [cm]: {median(cam_t):.6f}",
        f"- mean rotation error [deg]: {mean(cam_r):.6f}",
        f"- median rotation error [deg]: {median(cam_r):.6f}",
        "",
        "Marker-map pose errors:",
        f"- count: {len(marker_rows)}",
        f"- mean translation error [cm]: {mean(marker_t):.6f}",
        f"- median translation error [cm]: {median(marker_t):.6f}",
        f"- mean rotation error [deg]: {mean(marker_r):.6f}",
        f"- median rotation error [deg]: {median(marker_r):.6f}",
        "",
        "Held-out reference marker result:",
        f"- {marker14['entity_id']}: "
        f"{float(marker14['translation_error_cm']):.3f} cm, "
        f"{float(marker14['rotation_error_deg']):.3f} deg",
        "",
        "Per-camera:",
    ]

    for r in camera_rows:
        lines.append(
            f"- {r['entity_id']}: "
            f"{float(r['translation_error_cm']):.3f} cm, "
            f"{float(r['rotation_error_deg']):.3f} deg | "
            f"aligned_est=({float(r['est_gt_aligned_x_m']):+.3f}, "
            f"{float(r['est_gt_aligned_y_m']):+.3f}, "
            f"{float(r['est_gt_aligned_z_m']):+.3f}) m | "
            f"gt=({float(r['gt_world_x_m']):+.3f}, "
            f"{float(r['gt_world_y_m']):+.3f}, "
            f"{float(r['gt_world_z_m']):+.3f}) m"
        )

    lines += ["", "Per-marker:"]

    for r in marker_rows:
        align_tag = "alignment" if r["used_for_alignment"] == "yes" else "held-out"
        lines.append(
            f"- {r['entity_id']} [{align_tag}]: "
            f"{float(r['translation_error_cm']):.3f} cm, "
            f"{float(r['rotation_error_deg']):.3f} deg | "
            f"aligned_est=({float(r['est_gt_aligned_x_m']):+.3f}, "
            f"{float(r['est_gt_aligned_y_m']):+.3f}, "
            f"{float(r['est_gt_aligned_z_m']):+.3f}) m | "
            f"gt=({float(r['gt_world_x_m']):+.3f}, "
            f"{float(r['gt_world_y_m']):+.3f}, "
            f"{float(r['gt_world_z_m']):+.3f}) m"
        )

    lines += [
        "",
        "Output files:",
        f"- {OUT_CSV}",
        f"- {OUT_TXT}",
        f"- {OUT_META}",
        "",
    ]

    OUT_TXT.write_text("\n".join(lines) + "\n")

    print("[OK] wrote:", OUT_CSV)
    print("[OK] wrote:", OUT_TXT)
    print("[OK] wrote:", OUT_META)
    print()
    print(OUT_TXT.read_text())


if __name__ == "__main__":
    main()
