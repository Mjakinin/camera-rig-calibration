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

OUT_ROOT = Path("results/bus_real_data/90_approach_comparison_ref_aruco")
OUT = OUT_ROOT / "02_ref_marker_graph_ba"
COMBINED = OUT_ROOT / "combined"

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
# It was chosen from the known A4 marker model geometry:
#   x_cv -> +X_model
#   y_cv -> +Z_model
#   z_cv -> -Y_model
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

    # <model name="..."><pose>...</pose>
    for model in root.iter("model"):
        name = model.attrib.get("name", "").strip()
        pose_el = model.find("pose")
        if not name or pose_el is None or not pose_el.text:
            continue
        T, vals = parse_pose_text(pose_el.text)
        poses[name] = {"T_W_model": T, "pose_vals": vals, "source": "model"}

    # <include><name>...</name><pose>...</pose></include>
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


def load_gt_ref_frame_and_entities():
    poses = parse_world_poses()

    if REF_MARKER_ENTITY not in poses:
        raise RuntimeError(f"Could not find reference marker entity {REF_MARKER_ENTITY} in {WORLD_SDF}")

    T_W_ref_model = poses[REF_MARKER_ENTITY]["T_W_model"]
    T_W_ref_cv = sdf_marker_model_to_opencv_frame(T_W_ref_model)
    T_ref_W = invT(T_W_ref_cv)

    gt_cameras = {}
    for cam in STATIC_CAMERAS:
        if cam not in poses:
            raise RuntimeError(f"Could not find camera {cam} in {WORLD_SDF}")
        T_W_cam_optical = sdf_model_pose_to_optical(poses[cam]["T_W_model"])
        gt_cameras[cam] = T_ref_W @ T_W_cam_optical

    gt_markers = {}
    for mid in range(0, 15):
        name = marker_name(mid)
        if name not in poses:
            raise RuntimeError(f"Could not find marker {name} in {WORLD_SDF}")
        T_W_marker_cv = sdf_marker_model_to_opencv_frame(poses[name]["T_W_model"])
        gt_markers[mid] = T_ref_W @ T_W_marker_cv

    return gt_cameras, gt_markers, poses[REF_MARKER_ENTITY]["pose_vals"]


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


def delta_columns(T_est, T_gt):
    dt = T_est[:3, 3] - T_gt[:3, 3]
    return {
        "delta_x_cm": 100.0 * dt[0],
        "delta_y_cm": 100.0 * dt[1],
        "delta_z_cm": 100.0 * dt[2],
    }


def eval_camera_rows(gt_cameras):
    est_rows = read_csv(AP02_CAMERA_CSV)
    est = {r["entity_id"]: T_from_ap02_row(r) for r in est_rows}

    out = []
    for cam in STATIC_CAMERAS:
        if cam not in est:
            raise RuntimeError(f"Missing AP02 estimated camera pose: {cam}")

        T_est = est[cam]
        T_gt = gt_cameras[cam]

        row = {
            "approach": "AP02_ref_marker_graph_ba",
            "entity_type": "static_camera",
            "entity_id": cam,
            "reference_frame": f"aruco_marker_{REF_MARKER_ID}",
            "translation_error_cm": trans_error_cm(T_est, T_gt),
            "rotation_error_deg": rot_error_deg(T_est, T_gt),
        }
        row.update(delta_columns(T_est, T_gt))
        row.update(pose_columns("est_ref_aruco", T_est))
        row.update(pose_columns("gt_ref_aruco", T_gt))
        out.append(row)

    return out


def eval_marker_rows(gt_markers):
    est_rows = read_csv(AP02_MARKER_CSV)

    est = {}
    for r in est_rows:
        mid = int(float(r["entity_id"]))
        est[mid] = T_from_ap02_row(r)

    out = []
    for mid in range(0, 15):
        if mid not in est:
            raise RuntimeError(f"Missing AP02 estimated marker pose: marker {mid}")

        T_est = est[mid]
        T_gt = gt_markers[mid]

        row = {
            "approach": "AP02_ref_marker_graph_ba",
            "entity_type": "aruco_marker",
            "entity_id": f"marker_{mid:03d}" if mid != REF_MARKER_ID else REF_MARKER_ENTITY,
            "marker_id": mid,
            "reference_frame": f"aruco_marker_{REF_MARKER_ID}",
            "translation_error_cm": trans_error_cm(T_est, T_gt),
            "rotation_error_deg": rot_error_deg(T_est, T_gt),
        }
        row.update(delta_columns(T_est, T_gt))
        row.update(pose_columns("est_ref_aruco", T_est))
        row.update(pose_columns("gt_ref_aruco", T_gt))
        out.append(row)

    return out


def fieldnames():
    return [
        "approach",
        "entity_type",
        "entity_id",
        "marker_id",
        "reference_frame",
        "translation_error_cm",
        "rotation_error_deg",
        "delta_x_cm",
        "delta_y_cm",
        "delta_z_cm",
        "est_ref_aruco_x_m",
        "est_ref_aruco_y_m",
        "est_ref_aruco_z_m",
        "est_ref_aruco_roll_deg",
        "est_ref_aruco_pitch_deg",
        "est_ref_aruco_yaw_deg",
        "est_ref_aruco_rvec_x",
        "est_ref_aruco_rvec_y",
        "est_ref_aruco_rvec_z",
        "gt_ref_aruco_x_m",
        "gt_ref_aruco_y_m",
        "gt_ref_aruco_z_m",
        "gt_ref_aruco_roll_deg",
        "gt_ref_aruco_pitch_deg",
        "gt_ref_aruco_yaw_deg",
        "gt_ref_aruco_rvec_x",
        "gt_ref_aruco_rvec_y",
        "gt_ref_aruco_rvec_z",
    ]


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


def write_combined_summary(camera_rows, marker_rows):
    ensure_dir(COMBINED)

    summary = [{
        "approach": "AP02_ref_marker_graph_ba",
        "reference_frame": f"aruco_marker_{REF_MARKER_ID}",
        "camera_count": len(camera_rows),
        "marker_count": len(marker_rows),
        "camera_mean_translation_error_cm": mean([r["translation_error_cm"] for r in camera_rows]),
        "camera_median_translation_error_cm": median([r["translation_error_cm"] for r in camera_rows]),
        "camera_mean_rotation_error_deg": mean([r["rotation_error_deg"] for r in camera_rows]),
        "camera_median_rotation_error_deg": median([r["rotation_error_deg"] for r in camera_rows]),
        "marker_mean_translation_error_cm": mean([r["translation_error_cm"] for r in marker_rows]),
        "marker_median_translation_error_cm": median([r["translation_error_cm"] for r in marker_rows]),
        "marker_mean_rotation_error_deg": mean([r["rotation_error_deg"] for r in marker_rows]),
        "marker_median_rotation_error_deg": median([r["rotation_error_deg"] for r in marker_rows]),
        "camera_eval_file": str(OUT / "ap02_static_cameras_ref_aruco_vs_gt.csv"),
        "marker_eval_file": str(OUT / "ap02_markers_ref_aruco_vs_gt.csv"),
    }]

    write_csv(
        COMBINED / "approach_ref_aruco_summary.csv",
        summary,
        list(summary[0].keys()),
    )


def main():
    ensure_dir(OUT)
    ensure_dir(COMBINED)

    gt_cameras, gt_markers, ref_pose_vals = load_gt_ref_frame_and_entities()

    camera_rows = eval_camera_rows(gt_cameras)
    marker_rows = eval_marker_rows(gt_markers)
    all_rows = camera_rows + marker_rows

    fields = fieldnames()

    write_csv(OUT / "ap02_static_cameras_ref_aruco_vs_gt.csv", camera_rows, fields)
    write_csv(OUT / "ap02_markers_ref_aruco_vs_gt.csv", marker_rows, fields)
    write_csv(OUT / "ap02_all_entities_ref_aruco_vs_gt.csv", all_rows, fields)

    write_combined_summary(camera_rows, marker_rows)

    meta = {
        "approach": "AP02_ref_marker_graph_ba",
        "reference_marker_id": REF_MARKER_ID,
        "reference_marker_entity": REF_MARKER_ENTITY,
        "reference_frame": "OpenCV ArUco marker coordinate frame of marker 14",
        "world_sdf": str(WORLD_SDF),
        "ap02_camera_input": str(AP02_CAMERA_CSV),
        "ap02_marker_input": str(AP02_MARKER_CSV),
        "ref_marker_sdf_pose_xyz_rpy": ref_pose_vals,
        "R_MODEL_FROM_OPENCV_MARKER": R_MODEL_FROM_OPENCV_MARKER.tolist(),
        "important_note": "GT is used only for evaluation. AP02 estimation does not use GT.",
    }

    (OUT / "ap02_ref_aruco_eval_metadata.json").write_text(json.dumps(meta, indent=2) + "\n")

    cam_t = [r["translation_error_cm"] for r in camera_rows]
    cam_r = [r["rotation_error_deg"] for r in camera_rows]
    marker_t = [r["translation_error_cm"] for r in marker_rows]
    marker_r = [r["rotation_error_deg"] for r in marker_rows]

    lines = [
        "AP02 Ref-ArUco Evaluation vs Ground Truth",
        "=========================================",
        "",
        "Reference frame:",
        f"- aruco marker {REF_MARKER_ID} / {REF_MARKER_ENTITY}",
        "- all estimated and GT poses are expressed in this reference-marker coordinate frame",
        "",
        "Camera pose errors:",
        f"- count: {len(camera_rows)}",
        f"- mean translation error [cm]: {mean(cam_t):.6f}",
        f"- median translation error [cm]: {median(cam_t):.6f}",
        f"- mean rotation error [deg]: {mean(cam_r):.6f}",
        f"- median rotation error [deg]: {median(cam_r):.6f}",
        "",
        "Per-camera:",
    ]

    for r in camera_rows:
        lines.append(
            f"- {r['entity_id']}: "
            f"{float(r['translation_error_cm']):.3f} cm, "
            f"{float(r['rotation_error_deg']):.3f} deg | "
            f"est=({float(r['est_ref_aruco_x_m']):+.3f}, "
            f"{float(r['est_ref_aruco_y_m']):+.3f}, "
            f"{float(r['est_ref_aruco_z_m']):+.3f}) m | "
            f"gt=({float(r['gt_ref_aruco_x_m']):+.3f}, "
            f"{float(r['gt_ref_aruco_y_m']):+.3f}, "
            f"{float(r['gt_ref_aruco_z_m']):+.3f}) m"
        )

    lines += [
        "",
        "Marker-map pose errors:",
        f"- count: {len(marker_rows)}",
        f"- mean translation error [cm]: {mean(marker_t):.6f}",
        f"- median translation error [cm]: {median(marker_t):.6f}",
        f"- mean rotation error [deg]: {mean(marker_r):.6f}",
        f"- median rotation error [deg]: {median(marker_r):.6f}",
        "",
        "Per-marker:",
    ]

    for r in marker_rows:
        lines.append(
            f"- {r['entity_id']}: "
            f"{float(r['translation_error_cm']):.3f} cm, "
            f"{float(r['rotation_error_deg']):.3f} deg | "
            f"est=({float(r['est_ref_aruco_x_m']):+.3f}, "
            f"{float(r['est_ref_aruco_y_m']):+.3f}, "
            f"{float(r['est_ref_aruco_z_m']):+.3f}) m | "
            f"gt=({float(r['gt_ref_aruco_x_m']):+.3f}, "
            f"{float(r['gt_ref_aruco_y_m']):+.3f}, "
            f"{float(r['gt_ref_aruco_z_m']):+.3f}) m"
        )

    lines += [
        "",
        "Output files:",
        f"- {OUT / 'ap02_static_cameras_ref_aruco_vs_gt.csv'}",
        f"- {OUT / 'ap02_markers_ref_aruco_vs_gt.csv'}",
        f"- {OUT / 'ap02_all_entities_ref_aruco_vs_gt.csv'}",
        f"- {COMBINED / 'approach_ref_aruco_summary.csv'}",
        "",
    ]

    (OUT / "ap02_ref_aruco_eval_report.txt").write_text("\n".join(lines) + "\n")

    print("[OK] wrote AP02 ref-ArUco evaluation:", OUT)
    print()
    print((OUT / "ap02_ref_aruco_eval_report.txt").read_text())


if __name__ == "__main__":
    main()
