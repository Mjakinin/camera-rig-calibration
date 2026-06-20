#!/usr/bin/env python3

import csv
import json
import math
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np


AP3_ROOT = Path("results/bus_real_data/03_targetless_colmap_aruco_scale")
DATASET_ROOT = AP3_ROOT / "01_colmap_dataset"
IMAGE_DIR = DATASET_ROOT / "images"
TXT_ROOT = AP3_ROOT / "02_colmap_sparse" / "sparse_txt"
INSPECT_SUMMARY = AP3_ROOT / "03_reconstruction_inspection" / "colmap_model_summary.csv"

SHARED_RAW = Path("results/bus_real_data/00_raw_images/bus_real_data_ref_marker_v1")
CAMERA_INFO_DIR = SHARED_RAW / "camera_info"
WORLD_SDF = Path("src/calib_lab/bus_real_data/worlds/bus_real_data_moving_camera.sdf")

OUT = AP3_ROOT / "04_ref_aruco_registration"
CMP_OUT = Path("results/bus_real_data/90_approach_comparison_ref_aruco/03_targetless_colmap_aruco_scale")
COMBINED = Path("results/bus_real_data/90_approach_comparison_ref_aruco/combined")

REF_MARKER_ID = 14
REF_MARKER_ENTITY = "aruco_ref_floor_14"
MARKER_LENGTH_M = 0.170

STATIC_CAMERAS = ["cam_edge_0", "cam_edge_1", "cam_edge_3", "cam_edge_5"]

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


def qvec_to_R(qvec):
    qw, qx, qy, qz = qvec
    return np.array([
        [1 - 2*qy*qy - 2*qz*qz,     2*qx*qy - 2*qz*qw,     2*qx*qz + 2*qy*qw],
        [    2*qx*qy + 2*qz*qw, 1 - 2*qx*qx - 2*qz*qz,     2*qy*qz - 2*qx*qw],
        [    2*qx*qz - 2*qy*qw,     2*qy*qz + 2*qx*qw, 1 - 2*qx*qx - 2*qy*qy],
    ], dtype=np.float64)


def rpy_to_R(roll, pitch, yaw):
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)

    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float64)
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float64)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float64)
    return Rz @ Ry @ Rx


def R_to_rpy_deg(R):
    pitch = math.atan2(-R[2, 0], math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2))
    roll = math.atan2(R[2, 1], R[2, 2])
    yaw = math.atan2(R[1, 0], R[0, 0])
    return [math.degrees(roll), math.degrees(pitch), math.degrees(yaw)]


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


def rvec_to_R(rvec):
    R, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
    return R.astype(np.float64)


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


def pose_row(entity_type, entity_id, T, source):
    rpy = R_to_rpy_deg(T[:3, :3])
    rvec = R_to_rvec(T[:3, :3])
    return {
        "entity_type": entity_type,
        "entity_id": entity_id,
        "source": source,
        "x_m": T[0, 3],
        "y_m": T[1, 3],
        "z_m": T[2, 3],
        "roll_deg": rpy[0],
        "pitch_deg": rpy[1],
        "yaw_deg": rpy[2],
        "rvec_x": rvec[0],
        "rvec_y": rvec[1],
        "rvec_z": rvec[2],
    }


def pose_fields():
    return [
        "entity_type", "entity_id", "source",
        "x_m", "y_m", "z_m",
        "roll_deg", "pitch_deg", "yaw_deg",
        "rvec_x", "rvec_y", "rvec_z",
    ]


def parse_best_model_name():
    rows = read_csv(INSPECT_SUMMARY)

    if not rows:
        raise RuntimeError(f"No COLMAP model summary rows in {INSPECT_SUMMARY}")

    rows = sorted(
        rows,
        key=lambda r: (
            int(r["registered_static_cameras"]),
            int(r["registered_images"]),
            int(r["num_3d_points"]),
        ),
        reverse=True,
    )

    return rows[0]["model"]


def parse_images_txt(path: Path):
    out = {}

    lines = path.read_text().splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()

        if not line or line.startswith("#"):
            i += 1
            continue

        parts = line.split()
        if len(parts) >= 10:
            image_id = int(parts[0])
            qvec = np.array([float(x) for x in parts[1:5]], dtype=np.float64)
            tvec = np.array([float(x) for x in parts[5:8]], dtype=np.float64)
            camera_id = int(parts[8])
            name = parts[9]

            R_cw = qvec_to_R(qvec)
            T_cam_col = make_T(R_cw, tvec)
            T_col_cam = invT(T_cam_col)

            out[name] = {
                "image_id": image_id,
                "camera_id": camera_id,
                "qvec": qvec,
                "tvec": tvec,
                "T_col_cam": T_col_cam,
            }

            i += 2
        else:
            i += 1

    return out


def parse_points3D_txt(path: Path):
    pts = []

    if not path.exists():
        return pts

    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        parts = line.split()
        if len(parts) < 8:
            continue

        pts.append({
            "point3d_id": int(parts[0]),
            "x_colmap": float(parts[1]),
            "y_colmap": float(parts[2]),
            "z_colmap": float(parts[3]),
            "r": int(parts[4]),
            "g": int(parts[5]),
            "b": int(parts[6]),
            "error": float(parts[7]),
        })

    return pts


def source_id_from_image_name(name):
    if name.startswith("static_") and name.endswith(".png"):
        return name[len("static_"):-len(".png")]
    if name.startswith("moving_") and name.endswith(".png"):
        return name[len("moving_"):-len(".png")]
    return name


def source_type_from_image_name(name):
    if name.startswith("static_"):
        return "static"
    if name.startswith("moving_"):
        return "moving"
    return "unknown"


def load_camera_info_for_colmap_image(image_name):
    source_type = source_type_from_image_name(image_name)
    source_id = source_id_from_image_name(image_name)

    candidates = []

    if source_type == "static":
        candidates += [
            CAMERA_INFO_DIR / f"{source_id}.json",
            CAMERA_INFO_DIR / f"{source_id}_camera_info.json",
        ]
    else:
        candidates += [
            CAMERA_INFO_DIR / "moving_calib_camera.json",
            CAMERA_INFO_DIR / "moving_camera.json",
            CAMERA_INFO_DIR / "calib_camera.json",
            CAMERA_INFO_DIR / "camera_info.json",
        ]

    path = None
    for c in candidates:
        if c.exists():
            path = c
            break

    if path is None:
        raise RuntimeError(
            f"Missing camera_info for {image_name}. Tried: " +
            ", ".join(str(c) for c in candidates)
        )

    data = json.loads(path.read_text())

    if "K" in data:
        Kvals = data["K"]
        K = np.array(Kvals, dtype=np.float64).reshape(3, 3)
    elif "camera_matrix" in data:
        cm = data["camera_matrix"]
        if isinstance(cm, dict) and "data" in cm:
            K = np.array(cm["data"], dtype=np.float64).reshape(3, 3)
        else:
            K = np.array(cm, dtype=np.float64).reshape(3, 3)
    else:
        fx = float(data.get("fx", data.get("f_x")))
        fy = float(data.get("fy", data.get("f_y")))
        cx = float(data.get("cx", data.get("c_x")))
        cy = float(data.get("cy", data.get("c_y")))
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)

    if "D" in data:
        D = np.array(data["D"], dtype=np.float64).reshape(-1)
    elif "distortion_coefficients" in data:
        dc = data["distortion_coefficients"]
        if isinstance(dc, dict) and "data" in dc:
            D = np.array(dc["data"], dtype=np.float64).reshape(-1)
        else:
            D = np.array(dc, dtype=np.float64).reshape(-1)
    else:
        D = np.zeros(5, dtype=np.float64)

    return K, D, path


def make_aruco_detector():
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

    if hasattr(cv2.aruco, "ArucoDetector"):
        params = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(dictionary, params)

        def detect(gray):
            return detector.detectMarkers(gray)

        return detect

    params = cv2.aruco.DetectorParameters_create()

    def detect(gray):
        return cv2.aruco.detectMarkers(gray, dictionary, parameters=params)

    return detect


def marker_object_points():
    s = MARKER_LENGTH_M / 2.0
    return np.array([
        [-s,  s, 0.0],
        [ s,  s, 0.0],
        [ s, -s, 0.0],
        [-s, -s, 0.0],
    ], dtype=np.float64)


def detect_ref_marker_pose(image_name):
    image_path = IMAGE_DIR / image_name
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

    if img is None:
        raise RuntimeError(f"Could not read image: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    detect = make_aruco_detector()
    corners, ids, _ = detect(gray)

    if ids is None or len(ids) == 0:
        return None

    ids = ids.reshape(-1)

    matches = [i for i, mid in enumerate(ids.tolist()) if int(mid) == REF_MARKER_ID]
    if not matches:
        return None

    idx = matches[0]
    img_pts = np.asarray(corners[idx], dtype=np.float64).reshape(4, 2)
    obj_pts = marker_object_points()

    K, D, info_path = load_camera_info_for_colmap_image(image_name)

    ok, rvec, tvec = cv2.solvePnP(
        obj_pts,
        img_pts,
        K,
        D,
        flags=cv2.SOLVEPNP_IPPE_SQUARE,
    )

    if not ok:
        ok, rvec, tvec = cv2.solvePnP(
            obj_pts,
            img_pts,
            K,
            D,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )

    if not ok:
        return None

    R_cam_ref = rvec_to_R(rvec.reshape(3))
    T_cam_ref = make_T(R_cam_ref, tvec.reshape(3))
    T_ref_cam = invT(T_cam_ref)

    area = float(cv2.contourArea(img_pts.astype(np.float32)))

    return {
        "image_name": image_name,
        "source_type": source_type_from_image_name(image_name),
        "source_id": source_id_from_image_name(image_name),
        "camera_info": str(info_path),
        "area_px2": area,
        "T_ref_cam_from_refmarker_pnp": T_ref_cam,
    }


def umeyama_similarity(X, Y):
    # Estimate Y ~= s * R * X + t.
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)

    if X.shape != Y.shape or X.shape[0] < 3 or X.shape[1] != 3:
        raise RuntimeError(f"Need Nx3 correspondence arrays with N>=3, got {X.shape} and {Y.shape}")

    n = X.shape[0]
    mu_x = X.mean(axis=0)
    mu_y = Y.mean(axis=0)

    Xc = X - mu_x
    Yc = Y - mu_y

    var_x = np.mean(np.sum(Xc * Xc, axis=1))
    Sigma = (Yc.T @ Xc) / n

    U, D, Vt = np.linalg.svd(Sigma)

    S = np.eye(3)
    if np.linalg.det(U @ Vt) < 0:
        S[2, 2] = -1.0

    R = U @ S @ Vt
    scale = float(np.trace(np.diag(D) @ S) / var_x)
    t = mu_y - scale * R @ mu_x

    return scale, R, t


def apply_sim3_to_pose(T_col_cam, scale, R_ref_col, t_ref_col):
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R_ref_col @ T_col_cam[:3, :3]
    T[:3, 3] = scale * (R_ref_col @ T_col_cam[:3, 3]) + t_ref_col
    return T


def apply_sim3_to_point(p_col, scale, R_ref_col, t_ref_col):
    p_col = np.asarray(p_col, dtype=np.float64).reshape(3)
    return scale * (R_ref_col @ p_col) + t_ref_col


def parse_pose_text(text):
    vals = [float(x) for x in text.split()]
    if len(vals) < 6:
        raise RuntimeError(f"Invalid SDF pose: {text}")
    x, y, z, roll, pitch, yaw = vals[:6]
    return make_T(rpy_to_R(roll, pitch, yaw), [x, y, z]), vals[:6]


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
        poses[name] = {"T_W_model": T, "pose_vals": vals}

    for inc in root.iter("include"):
        name_el = inc.find("name")
        pose_el = inc.find("pose")
        if name_el is None or pose_el is None or not name_el.text or not pose_el.text:
            continue
        name = name_el.text.strip()
        T, vals = parse_pose_text(pose_el.text)
        poses[name] = {"T_W_model": T, "pose_vals": vals}

    return poses


def get_R_opt_to_link():
    return rpy_to_R(0.0, -math.pi / 2.0, math.pi / 2.0)


def sdf_model_pose_to_optical(T_W_model):
    T_opt_to_link = make_T(get_R_opt_to_link(), np.zeros(3))
    return T_W_model @ invT(T_opt_to_link)


def sdf_marker_model_to_opencv_frame(T_W_model):
    T_model_cvmarker = make_T(R_MODEL_FROM_OPENCV_MARKER, np.zeros(3))
    return T_W_model @ T_model_cvmarker


def gt_static_camera_poses_ref_aruco():
    poses = parse_world_poses()

    if REF_MARKER_ENTITY not in poses:
        raise RuntimeError(f"Missing reference marker in SDF: {REF_MARKER_ENTITY}")

    T_W_ref_cv = sdf_marker_model_to_opencv_frame(poses[REF_MARKER_ENTITY]["T_W_model"])
    T_ref_W = invT(T_W_ref_cv)

    gt = {}
    for cam in STATIC_CAMERAS:
        if cam not in poses:
            raise RuntimeError(f"Missing camera in SDF: {cam}")
        T_W_cam = sdf_model_pose_to_optical(poses[cam]["T_W_model"])
        gt[cam] = T_ref_W @ T_W_cam

    return gt


def metric_pose_columns(prefix, T):
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


def main():
    ensure_dir(OUT)
    ensure_dir(CMP_OUT)
    ensure_dir(COMBINED)

    best_model = parse_best_model_name()
    model_dir = TXT_ROOT / best_model

    if not model_dir.exists():
        raise RuntimeError(f"Missing best COLMAP txt model dir: {model_dir}")

    images = parse_images_txt(model_dir / "images.txt")
    points3d = parse_points3D_txt(model_dir / "points3D.txt")

    if not images:
        raise RuntimeError(f"No registered images parsed from {model_dir / 'images.txt'}")

    anchors = []

    for image_name, payload in sorted(images.items()):
        det = detect_ref_marker_pose(image_name)
        if det is None:
            continue

        T_col_cam = payload["T_col_cam"]
        T_ref_cam = det["T_ref_cam_from_refmarker_pnp"]

        anchors.append({
            "image_name": image_name,
            "source_type": det["source_type"],
            "source_id": det["source_id"],
            "area_px2": det["area_px2"],
            "camera_info": det["camera_info"],
            "T_col_cam": T_col_cam,
            "T_ref_cam": T_ref_cam,
        })

    if len(anchors) < 3:
        raise RuntimeError(
            f"Need at least 3 images observing ref marker {REF_MARKER_ID} for Sim(3), got {len(anchors)}"
        )

    X_col = np.array([a["T_col_cam"][:3, 3] for a in anchors], dtype=np.float64)
    Y_ref = np.array([a["T_ref_cam"][:3, 3] for a in anchors], dtype=np.float64)

    scale, R_ref_col, t_ref_col = umeyama_similarity(X_col, Y_ref)

    anchor_rows = []
    anchor_residuals_cm = []
    anchor_rot_err_deg = []

    for a in anchors:
        T_ref_cam_from_colmap = apply_sim3_to_pose(a["T_col_cam"], scale, R_ref_col, t_ref_col)
        T_ref_cam_pnp = a["T_ref_cam"]

        terr = trans_error_cm(T_ref_cam_from_colmap, T_ref_cam_pnp)
        rerr = rot_error_deg(T_ref_cam_from_colmap, T_ref_cam_pnp)

        anchor_residuals_cm.append(terr)
        anchor_rot_err_deg.append(rerr)

        row = {
            "image_name": a["image_name"],
            "source_type": a["source_type"],
            "source_id": a["source_id"],
            "area_px2": a["area_px2"],
            "translation_residual_cm": terr,
            "rotation_residual_deg": rerr,
            "camera_info": a["camera_info"],
        }
        row.update(metric_pose_columns("colmap_registered_ref_aruco", T_ref_cam_from_colmap))
        row.update(metric_pose_columns("ref_marker_pnp_ref_aruco", T_ref_cam_pnp))
        anchor_rows.append(row)

    anchor_fields = [
        "image_name", "source_type", "source_id", "area_px2",
        "translation_residual_cm", "rotation_residual_deg", "camera_info",
        "colmap_registered_ref_aruco_x_m",
        "colmap_registered_ref_aruco_y_m",
        "colmap_registered_ref_aruco_z_m",
        "colmap_registered_ref_aruco_roll_deg",
        "colmap_registered_ref_aruco_pitch_deg",
        "colmap_registered_ref_aruco_yaw_deg",
        "colmap_registered_ref_aruco_rvec_x",
        "colmap_registered_ref_aruco_rvec_y",
        "colmap_registered_ref_aruco_rvec_z",
        "ref_marker_pnp_ref_aruco_x_m",
        "ref_marker_pnp_ref_aruco_y_m",
        "ref_marker_pnp_ref_aruco_z_m",
        "ref_marker_pnp_ref_aruco_roll_deg",
        "ref_marker_pnp_ref_aruco_pitch_deg",
        "ref_marker_pnp_ref_aruco_yaw_deg",
        "ref_marker_pnp_ref_aruco_rvec_x",
        "ref_marker_pnp_ref_aruco_rvec_y",
        "ref_marker_pnp_ref_aruco_rvec_z",
    ]

    write_csv(OUT / "ap03_ref_marker_anchor_residuals.csv", anchor_rows, anchor_fields)

    static_pose_rows = []
    moving_pose_rows = []

    for image_name, payload in sorted(images.items()):
        T_ref_cam = apply_sim3_to_pose(payload["T_col_cam"], scale, R_ref_col, t_ref_col)

        if image_name.startswith("static_"):
            cam = source_id_from_image_name(image_name)
            static_pose_rows.append(pose_row("static_camera", cam, T_ref_cam, "ap03_colmap_ref_aruco_sim3"))
        elif image_name.startswith("moving_"):
            frame = source_id_from_image_name(image_name)
            moving_pose_rows.append(pose_row("moving_frame", frame, T_ref_cam, "ap03_colmap_ref_aruco_sim3"))

    write_csv(OUT / "ap03_static_camera_poses_ref_aruco.csv", static_pose_rows, pose_fields())
    write_csv(OUT / "ap03_moving_frame_poses_ref_aruco.csv", moving_pose_rows, pose_fields())

    point_rows = []
    for p in points3d:
        p_ref = apply_sim3_to_point(
            np.array([p["x_colmap"], p["y_colmap"], p["z_colmap"]], dtype=np.float64),
            scale,
            R_ref_col,
            t_ref_col,
        )
        point_rows.append({
            "point3d_id": p["point3d_id"],
            "x_ref_aruco_m": p_ref[0],
            "y_ref_aruco_m": p_ref[1],
            "z_ref_aruco_m": p_ref[2],
            "r": p["r"],
            "g": p["g"],
            "b": p["b"],
            "colmap_reprojection_error": p["error"],
        })

    write_csv(
        OUT / "ap03_sparse_points3d_ref_aruco.csv",
        point_rows,
        ["point3d_id", "x_ref_aruco_m", "y_ref_aruco_m", "z_ref_aruco_m", "r", "g", "b", "colmap_reprojection_error"],
    )

    gt = gt_static_camera_poses_ref_aruco()
    est = {r["entity_id"]: r for r in static_pose_rows}

    eval_rows = []

    for cam in STATIC_CAMERAS:
        if cam not in est:
            raise RuntimeError(f"Missing AP03 static camera estimate: {cam}")

        r = est[cam]
        T_est = make_T(
            rvec_to_R([float(r["rvec_x"]), float(r["rvec_y"]), float(r["rvec_z"])]),
            [float(r["x_m"]), float(r["y_m"]), float(r["z_m"])],
        )
        T_gt = gt[cam]

        row = {
            "approach": "AP03_targetless_colmap_aruco_scale",
            "entity_type": "static_camera",
            "entity_id": cam,
            "reference_frame": f"aruco_marker_{REF_MARKER_ID}",
            "translation_error_cm": trans_error_cm(T_est, T_gt),
            "rotation_error_deg": rot_error_deg(T_est, T_gt),
            "delta_x_cm": 100.0 * (T_est[0, 3] - T_gt[0, 3]),
            "delta_y_cm": 100.0 * (T_est[1, 3] - T_gt[1, 3]),
            "delta_z_cm": 100.0 * (T_est[2, 3] - T_gt[2, 3]),
        }
        row.update(metric_pose_columns("est_ref_aruco", T_est))
        row.update(metric_pose_columns("gt_ref_aruco", T_gt))
        eval_rows.append(row)

    eval_fields = [
        "approach",
        "entity_type",
        "entity_id",
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

    write_csv(CMP_OUT / "ap03_static_cameras_ref_aruco_vs_gt.csv", eval_rows, eval_fields)

    cam_t = [r["translation_error_cm"] for r in eval_rows]
    cam_r = [r["rotation_error_deg"] for r in eval_rows]

    summary_rows = [{
        "approach": "AP03_targetless_colmap_aruco_scale",
        "reference_frame": f"aruco_marker_{REF_MARKER_ID}",
        "final_variant": "targetless_colmap_sim3_ref_marker_14",
        "registered_images": len(images),
        "registered_static_cameras": len(static_pose_rows),
        "registered_moving_frames": len(moving_pose_rows),
        "num_sparse_points3d": len(point_rows),
        "ref_marker_anchor_count": len(anchor_rows),
        "estimated_colmap_to_metric_scale": scale,
        "anchor_mean_translation_residual_cm": mean(anchor_residuals_cm),
        "anchor_median_translation_residual_cm": median(anchor_residuals_cm),
        "anchor_mean_rotation_residual_deg": mean(anchor_rot_err_deg),
        "anchor_median_rotation_residual_deg": median(anchor_rot_err_deg),
        "camera_mean_translation_error_cm": mean(cam_t),
        "camera_median_translation_error_cm": median(cam_t),
        "camera_mean_rotation_error_deg": mean(cam_r),
        "camera_median_rotation_error_deg": median(cam_r),
        "camera_eval_file": str(CMP_OUT / "ap03_static_cameras_ref_aruco_vs_gt.csv"),
    }]

    write_csv(
        COMBINED / "ap03_final_readable_summary.csv",
        summary_rows,
        list(summary_rows[0].keys()),
    )

    meta = {
        "approach": "AP03_targetless_colmap_aruco_scale",
        "description": "Targetless COLMAP/SfM reconstruction registered to metric Ref-ArUco frame using marker 14 PnP anchors.",
        "best_colmap_model": best_model,
        "reference_marker_id": REF_MARKER_ID,
        "marker_length_m": MARKER_LENGTH_M,
        "scale_colmap_to_metric": scale,
        "registered_images": len(images),
        "registered_static_cameras": len(static_pose_rows),
        "registered_moving_frames": len(moving_pose_rows),
        "num_sparse_points3d": len(point_rows),
        "anchor_count": len(anchor_rows),
        "important_note": "ArUco is used only for final metric scale/ref-frame registration, not as the COLMAP reconstruction frontend.",
    }

    (OUT / "ap03_ref_aruco_registration_metadata.json").write_text(json.dumps(meta, indent=2) + "\n")

    report = [
        "AP03 Targetless COLMAP + Ref-ArUco Scale Registration",
        "=====================================================",
        "",
        "Method:",
        "- COLMAP reconstructs all images from natural image features.",
        "- The resulting reconstruction is in arbitrary COLMAP scale and coordinate frame.",
        "- Ref-ArUco marker 14 is detected after COLMAP only to estimate metric Sim(3) scale/registration.",
        "- Final static camera poses are expressed in the Ref-ArUco frame.",
        "",
        "COLMAP reconstruction:",
        f"- best model: {best_model}",
        f"- registered images: {len(images)}",
        f"- registered static cameras: {len(static_pose_rows)} / 4",
        f"- registered moving frames: {len(moving_pose_rows)}",
        f"- sparse 3D points: {len(point_rows)}",
        "",
        "Ref-ArUco registration:",
        f"- ref marker id: {REF_MARKER_ID}",
        f"- marker length [m]: {MARKER_LENGTH_M}",
        f"- ref-marker anchor images: {len(anchor_rows)}",
        f"- estimated COLMAP-to-metric scale: {scale:.9f}",
        f"- anchor mean translation residual [cm]: {mean(anchor_residuals_cm):.6f}",
        f"- anchor median translation residual [cm]: {median(anchor_residuals_cm):.6f}",
        f"- anchor mean rotation residual [deg]: {mean(anchor_rot_err_deg):.6f}",
        f"- anchor median rotation residual [deg]: {median(anchor_rot_err_deg):.6f}",
        "",
        "GT evaluation of static cameras in Ref-ArUco frame:",
        f"- mean translation error [cm]: {mean(cam_t):.6f}",
        f"- median translation error [cm]: {median(cam_t):.6f}",
        f"- mean rotation error [deg]: {mean(cam_r):.6f}",
        f"- median rotation error [deg]: {median(cam_r):.6f}",
        "",
        "Per-camera:",
    ]

    for r in eval_rows:
        report.append(
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

    report += [
        "",
        "Output files:",
        f"- {OUT / 'ap03_static_camera_poses_ref_aruco.csv'}",
        f"- {OUT / 'ap03_moving_frame_poses_ref_aruco.csv'}",
        f"- {OUT / 'ap03_sparse_points3d_ref_aruco.csv'}",
        f"- {OUT / 'ap03_ref_marker_anchor_residuals.csv'}",
        f"- {CMP_OUT / 'ap03_static_cameras_ref_aruco_vs_gt.csv'}",
        f"- {COMBINED / 'ap03_final_readable_summary.csv'}",
        "",
    ]

    report_text = "\n".join(report) + "\n"

    (OUT / "ap03_ref_aruco_registration_report.txt").write_text(report_text)
    (CMP_OUT / "AP03_FINAL_READABLE_REF_ARUCO_REPORT.txt").write_text(report_text)

    print(report_text)


if __name__ == "__main__":
    main()
