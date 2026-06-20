#!/usr/bin/env python3

import argparse
import csv
import json
import math
import random
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np


AP3_ROOT = Path("results/bus_real_data/03_targetless_colmap_aruco_scale")
TXT_ROOT = AP3_ROOT / "02_colmap_sparse" / "sparse_txt"
IMAGE_DIR = AP3_ROOT / "01_colmap_dataset" / "images"
INSPECT_SUMMARY = AP3_ROOT / "03_reconstruction_inspection" / "colmap_model_summary.csv"

WORLD_SDF = Path("src/calib_lab/bus_real_data/worlds/bus_real_data_moving_camera.sdf")

OUT_ROOT = AP3_ROOT / "06_triangulated_ref_aruco_registration"
CMP_ROOT = Path("results/bus_real_data/90_approach_comparison_ref_aruco")
AP3_CMP = CMP_ROOT / "03_targetless_colmap_aruco_scale"
COMBINED = CMP_ROOT / "combined"

STATIC_CAMERAS = ["cam_edge_0", "cam_edge_1", "cam_edge_3", "cam_edge_5"]

REF_MARKER_ID = 14
REF_MARKER_ENTITY = "aruco_ref_floor_14"
MARKER_LENGTH_M = 0.170

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
    rvec, _ = cv2.Rodrigues(np.asarray(R, dtype=np.float64))
    return rvec.reshape(3)


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


def parse_best_model_name():
    rows = read_csv(INSPECT_SUMMARY)
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


def parse_cameras_txt(path: Path):
    cams = {}

    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        parts = line.split()
        camera_id = int(parts[0])
        model = parts[1]
        width = int(parts[2])
        height = int(parts[3])
        params = [float(x) for x in parts[4:]]

        if model == "PINHOLE":
            fx, fy, cx, cy = params[:4]
            K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
            D = np.zeros(5, dtype=np.float64)
        elif model == "SIMPLE_PINHOLE":
            f, cx, cy = params[:3]
            K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float64)
            D = np.zeros(5, dtype=np.float64)
        elif model == "SIMPLE_RADIAL":
            f, cx, cy, k = params[:4]
            K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float64)
            D = np.array([k, 0, 0, 0, 0], dtype=np.float64)
        elif model == "RADIAL":
            f, cx, cy, k1, k2 = params[:5]
            K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float64)
            D = np.array([k1, k2, 0, 0, 0], dtype=np.float64)
        elif model == "OPENCV":
            fx, fy, cx, cy, k1, k2, p1, p2 = params[:8]
            K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
            D = np.array([k1, k2, p1, p2, 0], dtype=np.float64)
        else:
            raise RuntimeError(f"Unsupported COLMAP camera model: {model}")

        cams[camera_id] = {
            "camera_id": camera_id,
            "model": model,
            "width": width,
            "height": height,
            "params": params,
            "K": K,
            "D": D,
        }

    return cams


def parse_images_txt(path: Path):
    images = {}

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

            images[name] = {
                "image_id": image_id,
                "camera_id": camera_id,
                "qvec": qvec,
                "tvec": tvec,
                "R_cw": R_cw,
                "T_cam_col": T_cam_col,
                "T_col_cam": T_col_cam,
            }
            i += 2
        else:
            i += 1

    return images


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


def detect_ref14_observations(images, min_area_px2):
    detect = make_aruco_detector()
    obs = []

    for image_name in sorted(images.keys()):
        image_path = IMAGE_DIR / image_name
        img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if img is None:
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = detect(gray)

        if ids is None or len(ids) == 0:
            continue

        ids = ids.reshape(-1)
        for idx, marker_id in enumerate(ids.tolist()):
            if int(marker_id) != REF_MARKER_ID:
                continue

            pts = np.asarray(corners[idx], dtype=np.float64).reshape(4, 2)
            area = float(cv2.contourArea(pts.astype(np.float32)))

            if area < min_area_px2:
                continue

            for corner_idx in range(4):
                obs.append({
                    "image_name": image_name,
                    "corner_idx": corner_idx,
                    "u": float(pts[corner_idx, 0]),
                    "v": float(pts[corner_idx, 1]),
                    "area_px2": area,
                })

    return obs


def undistort_to_normalized(u, v, K, D):
    pts = np.array([[[float(u), float(v)]]], dtype=np.float64)
    und = cv2.undistortPoints(pts, K, D)
    return float(und[0, 0, 0]), float(und[0, 0, 1])


def triangulate_dlt(observations, images, cameras):
    A = []

    for o in observations:
        im = images[o["image_name"]]
        cam = cameras[im["camera_id"]]

        x, y = undistort_to_normalized(o["u"], o["v"], cam["K"], cam["D"])

        P = np.zeros((3, 4), dtype=np.float64)
        P[:3, :3] = im["T_cam_col"][:3, :3]
        P[:3, 3] = im["T_cam_col"][:3, 3]

        A.append(x * P[2, :] - P[0, :])
        A.append(y * P[2, :] - P[1, :])

    A = np.asarray(A, dtype=np.float64)
    _, _, Vt = np.linalg.svd(A)
    Xh = Vt[-1, :]
    if abs(Xh[3]) < 1e-12:
        raise RuntimeError("Triangulation produced invalid homogeneous point")
    X = Xh[:3] / Xh[3]
    return X.astype(np.float64)


def project_colmap_point(X, image_payload, camera_payload):
    X = np.asarray(X, dtype=np.float64).reshape(3)
    T = image_payload["T_cam_col"]
    Xc = T[:3, :3] @ X + T[:3, 3]

    if Xc[2] <= 1e-9:
        return None, False

    rvec = R_to_rvec(T[:3, :3])
    tvec = T[:3, 3].reshape(3, 1)
    pts, _ = cv2.projectPoints(
        X.reshape(1, 3),
        rvec.reshape(3, 1),
        tvec,
        camera_payload["K"],
        camera_payload["D"],
    )
    u, v = pts.reshape(2)
    return np.array([float(u), float(v)], dtype=np.float64), True


def reproj_errors_px(X, observations, images, cameras):
    errs = []
    for o in observations:
        im = images[o["image_name"]]
        cam = cameras[im["camera_id"]]
        p, ok = project_colmap_point(X, im, cam)
        if not ok:
            errs.append(float("inf"))
            continue
        q = np.array([o["u"], o["v"]], dtype=np.float64)
        errs.append(float(np.linalg.norm(p - q)))
    return errs


def camera_center(image_payload):
    return image_payload["T_col_cam"][:3, 3]


def ray_world_from_observation(o, images, cameras):
    im = images[o["image_name"]]
    cam = cameras[im["camera_id"]]
    x, y = undistort_to_normalized(o["u"], o["v"], cam["K"], cam["D"])
    ray_cam = np.array([x, y, 1.0], dtype=np.float64)
    ray_cam /= np.linalg.norm(ray_cam)
    R_cw = im["T_cam_col"][:3, :3]
    ray_w = R_cw.T @ ray_cam
    ray_w /= np.linalg.norm(ray_w)
    return ray_w


def pair_baseline_angle_deg(o1, o2, images, cameras):
    r1 = ray_world_from_observation(o1, images, cameras)
    r2 = ray_world_from_observation(o2, images, cameras)
    arg = clamp(float(np.dot(r1, r2)))
    return math.degrees(math.acos(arg))


def robust_triangulate_corner(corner_obs, images, cameras, ransac_iters, reproj_thresh_px, min_inliers):
    if len(corner_obs) < 2:
        raise RuntimeError("Need at least two observations per corner")

    rng = random.Random(7)
    best = None

    pairs = []
    for i in range(len(corner_obs)):
        for j in range(i + 1, len(corner_obs)):
            a = pair_baseline_angle_deg(corner_obs[i], corner_obs[j], images, cameras)
            pairs.append((a, i, j))

    pairs = sorted(pairs, reverse=True)

    if not pairs:
        raise RuntimeError("No observation pairs for triangulation")

    # Prefer pairs with larger triangulation angle.
    candidate_pairs = pairs[:min(len(pairs), 200)]

    for _ in range(ransac_iters):
        _, i, j = rng.choice(candidate_pairs)
        sample = [corner_obs[i], corner_obs[j]]

        try:
            X = triangulate_dlt(sample, images, cameras)
        except Exception:
            continue

        errs = reproj_errors_px(X, corner_obs, images, cameras)
        inlier_idx = [k for k, e in enumerate(errs) if math.isfinite(e) and e <= reproj_thresh_px]

        score = (len(inlier_idx), -median([errs[k] for k in inlier_idx]) if inlier_idx else -1e9)

        if best is None or score > best["score"]:
            best = {
                "X": X,
                "inlier_idx": inlier_idx,
                "errs": errs,
                "score": score,
            }

    if best is None or len(best["inlier_idx"]) < min_inliers:
        # Fallback: all observations.
        X = triangulate_dlt(corner_obs, images, cameras)
        errs = reproj_errors_px(X, corner_obs, images, cameras)
        inlier_idx = [i for i, e in enumerate(errs) if math.isfinite(e)]
    else:
        inliers = [corner_obs[i] for i in best["inlier_idx"]]
        X = triangulate_dlt(inliers, images, cameras)
        errs = reproj_errors_px(X, corner_obs, images, cameras)
        inlier_idx = [i for i, e in enumerate(errs) if math.isfinite(e) and e <= reproj_thresh_px]

    inlier_errs = [errs[i] for i in inlier_idx]

    return {
        "X": X,
        "inlier_idx": inlier_idx,
        "all_errors": errs,
        "inlier_count": len(inlier_idx),
        "obs_count": len(corner_obs),
        "mean_reproj_px": mean(inlier_errs) if inlier_errs else float("nan"),
        "median_reproj_px": median(inlier_errs) if inlier_errs else float("nan"),
        "max_reproj_px": max(inlier_errs) if inlier_errs else float("nan"),
    }


def ideal_ref14_corners():
    s = MARKER_LENGTH_M / 2.0
    return np.array([
        [-s,  s, 0.0],
        [ s,  s, 0.0],
        [ s, -s, 0.0],
        [-s, -s, 0.0],
    ], dtype=np.float64)


def umeyama_similarity(X, Y):
    # Estimate Y ~= scale * R * X + t.
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)

    if X.shape != Y.shape or X.shape[0] < 3 or X.shape[1] != 3:
        raise RuntimeError(f"Need Nx3 arrays with N>=3, got {X.shape}, {Y.shape}")

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


def apply_sim3_pose(T_col_cam, scale, R_ref_col, t_ref_col):
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R_ref_col @ T_col_cam[:3, :3]
    T[:3, 3] = scale * (R_ref_col @ T_col_cam[:3, 3]) + t_ref_col
    return T


def apply_sim3_point(p_col, scale, R_ref_col, t_ref_col):
    p_col = np.asarray(p_col, dtype=np.float64).reshape(3)
    return scale * (R_ref_col @ p_col) + t_ref_col


def source_id_from_image_name(name):
    if name.startswith("static_") and name.endswith(".png"):
        return name[len("static_"):-len(".png")]
    if name.startswith("moving_") and name.endswith(".png"):
        return name[len("moving_"):-len(".png")]
    return name


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


def format_table(rows, headers, keys):
    data = [headers]
    for r in rows:
        data.append([str(r.get(k, "")) for k in keys])
    widths = [max(len(row[i]) for row in data) for i in range(len(headers))]
    out = []
    out.append(" | ".join(data[0][i].ljust(widths[i]) for i in range(len(headers))))
    out.append("-+-".join("-" * widths[i] for i in range(len(headers))))
    for row in data[1:]:
        out.append(" | ".join(row[i].ljust(widths[i]) for i in range(len(headers))))
    return "\n".join(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-area-px2", type=float, default=1000.0)
    parser.add_argument("--reproj-thresh-px", type=float, default=5.0)
    parser.add_argument("--ransac-iters", type=int, default=1000)
    parser.add_argument("--min-inliers", type=int, default=4)
    args = parser.parse_args()

    ensure_dir(OUT_ROOT)
    ensure_dir(AP3_CMP)
    ensure_dir(COMBINED)

    best_model = parse_best_model_name()
    model_dir = TXT_ROOT / best_model

    if not model_dir.exists():
        raise RuntimeError(f"Missing COLMAP model dir: {model_dir}")

    cameras = parse_cameras_txt(model_dir / "cameras.txt")
    images = parse_images_txt(model_dir / "images.txt")
    points3d = parse_points3D_txt(model_dir / "points3D.txt")

    obs = detect_ref14_observations(images, args.min_area_px2)

    if len(obs) < 8:
        raise RuntimeError(
            f"Too few Ref14 corner observations after min-area filtering. "
            f"Got {len(obs)}. Try --min-area-px2 400"
        )

    write_csv(
        OUT_ROOT / "ap03_ref14_corner_observations.csv",
        obs,
        ["image_name", "corner_idx", "u", "v", "area_px2"],
    )

    corner_results = []
    X_col = []

    for corner_idx in range(4):
        corner_obs = [o for o in obs if int(o["corner_idx"]) == corner_idx]
        res = robust_triangulate_corner(
            corner_obs,
            images,
            cameras,
            args.ransac_iters,
            args.reproj_thresh_px,
            args.min_inliers,
        )
        X = res["X"]
        X_col.append(X)

        corner_results.append({
            "corner_idx": corner_idx,
            "x_colmap": X[0],
            "y_colmap": X[1],
            "z_colmap": X[2],
            "obs_count": res["obs_count"],
            "inlier_count": res["inlier_count"],
            "mean_reproj_px": res["mean_reproj_px"],
            "median_reproj_px": res["median_reproj_px"],
            "max_reproj_px": res["max_reproj_px"],
        })

    X_col = np.asarray(X_col, dtype=np.float64)
    Y_ref = ideal_ref14_corners()

    scale, R_ref_col, t_ref_col = umeyama_similarity(X_col, Y_ref)

    corner_fit_rows = []
    fit_errors_cm = []

    for i in range(4):
        p_est = apply_sim3_point(X_col[i], scale, R_ref_col, t_ref_col)
        p_gt = Y_ref[i]
        err_cm = 100.0 * float(np.linalg.norm(p_est - p_gt))
        fit_errors_cm.append(err_cm)

        row = dict(corner_results[i])
        row.update({
            "ideal_x_ref_m": p_gt[0],
            "ideal_y_ref_m": p_gt[1],
            "ideal_z_ref_m": p_gt[2],
            "registered_x_ref_m": p_est[0],
            "registered_y_ref_m": p_est[1],
            "registered_z_ref_m": p_est[2],
            "corner_fit_error_cm": err_cm,
        })
        corner_fit_rows.append(row)

    write_csv(
        OUT_ROOT / "ap03_triangulated_ref14_corners_colmap_and_registered.csv",
        corner_fit_rows,
        [
            "corner_idx",
            "x_colmap", "y_colmap", "z_colmap",
            "obs_count", "inlier_count",
            "mean_reproj_px", "median_reproj_px", "max_reproj_px",
            "ideal_x_ref_m", "ideal_y_ref_m", "ideal_z_ref_m",
            "registered_x_ref_m", "registered_y_ref_m", "registered_z_ref_m",
            "corner_fit_error_cm",
        ],
    )

    static_rows = []
    moving_rows = []

    for image_name, payload in sorted(images.items()):
        T_ref_cam = apply_sim3_pose(payload["T_col_cam"], scale, R_ref_col, t_ref_col)

        if image_name.startswith("static_"):
            static_rows.append(
                pose_row("static_camera", source_id_from_image_name(image_name), T_ref_cam, "ap03_repo_like_triangulated_ref14")
            )
        elif image_name.startswith("moving_"):
            moving_rows.append(
                pose_row("moving_frame", source_id_from_image_name(image_name), T_ref_cam, "ap03_repo_like_triangulated_ref14")
            )

    write_csv(OUT_ROOT / "ap03_static_camera_poses_ref_aruco.csv", static_rows, pose_fields())
    write_csv(OUT_ROOT / "ap03_moving_frame_poses_ref_aruco.csv", moving_rows, pose_fields())

    point_rows = []
    for p in points3d:
        p_ref = apply_sim3_point(
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
        OUT_ROOT / "ap03_sparse_points3d_ref_aruco.csv",
        point_rows,
        ["point3d_id", "x_ref_aruco_m", "y_ref_aruco_m", "z_ref_aruco_m", "r", "g", "b", "colmap_reprojection_error"],
    )

    gt = gt_static_camera_poses_ref_aruco()
    est = {r["entity_id"]: r for r in static_rows}

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
            "approach": "AP03_targetless_colmap_repo_like_aruco_scale",
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
        "approach", "entity_type", "entity_id", "reference_frame",
        "translation_error_cm", "rotation_error_deg",
        "delta_x_cm", "delta_y_cm", "delta_z_cm",
        "est_ref_aruco_x_m", "est_ref_aruco_y_m", "est_ref_aruco_z_m",
        "est_ref_aruco_roll_deg", "est_ref_aruco_pitch_deg", "est_ref_aruco_yaw_deg",
        "est_ref_aruco_rvec_x", "est_ref_aruco_rvec_y", "est_ref_aruco_rvec_z",
        "gt_ref_aruco_x_m", "gt_ref_aruco_y_m", "gt_ref_aruco_z_m",
        "gt_ref_aruco_roll_deg", "gt_ref_aruco_pitch_deg", "gt_ref_aruco_yaw_deg",
        "gt_ref_aruco_rvec_x", "gt_ref_aruco_rvec_y", "gt_ref_aruco_rvec_z",
    ]

    write_csv(OUT_ROOT / "ap03_static_cameras_ref_aruco_vs_gt.csv", eval_rows, eval_fields)
    write_csv(AP3_CMP / "ap03_static_cameras_ref_aruco_vs_gt_repo_like.csv", eval_rows, eval_fields)

    cam_t = [r["translation_error_cm"] for r in eval_rows]
    cam_r = [r["rotation_error_deg"] for r in eval_rows]

    summary = [{
        "approach": "AP03_targetless_colmap_repo_like_aruco_scale",
        "reference_frame": f"aruco_marker_{REF_MARKER_ID}",
        "final_variant": "colmap_then_triangulated_ref14_corner_sim3",
        "registered_images": len(images),
        "registered_static_cameras": len(static_rows),
        "registered_moving_frames": len(moving_rows),
        "num_sparse_points3d": len(point_rows),
        "ref14_corner_observation_count": len(obs),
        "ref14_unique_anchor_images": len(set(o["image_name"] for o in obs)),
        "min_area_px2": args.min_area_px2,
        "estimated_colmap_to_metric_scale": scale,
        "corner_fit_mean_error_cm": mean(fit_errors_cm),
        "corner_fit_median_error_cm": median(fit_errors_cm),
        "corner_triangulation_mean_reproj_px": mean([r["mean_reproj_px"] for r in corner_results]),
        "corner_triangulation_median_reproj_px": median([r["median_reproj_px"] for r in corner_results]),
        "camera_mean_translation_error_cm": mean(cam_t),
        "camera_median_translation_error_cm": median(cam_t),
        "camera_mean_rotation_error_deg": mean(cam_r),
        "camera_median_rotation_error_deg": median(cam_r),
    }]

    write_csv(COMBINED / "ap03_repo_like_final_summary.csv", summary, list(summary[0].keys()))

    cam_simple = []
    for r in eval_rows:
        cam_simple.append({
            "camera": r["entity_id"],
            "translation_error_cm": f"{float(r['translation_error_cm']):.3f}",
            "rotation_error_deg": f"{float(r['rotation_error_deg']):.3f}",
            "delta_x_cm": f"{float(r['delta_x_cm']):.3f}",
            "delta_y_cm": f"{float(r['delta_y_cm']):.3f}",
            "delta_z_cm": f"{float(r['delta_z_cm']):.3f}",
            "estimated_xyz_m": f"({float(r['est_ref_aruco_x_m']):+.3f}, {float(r['est_ref_aruco_y_m']):+.3f}, {float(r['est_ref_aruco_z_m']):+.3f})",
            "gt_xyz_m": f"({float(r['gt_ref_aruco_x_m']):+.3f}, {float(r['gt_ref_aruco_y_m']):+.3f}, {float(r['gt_ref_aruco_z_m']):+.3f})",
        })

    cam_table = format_table(
        cam_simple,
        ["camera", "t_err_cm", "r_err_deg", "dX", "dY", "dZ", "estimated xyz [m]", "GT xyz [m]"],
        ["camera", "translation_error_cm", "rotation_error_deg", "delta_x_cm", "delta_y_cm", "delta_z_cm", "estimated_xyz_m", "gt_xyz_m"],
    )

    corner_simple = []
    for r in corner_fit_rows:
        corner_simple.append({
            "corner": r["corner_idx"],
            "obs": r["obs_count"],
            "inliers": r["inlier_count"],
            "mean_reproj_px": f"{float(r['mean_reproj_px']):.3f}",
            "median_reproj_px": f"{float(r['median_reproj_px']):.3f}",
            "fit_error_cm": f"{float(r['corner_fit_error_cm']):.4f}",
            "registered_xyz_m": f"({float(r['registered_x_ref_m']):+.4f}, {float(r['registered_y_ref_m']):+.4f}, {float(r['registered_z_ref_m']):+.4f})",
            "ideal_xyz_m": f"({float(r['ideal_x_ref_m']):+.4f}, {float(r['ideal_y_ref_m']):+.4f}, {float(r['ideal_z_ref_m']):+.4f})",
        })

    corner_table = format_table(
        corner_simple,
        ["corner", "obs", "inliers", "mean_repr", "med_repr", "fit_cm", "registered xyz [m]", "ideal xyz [m]"],
        ["corner", "obs", "inliers", "mean_reproj_px", "median_reproj_px", "fit_error_cm", "registered_xyz_m", "ideal_xyz_m"],
    )

    report = f"""AP03 FINAL REPO-LIKE SCALE REGISTRATION REPORT
=============================================

Approach:
- AP03: Targetless COLMAP / SfM + ArUco-based metric scale registration
- COLMAP reconstruction is already done and uses natural image features.
- This step follows the aruco-estimator idea:
  reconstruct/triangulate Ref-ArUco marker corners in COLMAP coordinates,
  then map them to ideal metric marker corners with known side length.

Important distinction:
- AP01 uses ArUco geometry inside a target-based relay calibration pipeline.
- AP02 uses ArUco geometry directly in every PnP/BA marker observation.
- AP03 uses COLMAP as the main estimator and uses ArUco geometry only afterwards
  to scale/register the arbitrary COLMAP model.

Input COLMAP model:
- model: {best_model}
- registered images: {len(images)}
- registered static cameras: {len(static_rows)} / 4
- registered moving frames: {len(moving_rows)}
- sparse 3D points: {len(point_rows)}

Ref-ArUco scale registration:
- reference marker id: {REF_MARKER_ID}
- marker length: {MARKER_LENGTH_M:.3f} m
- min marker area used: {args.min_area_px2:.1f} px^2
- Ref14 corner observations: {len(obs)}
- Ref14 anchor images: {len(set(o['image_name'] for o in obs))}
- estimated COLMAP-to-meter scale: {scale:.9f}

Triangulated Ref14 corner quality:
- corner fit mean error: {mean(fit_errors_cm):.6f} cm
- corner fit median error: {median(fit_errors_cm):.6f} cm
- mean corner reprojection error: {mean([r['mean_reproj_px'] for r in corner_results]):.6f} px
- median corner reprojection error: {median([r['median_reproj_px'] for r in corner_results]):.6f} px

{corner_table}

Static camera GT evaluation in Ref-ArUco frame:
- mean translation error: {mean(cam_t):.6f} cm
- median translation error: {median(cam_t):.6f} cm
- mean rotation error: {mean(cam_r):.6f} deg
- median rotation error: {median(cam_r):.6f} deg

{cam_table}

Output files:
- {OUT_ROOT / "ap03_triangulated_ref14_corners_colmap_and_registered.csv"}
- {OUT_ROOT / "ap03_static_camera_poses_ref_aruco.csv"}
- {OUT_ROOT / "ap03_static_cameras_ref_aruco_vs_gt.csv"}
- {AP3_CMP / "ap03_static_cameras_ref_aruco_vs_gt_repo_like.csv"}
- {COMBINED / "ap03_repo_like_final_summary.csv"}

Interpretation:
- This is the cleaner AP03 scale fix compared to the earlier PnP-anchor Sim(3) attempt.
- The scale is estimated from the reconstructed Ref-ArUco marker geometry itself.
- If translation error is still high, the issue is likely COLMAP pose/geometry bias or weak corner triangulation,
  not simply missing metric scale.
"""

    (OUT_ROOT / "AP03_FINAL_REPO_LIKE_SCALE_REGISTRATION_REPORT.txt").write_text(report)
    (AP3_CMP / "AP03_FINAL_REPO_LIKE_SCALE_REGISTRATION_REPORT.txt").write_text(report)

    metadata = {
        "approach": "AP03_targetless_colmap_repo_like_aruco_scale",
        "best_colmap_model": best_model,
        "registered_images": len(images),
        "registered_static_cameras": len(static_rows),
        "registered_moving_frames": len(moving_rows),
        "num_sparse_points3d": len(point_rows),
        "reference_marker_id": REF_MARKER_ID,
        "marker_length_m": MARKER_LENGTH_M,
        "min_area_px2": args.min_area_px2,
        "reproj_thresh_px": args.reproj_thresh_px,
        "scale_colmap_to_metric": scale,
        "corner_fit_errors_cm": fit_errors_cm,
        "camera_mean_translation_error_cm": mean(cam_t),
        "camera_mean_rotation_error_deg": mean(cam_r),
    }
    (OUT_ROOT / "ap03_repo_like_registration_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

    print(report)


if __name__ == "__main__":
    main()
