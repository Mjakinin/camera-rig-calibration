#!/usr/bin/env python3

import csv
import os
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np


AP01 = Path(os.environ.get("AP01_DIR", "results/bus_real_data/01_marker_direct_relay_multimarker_multichain/07_final_extrinsics_cam3_reference"))
WORLD_SDF = Path("src/calib_lab/bus_real_data/worlds/bus_real_data_moving_camera.sdf")
OBS_CSV = Path(os.environ.get("AP01_OBS_CSV", "results/bus_real_data/00_shared_baseline/bus_real_data_ref_marker_v1/aruco_observations/shared_static_aruco_observations.csv"))

if os.environ.get("AP01_SOURCE_CSV"):
    SRC_CANDIDATES = [Path(os.environ["AP01_SOURCE_CSV"])]
else:
    SRC_CANDIDATES = [
        AP01 / "final_extrinsics_summary.csv",
        AP01 / "99_ARCHIVE_EXTRA_OUTPUTS" / "final_extrinsics_summary.csv",
    ]

OUT_DIR = Path(os.environ.get("AP01_OUT_DIR", str(AP01)))
OUT_TXT = OUT_DIR / "AP01_FINAL_RESULT.txt"
OUT_CSV = OUT_DIR / "AP01_FINAL_RESULT.csv"
OUT_TXT.parent.mkdir(parents=True, exist_ok=True)

ROOT_CAM = "cam_edge_3"
STATIC_CAMS = ["cam_edge_0", "cam_edge_1", "cam_edge_3", "cam_edge_5"]
TARGET_CAMS = ["cam_edge_1", "cam_edge_0", "cam_edge_5"]
REF_MARKER_ID = 14
REF_MARKER_ENTITY = "aruco_ref_floor_14"

R_MODEL_FROM_OPENCV_MARKER = np.array([
    [1.0, 0.0,  0.0],
    [0.0, 0.0, -1.0],
    [0.0, 1.0,  0.0],
], dtype=np.float64)


def read_csv(path):
    with Path(path).open(newline="") as fp:
        return list(csv.DictReader(fp))


def write_csv(path, rows, fields):
    with Path(path).open("w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def clamp(x):
    return max(-1.0, min(1.0, float(x)))


def rpy_to_R(roll, pitch, yaw):
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    Rx = np.array([[1,0,0],[0,cr,-sr],[0,sr,cr]], dtype=float)
    Ry = np.array([[cp,0,sp],[0,1,0],[-sp,0,cp]], dtype=float)
    Rz = np.array([[cy,-sy,0],[sy,cy,0],[0,0,1]], dtype=float)
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


def make_T(R, t):
    T = np.eye(4)
    T[:3, :3] = np.asarray(R, dtype=float)
    T[:3, 3] = np.asarray(t, dtype=float).reshape(3)
    return T


def invT(T):
    R = T[:3, :3]
    t = T[:3, 3]
    out = np.eye(4)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ t
    return out


def trans_error_cm(T_est, T_gt):
    return 100.0 * float(np.linalg.norm((invT(T_gt) @ T_est)[:3, 3]))


def rot_error_deg(T_est, T_gt):
    dT = invT(T_gt) @ T_est
    arg = clamp((float(np.trace(dT[:3, :3])) - 1.0) / 2.0)
    return math.degrees(math.acos(arg))


def R_to_rpy_deg(R):
    pitch = math.atan2(-R[2, 0], math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2))
    roll = math.atan2(R[2, 1], R[2, 2])
    yaw = math.atan2(R[1, 0], R[0, 0])
    return [math.degrees(roll), math.degrees(pitch), math.degrees(yaw)]


def T_from_summary_row(row, prefix):
    x = float(row[f"{prefix}_x_m"])
    y = float(row[f"{prefix}_y_m"])
    z = float(row[f"{prefix}_z_m"])
    roll = math.radians(float(row[f"{prefix}_roll_deg"]))
    pitch = math.radians(float(row[f"{prefix}_pitch_deg"]))
    yaw = math.radians(float(row[f"{prefix}_yaw_deg"]))
    return make_T(rpy_to_R(roll, pitch, yaw), [x, y, z])


def T_from_pnp_observation(row):
    rvec = np.array([float(row["rvec_x"]), float(row["rvec_y"]), float(row["rvec_z"])], dtype=float)
    tvec = np.array([float(row["tvec_x_m"]), float(row["tvec_y_m"]), float(row["tvec_z_m"])], dtype=float)
    return make_T(rvec_to_R(rvec), tvec)  # T_camera_marker


def parse_sdf_poses():
    tree = ET.parse(WORLD_SDF)
    root = tree.getroot()
    poses = {}

    for model in root.iter("model"):
        name = model.attrib.get("name", "").strip()
        pose_el = model.find("pose")
        if not name or pose_el is None or not pose_el.text:
            continue
        vals = [float(x) for x in pose_el.text.split()]
        x, y, z, roll, pitch, yaw = vals[:6]
        poses[name] = make_T(rpy_to_R(roll, pitch, yaw), [x, y, z])

    for inc in root.iter("include"):
        name_el = inc.find("name")
        pose_el = inc.find("pose")
        if name_el is None or pose_el is None or not name_el.text or not pose_el.text:
            continue
        name = name_el.text.strip()
        vals = [float(x) for x in pose_el.text.split()]
        x, y, z, roll, pitch, yaw = vals[:6]
        poses[name] = make_T(rpy_to_R(roll, pitch, yaw), [x, y, z])

    return poses


def camera_model_to_optical(T_W_model):
    R_opt_to_link = rpy_to_R(0.0, -math.pi / 2.0, math.pi / 2.0)
    T_opt_to_link = make_T(R_opt_to_link, [0, 0, 0])
    return T_W_model @ invT(T_opt_to_link)


def marker_model_to_opencv(T_W_model):
    T_model_cvmarker = make_T(R_MODEL_FROM_OPENCV_MARKER, [0, 0, 0])
    return T_W_model @ T_model_cvmarker


def f(x, nd=3):
    try:
        return f"{float(x):.{nd}f}"
    except Exception:
        return str(x)


def mean(xs):
    xs = [float(x) for x in xs]
    return sum(xs) / len(xs) if xs else 0.0


def median(xs):
    xs = sorted(float(x) for x in xs)
    if not xs:
        return 0.0
    n = len(xs)
    return xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2])


def md_table(headers, rows):
    widths = [len(h) for h in headers]
    for row in rows:
        for i, c in enumerate(row):
            widths[i] = max(widths[i], len(str(c)))
    out = []
    out.append(" | ".join(str(h).ljust(widths[i]) for i, h in enumerate(headers)))
    out.append(" | ".join("-" * widths[i] for i in range(len(headers))))
    for row in rows:
        out.append(" | ".join(str(c).ljust(widths[i]) for i, c in enumerate(row)))
    return "\n".join(out)


src = next((p for p in SRC_CANDIDATES if p.exists()), None)
if src is None:
    raise SystemExit(
        "Missing AP01 source final_extrinsics_summary.csv. "
        "Run AP01 pipeline again or restore it from 99_ARCHIVE_EXTRA_OUTPUTS."
    )
if not OBS_CSV.exists():
    raise SystemExit(f"Missing shared static ArUco observations: {OBS_CSV}")

summary_rows = [r for r in read_csv(src) if r.get("category") == "main_no_gt"]
by_cam = {r["target_camera"]: r for r in summary_rows}

missing = [c for c in TARGET_CAMS if c not in by_cam]
if missing:
    raise SystemExit(f"Missing AP01 target rows: {missing}")

# AP01 local camera rig: T_cam3_cam
T_cam3_cam = {ROOT_CAM: np.eye(4)}
for cam in TARGET_CAMS:
    T_cam3_cam[cam] = T_from_summary_row(by_cam[cam], "est")

# Find usable Ref14 observation from static cameras.
obs_rows = []
for r in read_csv(OBS_CSV):
    if str(r.get("observer_type")) != "static":
        continue
    if str(r.get("marker_id")) not in {"14", "14.0"}:
        continue
    if str(r.get("pnp_success")).lower() not in {"true", "1"}:
        continue
    cam = r.get("camera_name") or r.get("observer_id")
    if cam not in STATIC_CAMS:
        continue
    if cam not in T_cam3_cam:
        continue
    obs_rows.append(r)

if not obs_rows:
    raise SystemExit(
        "No static Ref14 observation found for AP01 ref-origin output. "
        "AP01 cannot produce deployable Ref14-origin poses without a measured Ref14-to-camera anchor."
    )

# Prefer root camera if it sees Ref14; otherwise take largest marker area.
root_obs = [r for r in obs_rows if (r.get("camera_name") or r.get("observer_id")) == ROOT_CAM]
if root_obs:
    anchor = max(root_obs, key=lambda r: float(r.get("area_px2") or 0.0))
else:
    anchor = max(obs_rows, key=lambda r: float(r.get("area_px2") or 0.0))

anchor_cam = anchor.get("camera_name") or anchor.get("observer_id")

T_anchor_ref14 = T_from_pnp_observation(anchor)      # T_anchor_camera_ref14_marker
T_ref14_anchor = invT(T_anchor_ref14)                # T_ref14_marker_anchor_camera
T_anchor_cam3 = invT(T_cam3_cam[anchor_cam])         # T_anchor_camera_cam3_camera
T_ref14_cam3 = T_ref14_anchor @ T_anchor_cam3        # T_ref14_marker_cam3_camera

# Estimated Ref14-origin camera poses.
T_ref14_cam_est = {}
for cam in STATIC_CAMS:
    T_ref14_cam_est[cam] = T_ref14_cam3 @ T_cam3_cam[cam]

# GT Ref14-origin camera poses.
sdf = parse_sdf_poses()
if REF_MARKER_ENTITY not in sdf:
    raise SystemExit(f"Missing GT marker pose in SDF: {REF_MARKER_ENTITY}")

T_W_ref14_gt = marker_model_to_opencv(sdf[REF_MARKER_ENTITY])
T_ref14_W_gt = invT(T_W_ref14_gt)

T_ref14_cam_gt = {}
for cam in STATIC_CAMS:
    if cam not in sdf:
        raise SystemExit(f"Missing GT camera pose in SDF: {cam}")
    T_W_cam_gt = camera_model_to_optical(sdf[cam])
    T_ref14_cam_gt[cam] = T_ref14_W_gt @ T_W_cam_gt

rows = []
for cam in STATIC_CAMS:
    T_est = T_ref14_cam_est[cam]
    T_gt = T_ref14_cam_gt[cam]
    dxyz = 100.0 * (T_est[:3, 3] - T_gt[:3, 3])
    rpy_est = R_to_rpy_deg(T_est[:3, :3])
    rpy_gt = R_to_rpy_deg(T_gt[:3, :3])

    method = "root_camera_chained_from_ref14_anchor" if cam == ROOT_CAM else by_cam[cam]["method"]
    rows.append({
        "approach": "AP01_marker_direct_relay_multichain",
        "evaluation": "ref14_origin_deployable_camera_rig",
        "entity_type": "static_camera",
        "entity_id": cam,
        "anchor_camera_for_ref14": anchor_cam,
        "anchor_marker_id": REF_MARKER_ID,
        "method": method,
        "translation_error_cm": trans_error_cm(T_est, T_gt),
        "rotation_error_deg": rot_error_deg(T_est, T_gt),
        "delta_x_cm": dxyz[0],
        "delta_y_cm": dxyz[1],
        "delta_z_cm": dxyz[2],
        "est_ref14_x_m": T_est[0, 3],
        "est_ref14_y_m": T_est[1, 3],
        "est_ref14_z_m": T_est[2, 3],
        "gt_ref14_x_m": T_gt[0, 3],
        "gt_ref14_y_m": T_gt[1, 3],
        "gt_ref14_z_m": T_gt[2, 3],
        "est_ref14_roll_deg": rpy_est[0],
        "est_ref14_pitch_deg": rpy_est[1],
        "est_ref14_yaw_deg": rpy_est[2],
        "gt_ref14_roll_deg": rpy_gt[0],
        "gt_ref14_pitch_deg": rpy_gt[1],
        "gt_ref14_yaw_deg": rpy_gt[2],
        "note": "AP01 remains cam3-rooted internally; Ref14 origin is obtained from a measured static ArUco/PnP anchor, not GT.",
    })

fields = [
    "approach", "evaluation", "entity_type", "entity_id",
    "anchor_camera_for_ref14", "anchor_marker_id", "method",
    "translation_error_cm", "rotation_error_deg",
    "delta_x_cm", "delta_y_cm", "delta_z_cm",
    "est_ref14_x_m", "est_ref14_y_m", "est_ref14_z_m",
    "gt_ref14_x_m", "gt_ref14_y_m", "gt_ref14_z_m",
    "est_ref14_roll_deg", "est_ref14_pitch_deg", "est_ref14_yaw_deg",
    "gt_ref14_roll_deg", "gt_ref14_pitch_deg", "gt_ref14_yaw_deg",
    "note",
]
write_csv(OUT_CSV, rows, fields)

summary_rows = [[
    "AP01 Ref14-origin static cameras",
    len(rows),
    f(mean([r["translation_error_cm"] for r in rows])),
    f(median([r["translation_error_cm"] for r in rows])),
    f(mean([r["rotation_error_deg"] for r in rows])),
    f(median([r["rotation_error_deg"] for r in rows])),
]]

cam_rows = []
for r in rows:
    cam_rows.append([
        r["entity_id"],
        r["method"],
        f(r["translation_error_cm"]),
        f(r["rotation_error_deg"]),
        f(r["delta_x_cm"]),
        f(r["delta_y_cm"]),
        f(r["delta_z_cm"]),
        f"({f(r['est_ref14_x_m'])}, {f(r['est_ref14_y_m'])}, {f(r['est_ref14_z_m'])})",
        f"({f(r['gt_ref14_x_m'])}, {f(r['gt_ref14_y_m'])}, {f(r['gt_ref14_z_m'])})",
    ])

cam3_rows = []
for cam in TARGET_CAMS:
    r = by_cam[cam]
    cam3_rows.append([
        cam,
        r["method"],
        f(r["translation_error_cm"]),
        f(r["rotation_error_deg"]),
        r.get("num_candidates", ""),
        r.get("num_inliers", ""),
        r.get("num_outliers", ""),
    ])

txt = f"""AP01 FINAL RESULT — Marker Direct Relay / Multichain
=====================================================

Method:
AP01 estimates a camera rig internally with cam_edge_3 as local root camera.
GT is NOT used for AP01 aggregation.

Final deployable Ref14-origin output:
- Ref14 -> anchor camera is estimated from a real static ArUco/PnP observation.
- Anchor camera used for Ref14 origin: {anchor_cam}
- All AP01 cam3-rooted camera transforms are chained into the Ref14 frame.
- This answers: where are all static cameras relative to marker 14?
- The old GT-based Ref14 table is not used, because it gave cam_edge_3 a fake zero error.

Summary:
{md_table(["metric", "count", "mean_t_cm", "median_t_cm", "mean_r_deg", "median_r_deg"], summary_rows)}

Static camera poses relative to Ref14:
{md_table(["camera", "method", "t_cm", "r_deg", "dX_cm", "dY_cm", "dZ_cm", "est_ref14_xyz_m", "gt_ref14_xyz_m"], cam_rows)}

Method-internal cam_edge_3-rooted details:
{md_table(["target", "method", "t_cm", "r_deg", "candidates", "inliers", "outliers"], cam3_rows)}
"""

OUT_TXT.write_text(txt)

print("[OK] wrote", OUT_TXT)
print("[OK] wrote", OUT_CSV)
print()
print(txt)
