#!/usr/bin/env python3

import csv
import json
import math
import importlib.util
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np


ROOT_CAM = "cam_edge_3"
REF14_ENTITY = "aruco_ref_floor_14"

CHAIN_SCRIPT = Path(__file__).resolve().parent / "14_eval_moving_relay_chains.py"
DIRECT_MULTI_CSV = Path("results/bus_real_data/01_marker_direct_relay_multimarker_multichain/05_direct_static_cam3_cam1_multimarker/05_multimarker_aggregate_estimates.csv")
RELAY_MULTI_CSV = Path("results/bus_real_data/01_marker_direct_relay_multimarker_multichain/06_moving_relay_chain_eval/relay_chain_multichain_aggregate.csv")
WORLD_SDF = Path("src/calib_lab/bus_real_data/worlds/bus_real_data_moving_camera.sdf")

OUT = Path("results/bus_real_data/01_marker_direct_relay_multimarker_multichain/07_final_extrinsics_cam3_reference")
OUT.mkdir(parents=True, exist_ok=True)


def load_chain_module():
    spec = importlib.util.spec_from_file_location("relay_chain_eval", CHAIN_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def clamp(x, lo=-1.0, hi=1.0):
    return max(lo, min(hi, x))


def rpy_to_R(roll, pitch, yaw):
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    Rx = np.array([[1,0,0],[0,cr,-sr],[0,sr,cr]], dtype=np.float64)
    Ry = np.array([[cp,0,sp],[0,1,0],[-sp,0,cp]], dtype=np.float64)
    Rz = np.array([[cy,-sy,0],[sy,cy,0],[0,0,1]], dtype=np.float64)
    return Rz @ Ry @ Rx


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


def R_to_rpy_deg(R):
    pitch = math.atan2(-R[2,0], math.sqrt(R[0,0] ** 2 + R[1,0] ** 2))
    roll = math.atan2(R[2,1], R[2,2])
    yaw = math.atan2(R[1,0], R[0,0])
    return [math.degrees(roll), math.degrees(pitch), math.degrees(yaw)]


def trans_error(T_est, T_gt):
    dT = invT(T_gt) @ T_est
    return float(np.linalg.norm(dT[:3, 3]))


def rot_error_deg(T_est, T_gt):
    dT = invT(T_gt) @ T_est
    arg = (np.trace(dT[:3, :3]) - 1.0) / 2.0
    return math.degrees(math.acos(clamp(float(arg))))


def T_from_row(row, prefix="estimated_target_in_root"):
    x = float(row[f"{prefix}_x"])
    y = float(row[f"{prefix}_y"])
    z = float(row[f"{prefix}_z"])
    roll = math.radians(float(row[f"{prefix}_roll_deg"]))
    pitch = math.radians(float(row[f"{prefix}_pitch_deg"]))
    yaw = math.radians(float(row[f"{prefix}_yaw_deg"]))
    return make_T(rpy_to_R(roll, pitch, yaw), np.array([x, y, z], dtype=np.float64))


def read_csv(path):
    if not path.exists():
        raise RuntimeError(f"Missing CSV: {path}")
    with path.open() as fp:
        return list(csv.DictReader(fp))


def parse_ref14_world_pose():
    tree = ET.parse(WORLD_SDF)
    root = tree.getroot()

    for inc in root.iter("include"):
        name_el = inc.find("name")
        pose_el = inc.find("pose")
        if name_el is None or pose_el is None:
            continue
        if name_el.text.strip() != REF14_ENTITY:
            continue

        vals = [float(x) for x in pose_el.text.split()]
        x, y, z, roll, pitch, yaw = vals[:6]
        return make_T(rpy_to_R(roll, pitch, yaw), np.array([x, y, z], dtype=np.float64)), vals[:6]

    raise RuntimeError(f"Could not find {REF14_ENTITY} in {WORLD_SDF}")


def pose_payload(T):
    rpy = R_to_rpy_deg(T[:3, :3])
    return {
        "translation_m": {"x": float(T[0,3]), "y": float(T[1,3]), "z": float(T[2,3])},
        "rotation_rpy_deg": {"roll": float(rpy[0]), "pitch": float(rpy[1]), "yaw": float(rpy[2])},
        "matrix_4x4": [[float(x) for x in row] for row in T],
    }


def load_direct_multimarker(m):
    rows = read_csv(DIRECT_MULTI_CSV)
    rows = [r for r in rows if r["aggregate_type"] == "weighted_mean_of_mad_inliers_no_gt_selection"]
    if not rows:
        raise RuntimeError("No weighted direct multimarker row found")

    r = rows[0]
    T_est = T_from_row(r)
    T_gt = T_from_row(r, "gt_target_in_root")

    return {
        "name": "cam_edge_3_to_cam_edge_1_direct_static_multimarker",
        "target_camera": "cam_edge_1",
        "method": "direct_static_aruco_multimarker_weighted_mad_inliers",
        "category": "main_no_gt",
        "pair": "",
        "root_marker": "",
        "target_marker": "",
        "root_frame": "",
        "target_frame": "",
        "num_candidates": r.get("num_candidates", ""),
        "num_inliers": r.get("num_inliers", ""),
        "num_outliers": r.get("num_outliers", ""),
        "T_est": T_est,
        "T_gt": T_gt,
        "translation_error_m": trans_error(T_est, T_gt),
        "rotation_error_deg": rot_error_deg(T_est, T_gt),
        "notes": "Direct static robust multi-marker aggregation. No GT candidate selection.",
    }


def load_relay_multichain(pair, method, category):
    rows = read_csv(RELAY_MULTI_CSV)
    rows = [
        r for r in rows
        if r["pair"] == pair
        and r["method"] == method
        and r["aggregate_type"] == "weighted_mean_of_mad_inliers_no_gt_selection"
    ]
    if not rows:
        raise RuntimeError(f"No relay multichain aggregate row found for {pair} / {method}")

    r = rows[0]
    T_est = T_from_row(r)
    T_gt = T_from_row(r, "gt_target_in_root")

    target = r["target_camera"]

    return {
        "name": f"cam_edge_3_to_{target}_{method}_multichain",
        "target_camera": target,
        "method": "moving_relay_multichain_colmap_motion_aruco_metric_scale" if method == "COLMAP_motion" else "moving_relay_multichain_gt_motion_oracle",
        "raw_method": method,
        "category": category,
        "pair": pair,
        "root_marker": "ALL",
        "target_marker": "ALL",
        "root_frame": "ALL",
        "target_frame": "ALL",
        "num_candidates": r.get("num_candidates", ""),
        "num_inliers": r.get("num_inliers", ""),
        "num_outliers": r.get("num_outliers", ""),
        "best_oracle_root_marker": r.get("best_oracle_root_marker", ""),
        "best_oracle_target_marker": r.get("best_oracle_target_marker", ""),
        "best_oracle_root_frame": r.get("best_oracle_root_frame", ""),
        "best_oracle_target_frame": r.get("best_oracle_target_frame", ""),
        "best_oracle_translation_error_cm": r.get("best_oracle_translation_error_cm", ""),
        "best_oracle_rotation_error_deg": r.get("best_oracle_rotation_error_deg", ""),
        "inlier_rule": r.get("inlier_rule", ""),
        "T_est": T_est,
        "T_gt": T_gt,
        "translation_error_m": trans_error(T_est, T_gt),
        "rotation_error_deg": rot_error_deg(T_est, T_gt),
        "notes": (
            "Moving relay multichain aggregation over all valid marker/frame combinations. "
            "No GT is used for aggregation; GT is evaluation only."
        ),
    }


def write_final_summary(entries):
    p = OUT / "final_extrinsics_summary.csv"
    fields = [
        "name", "target_camera", "method", "category",
        "pair", "root_marker", "target_marker", "root_frame", "target_frame",
        "num_candidates", "num_inliers", "num_outliers",
        "translation_error_cm", "rotation_error_deg",
        "est_x_m", "est_y_m", "est_z_m",
        "est_roll_deg", "est_pitch_deg", "est_yaw_deg",
        "gt_x_m", "gt_y_m", "gt_z_m",
        "gt_roll_deg", "gt_pitch_deg", "gt_yaw_deg",
        "best_oracle_root_marker", "best_oracle_target_marker",
        "best_oracle_root_frame", "best_oracle_target_frame",
        "best_oracle_translation_error_cm", "best_oracle_rotation_error_deg",
        "inlier_rule", "notes",
    ]

    with p.open("w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=fields)
        w.writeheader()

        for e in entries:
            T_est = e["T_est"]
            T_gt = e["T_gt"]
            erpy = R_to_rpy_deg(T_est[:3, :3])
            grpy = R_to_rpy_deg(T_gt[:3, :3])

            w.writerow({
                "name": e["name"],
                "target_camera": e["target_camera"],
                "method": e["method"],
                "category": e["category"],
                "pair": e.get("pair", ""),
                "root_marker": e.get("root_marker", ""),
                "target_marker": e.get("target_marker", ""),
                "root_frame": e.get("root_frame", ""),
                "target_frame": e.get("target_frame", ""),
                "num_candidates": e.get("num_candidates", ""),
                "num_inliers": e.get("num_inliers", ""),
                "num_outliers": e.get("num_outliers", ""),
                "translation_error_cm": 100.0 * e["translation_error_m"],
                "rotation_error_deg": e["rotation_error_deg"],
                "est_x_m": T_est[0,3],
                "est_y_m": T_est[1,3],
                "est_z_m": T_est[2,3],
                "est_roll_deg": erpy[0],
                "est_pitch_deg": erpy[1],
                "est_yaw_deg": erpy[2],
                "gt_x_m": T_gt[0,3],
                "gt_y_m": T_gt[1,3],
                "gt_z_m": T_gt[2,3],
                "gt_roll_deg": grpy[0],
                "gt_pitch_deg": grpy[1],
                "gt_yaw_deg": grpy[2],
                "best_oracle_root_marker": e.get("best_oracle_root_marker", ""),
                "best_oracle_target_marker": e.get("best_oracle_target_marker", ""),
                "best_oracle_root_frame": e.get("best_oracle_root_frame", ""),
                "best_oracle_target_frame": e.get("best_oracle_target_frame", ""),
                "best_oracle_translation_error_cm": e.get("best_oracle_translation_error_cm", ""),
                "best_oracle_rotation_error_deg": e.get("best_oracle_rotation_error_deg", ""),
                "inlier_rule": e.get("inlier_rule", ""),
                "notes": e.get("notes", ""),
            })

    print("[OK] wrote:", p)


def build_ref14_rows(m, entries):
    T_W_ref14, pose_vals = parse_ref14_world_pose()
    T_ref14_W = invT(T_W_ref14)

    T_W_cam3 = m.T_W_optical_from_sdf_model(ROOT_CAM)
    T_ref14_cam3_gt = T_ref14_W @ T_W_cam3

    rows = [{
        "camera": ROOT_CAM,
        "source_entry": "root_camera_eval_anchor",
        "method": "gt_eval_ref14_to_cam3_anchor",
        "category": "eval_anchor",
        "T_est": T_ref14_cam3_gt,
        "T_gt": T_ref14_cam3_gt,
        "translation_error_m": 0.0,
        "rotation_error_deg": 0.0,
        "notes": "EVAL_ONLY GT anchor for expressing cam3-root estimates in marker14 frame.",
    }]

    for e in entries:
        if e["category"] not in ["main_no_gt"]:
            continue

        T_ref14_target_est = T_ref14_cam3_gt @ e["T_est"]
        T_W_target = m.T_W_optical_from_sdf_model(e["target_camera"])
        T_ref14_target_gt = T_ref14_W @ T_W_target

        rows.append({
            "camera": e["target_camera"],
            "source_entry": e["name"],
            "method": e["method"],
            "category": e["category"],
            "T_est": T_ref14_target_est,
            "T_gt": T_ref14_target_gt,
            "translation_error_m": trans_error(T_ref14_target_est, T_ref14_target_gt),
            "rotation_error_deg": rot_error_deg(T_ref14_target_est, T_ref14_target_gt),
            "notes": "EVAL_ONLY marker14 GT frame. Calibration itself is cam3-rooted.",
        })

    meta = {
        "reference_marker_entity": REF14_ENTITY,
        "reference_marker_pose_world_sdf": {
            "x": pose_vals[0], "y": pose_vals[1], "z": pose_vals[2],
            "roll": pose_vals[3], "pitch": pose_vals[4], "yaw": pose_vals[5],
        },
        "T_ref14_cam3_gt": pose_payload(T_ref14_cam3_gt),
        "important_note": "Marker14 export is GT/evaluation-only.",
    }

    return rows, meta


def write_ref14_csv(rows):
    p = OUT / "final_camera_poses_ref14_gt_eval.csv"
    fields = [
        "camera", "source_entry", "method", "category",
        "translation_error_cm", "rotation_error_deg",
        "est_ref14_x_m", "est_ref14_y_m", "est_ref14_z_m",
        "est_ref14_roll_deg", "est_ref14_pitch_deg", "est_ref14_yaw_deg",
        "gt_ref14_x_m", "gt_ref14_y_m", "gt_ref14_z_m",
        "gt_ref14_roll_deg", "gt_ref14_pitch_deg", "gt_ref14_yaw_deg",
        "notes",
    ]

    with p.open("w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=fields)
        w.writeheader()

        for r in rows:
            T_est = r["T_est"]
            T_gt = r["T_gt"]
            erpy = R_to_rpy_deg(T_est[:3,:3])
            grpy = R_to_rpy_deg(T_gt[:3,:3])

            w.writerow({
                "camera": r["camera"],
                "source_entry": r["source_entry"],
                "method": r["method"],
                "category": r["category"],
                "translation_error_cm": 100.0 * r["translation_error_m"],
                "rotation_error_deg": r["rotation_error_deg"],
                "est_ref14_x_m": T_est[0,3],
                "est_ref14_y_m": T_est[1,3],
                "est_ref14_z_m": T_est[2,3],
                "est_ref14_roll_deg": erpy[0],
                "est_ref14_pitch_deg": erpy[1],
                "est_ref14_yaw_deg": erpy[2],
                "gt_ref14_x_m": T_gt[0,3],
                "gt_ref14_y_m": T_gt[1,3],
                "gt_ref14_z_m": T_gt[2,3],
                "gt_ref14_roll_deg": grpy[0],
                "gt_ref14_pitch_deg": grpy[1],
                "gt_ref14_yaw_deg": grpy[2],
                "notes": r["notes"],
            })

    print("[OK] wrote:", p)


def write_json(entries, ref14_rows, ref14_meta):
    p = OUT / "final_extrinsics_cam3_reference.json"
    payload = {
        "reference_camera": ROOT_CAM,
        "main_pipeline": "direct static multimarker + moving relay multichain",
        "important_note": "GT is evaluation only. Marker14 frame is evaluation-only.",
        "extrinsics_cam3_reference": [
            {
                "name": e["name"],
                "target_camera": e["target_camera"],
                "method": e["method"],
                "category": e["category"],
                "translation_error_m": e["translation_error_m"],
                "rotation_error_deg": e["rotation_error_deg"],
                "estimated_transform_cam3_to_target": pose_payload(e["T_est"]),
                "ground_truth_transform_cam3_to_target": pose_payload(e["T_gt"]),
                "num_candidates": e.get("num_candidates", ""),
                "num_inliers": e.get("num_inliers", ""),
                "num_outliers": e.get("num_outliers", ""),
                "notes": e.get("notes", ""),
            }
            for e in entries
        ],
        "marker14_gt_eval_reference": ref14_meta,
        "camera_poses_ref14_gt_eval": [
            {
                "camera": r["camera"],
                "method": r["method"],
                "translation_error_m": r["translation_error_m"],
                "rotation_error_deg": r["rotation_error_deg"],
                "estimated_pose_ref14_to_camera": pose_payload(r["T_est"]),
                "ground_truth_pose_ref14_to_camera": pose_payload(r["T_gt"]),
                "notes": r["notes"],
            }
            for r in ref14_rows
        ],
    }
    p.write_text(json.dumps(payload, indent=2) + "\n")
    print("[OK] wrote:", p)


def color(cm, deg, text):
    if cm <= 10 and deg <= 3:
        return "\033[92m" + text + "\033[0m"
    if cm <= 25 and deg <= 7:
        return "\033[93m" + text + "\033[0m"
    return "\033[91m" + text + "\033[0m"


def plain(s):
    for c in ["\033[92m", "\033[93m", "\033[91m", "\033[1m", "\033[0m"]:
        s = s.replace(c, "")
    return s


def write_readable_report(entries, ref14_rows):
    lines = []

    def add(s=""):
        lines.append(s)

    main = [e for e in entries if e["category"] == "main_no_gt"]
    oracle = [e for e in entries if e["category"] == "oracle_gt_motion"]

    add("\033[1mFINAL CAMERA RIG CALIBRATION REPORT\033[0m")
    add("=" * 80)
    add()
    add("Interpretation:")
    add("  MAIN result is cam_edge_3-rooted and deployable.")
    add("  Marker14 table is GT/evaluation-only: same cam3-root estimates expressed in the GT frame of marker14.")
    add("  Multichain relay tests all valid marker/frame combinations; GT is not used for aggregation.")
    add()

    add("\033[1m1) MAIN NO-GT RESULTS, cam_edge_3 reference\033[0m")
    add("-" * 80)
    for e in main:
        cm = 100.0 * e["translation_error_m"]
        deg = e["rotation_error_deg"]
        add(color(cm, deg, f"{e['target_camera']} | {e['method']} | error={cm:.3f} cm, {deg:.3f} deg"))

        T_est, T_gt = e["T_est"], e["T_gt"]
        add(f"  estimated target in cam3 [m]:  x={T_est[0,3]:+.4f}  y={T_est[1,3]:+.4f}  z={T_est[2,3]:+.4f}")
        add(f"  GT target in cam3 [m]:         x={T_gt[0,3]:+.4f}  y={T_gt[1,3]:+.4f}  z={T_gt[2,3]:+.4f}")
        add(f"  delta est-GT [cm]:             dx={100*(T_est[0,3]-T_gt[0,3]):+.2f}  dy={100*(T_est[1,3]-T_gt[1,3]):+.2f}  dz={100*(T_est[2,3]-T_gt[2,3]):+.2f}")

        if "multichain" in e["method"]:
            add(f"  multichain candidates: {e['num_candidates']} | inliers: {e['num_inliers']} | outliers: {e['num_outliers']}")
        else:
            add(f"  direct candidates: {e['num_candidates']} | inliers: {e['num_inliers']} | outliers: {e['num_outliers']}")
        add()

    add("\033[1m2) REF14 GT-EVAL CAMERA POSES\033[0m")
    add("-" * 80)
    add("These are the final camera poses expressed from the GT frame of marker14.")
    add("This is only for simulation/evaluation readability; it is not a deployable Ref14 estimate.")
    add()

    for r in ref14_rows:
        T_est, T_gt = r["T_est"], r["T_gt"]
        cm = 100.0 * r["translation_error_m"]
        deg = r["rotation_error_deg"]
        ed = float(np.linalg.norm(T_est[:3,3]))
        gd = float(np.linalg.norm(T_gt[:3,3]))
        add(color(cm, deg, f"{r['camera']} | error={cm:.3f} cm, {deg:.3f} deg | dist_from_ref14_est={ed:.3f} m | dist_GT={gd:.3f} m"))
        add(f"  estimated ref14->camera [m]:   x={T_est[0,3]:+.4f}  y={T_est[1,3]:+.4f}  z={T_est[2,3]:+.4f}")
        add(f"  GT ref14->camera [m]:          x={T_gt[0,3]:+.4f}  y={T_gt[1,3]:+.4f}  z={T_gt[2,3]:+.4f}")
        add(f"  delta est-GT [cm]:             dx={100*(T_est[0,3]-T_gt[0,3]):+.2f}  dy={100*(T_est[1,3]-T_gt[1,3]):+.2f}  dz={100*(T_est[2,3]-T_gt[2,3]):+.2f}")
        add()

    add("\033[1m3) MULTICHAIN RELAY DETAILS + ORACLE SANITY\033[0m")
    add("-" * 80)
    add("Multichain candidate generation:")
    add("  all valid root_marker × root_moving_frame × target_marker × target_moving_frame combinations.")
    add("Outlier rule:")
    add("  no GT; inlier if deviation to robust consensus <= median + 3*MAD for translation and rotation.")
    add("Final value:")
    add("  weighted mean of inlier transforms, weighted by observable marker quality.")
    add()

    for e in main:
        if "multichain" not in e["method"]:
            continue
        cm = 100.0 * e["translation_error_m"]
        deg = e["rotation_error_deg"]
        add(color(cm, deg, f"{e['target_camera']} | COLMAP multichain | error={cm:.3f} cm, {deg:.3f} deg"))
        add(f"  candidates/inliers/outliers: {e['num_candidates']} / {e['num_inliers']} / {e['num_outliers']}")
        add(f"  best individual candidate by GT, eval-only: m{e.get('best_oracle_root_marker','')}@f{e.get('best_oracle_root_frame','')} -> m{e.get('best_oracle_target_marker','')}@f{e.get('best_oracle_target_frame','')} | {e.get('best_oracle_translation_error_cm','')} cm, {e.get('best_oracle_rotation_error_deg','')} deg")
        add()

    for e in oracle:
        cm = 100.0 * e["translation_error_m"]
        deg = e["rotation_error_deg"]
        add(color(cm, deg, f"{e['target_camera']} | GT-motion multichain oracle | error={cm:.3f} cm, {deg:.3f} deg"))
        add(f"  candidates/inliers/outliers: {e['num_candidates']} / {e['num_inliers']} / {e['num_outliers']}")
        add()

    add("\033[1m4) TAKEAWAY\033[0m")
    add("-" * 80)
    add("Main deployable output:")
    for e in main:
        add(f"  {e['target_camera']}: {100.0*e['translation_error_m']:.3f} cm, {e['rotation_error_deg']:.3f} deg via {e['method']}")
    add()
    add("Ref14 output:")
    add("  Same estimates, transformed into the GT marker14 frame for simulation readability/evaluation.")
    add("  The root cam_edge_3 has 0 error there because T_ref14_cam3 is taken from GT for this eval-only coordinate transform.")

    color_text = "\n".join(lines)
    print(color_text)

    p = OUT / "FINAL_READABLE_REPORT.txt"
    p.write_text(plain(color_text) + "\n")
    print()
    print("[OK] wrote:", p)


def write_readme(entries):
    p = OUT / "README.txt"
    main = [e for e in entries if e["category"] == "main_no_gt"]

    lines = [
        "Final extrinsics summary, cam_edge_3 reference",
        "================================================",
        "",
        "Main calibration root:",
        "  cam_edge_3",
        "",
        "Pipeline:",
        "  1. cam_edge_3 -> cam_edge_0 via moving-camera relay multichain + COLMAP motion + ArUco metric scale.",
        "  2. cam_edge_3 -> cam_edge_1 via moving-camera relay multichain + COLMAP motion + ArUco metric scale.",
        "  3. cam_edge_3 -> cam_edge_5 via moving-camera relay multichain + COLMAP motion + ArUco metric scale.",
        "",
        "Main no-GT results:",
    ]

    for e in main:
        lines.append(f"  {e['target_camera']}: {100.0*e['translation_error_m']:.3f} cm, {e['rotation_error_deg']:.3f} deg via {e['method']}")

    lines += [
        "",
        "Multichain rule:",
        "  All valid marker/frame combinations are evaluated.",
        "  Outliers are removed without GT using median+3*MAD consistency filtering.",
        "  Final relay estimate is the weighted mean of inlier transforms.",
        "",
        "Marker14:",
        "  Marker14 export is GT/evaluation-only and does not replace the cam_edge_3-rooted pipeline.",
    ]

    p.write_text("\n".join(lines) + "\n")
    print("[OK] wrote:", p)


def main():
    m = load_chain_module()

    entries = [
        load_relay_multichain("cam3_to_cam0", "COLMAP_motion", "main_no_gt"),
        load_relay_multichain("cam3_to_cam1", "COLMAP_motion", "main_no_gt"),
        load_relay_multichain("cam3_to_cam5", "COLMAP_motion", "main_no_gt"),
    ]

    # Oracle rows if available.
    try:
        entries.append(load_relay_multichain("cam3_to_cam0", "GT_motion", "oracle_gt_motion"))
        entries.append(load_relay_multichain("cam3_to_cam1", "GT_motion", "oracle_gt_motion"))
        entries.append(load_relay_multichain("cam3_to_cam5", "GT_motion", "oracle_gt_motion"))
    except Exception as e:
        print("[WARN] GT_motion multichain oracle not available:", e)

    ref14_rows, ref14_meta = build_ref14_rows(m, entries)

    write_final_summary(entries)
    write_ref14_csv(ref14_rows)
    write_json(entries, ref14_rows, ref14_meta)
    write_readme(entries)
    write_readable_report(entries, ref14_rows)


if __name__ == "__main__":
    main()
