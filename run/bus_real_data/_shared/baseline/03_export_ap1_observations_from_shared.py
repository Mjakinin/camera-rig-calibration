#!/usr/bin/env python3
import argparse
import csv
import json
import math
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import yaml

BUS_RUN = Path(__file__).resolve().parents[1]
if str(BUS_RUN) not in sys.path:
    sys.path.insert(0, str(BUS_RUN))

from _shared.common.aruco_utils import marker_object_points
from _shared.common.constants import MARKER_LENGTH_M, STATIC_CAMERAS


WIDTH = 1280
HEIGHT = 720
HFOV_DEG = 69.1


def read_csv(path):
    path = Path(path)
    if not path.exists():
        return []
    with path.open() as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fields):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def load_intrinsics(path):
    data = yaml.safe_load(Path(path).read_text())
    out = {}

    for cam, intr in data.items():
        K = np.array([
            [float(intr["fx"]), 0.0, float(intr["cx"])],
            [0.0, float(intr["fy"]), float(intr["cy"])],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)

        D = np.array(intr.get("distortion", [0, 0, 0, 0, 0]), dtype=np.float64).reshape(-1, 1)

        out[cam] = {
            **intr,
            "K": K,
            "D": D,
        }

    return out


def make_moving_camera_matrix(width=WIDTH, height=HEIGHT, hfov_deg=HFOV_DEG):
    hfov = math.radians(hfov_deg)
    fx = width / (2.0 * math.tan(hfov / 2.0))
    fy = fx
    cx = width / 2.0
    cy = height / 2.0

    K = np.array([
        [fx, 0.0, cx],
        [0.0, fy, cy],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)

    D = np.zeros((5, 1), dtype=np.float64)
    return K, D, fx, fy, cx, cy


def make_moving_camera_matrix_from_json(path):
    path = Path(path)
    data = json.loads(path.read_text())

    if "k" in data:
        flat = data["k"]
    elif "K" in data:
        flat = data["K"]
    elif "camera_matrix" in data:
        cm = data["camera_matrix"]
        flat = cm["data"] if isinstance(cm, dict) else cm
    else:
        fx = float(data["fx"])
        fy = float(data.get("fy", fx))
        cx = float(data["cx"])
        cy = float(data["cy"])
        flat = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]

    K = np.asarray(flat, dtype=np.float64).reshape(3, 3)

    if "D" in data:
        D = np.asarray(data["D"], dtype=np.float64).reshape(-1, 1)
    elif "d" in data:
        D = np.asarray(data["d"], dtype=np.float64).reshape(-1, 1)
    elif "distortion" in data:
        D = np.asarray(data["distortion"], dtype=np.float64).reshape(-1, 1)
    else:
        D = np.zeros((5, 1), dtype=np.float64)

    fx = float(K[0, 0])
    fy = float(K[1, 1])
    cx = float(K[0, 2])
    cy = float(K[1, 2])
    return K, D, fx, fy, cx, cy


def solve_ap1_iterative_pnp(row, K, D):
    pts = np.array([
        [float(row["corner0_u"]), float(row["corner0_v"])],
        [float(row["corner1_u"]), float(row["corner1_v"])],
        [float(row["corner2_u"]), float(row["corner2_v"])],
        [float(row["corner3_u"]), float(row["corner3_v"])],
    ], dtype=np.float32)

    obj_pts = marker_object_points(MARKER_LENGTH_M).astype(np.float32)

    ok, rvec, tvec = cv2.solvePnP(
        obj_pts,
        pts,
        K,
        D,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )

    if ok:
        tx, ty, tz = [float(v) for v in tvec.reshape(3)]
        rx, ry, rz = [float(v) for v in rvec.reshape(3)]
        distance = math.sqrt(tx * tx + ty * ty + tz * tz)
    else:
        tx = ty = tz = rx = ry = rz = distance = float("nan")

    return bool(ok), (rx, ry, rz), (tx, ty, tz), distance


def load_route_commanded(route_csv):
    route_csv = Path(route_csv)
    out = {}
    if not route_csv.exists():
        return out

    with route_csv.open() as f:
        for r in csv.DictReader(f):
            try:
                frame = int(r["frame"])
            except Exception:
                continue
            out[frame] = r
    return out


def load_expected_ids():
    p = Path("src/calib_lab/bus_real_data/config/a4_marker_placements.json")
    if not p.exists():
        return list(range(15))

    data = json.loads(p.read_text())
    ids = []
    for item in data:
        try:
            ids.append(int(item["name"].split("_")[1]))
        except Exception:
            pass

    return sorted(set(ids))


def copy_if_exists(src, dst):
    src = Path(src)
    dst = Path(dst)
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def export_static(shared_rows, shared_raw, shared_debug, static_out, intrinsics, link_images):
    static_out = Path(static_out)
    raw_dir = static_out / "raw_images"
    dbg_dir = static_out / "debug_images"
    raw_dir.mkdir(parents=True, exist_ok=True)
    dbg_dir.mkdir(parents=True, exist_ok=True)

    rows_by_cam = defaultdict(list)
    for r in shared_rows:
        rows_by_cam[r["camera_name"]].append(r)

    detection_rows = []
    summary_rows = []

    for cam in sorted(STATIC_CAMERAS):
        intr = intrinsics[cam]
        K = intr["K"]
        D = intr["D"]

        raw_path = raw_dir / f"{cam}.png"
        dbg_path = dbg_dir / f"{cam}_debug.png"

        if link_images:
            copy_if_exists(Path(shared_raw) / "static" / f"{cam}.png", raw_path)
            copy_if_exists(Path(shared_debug) / "static" / f"{cam}_detections.png", dbg_path)

        detected_ids = []

        for r in rows_by_cam.get(cam, []):
            ok, rvec, tvec, distance = solve_ap1_iterative_pnp(r, K, D)
            detected_ids.append(int(float(r["marker_id"])))

            detection_rows.append({
                "camera": cam,
                "source_camera_info": intr.get("source_camera_info", ""),
                "image_width": int(intr["width"]),
                "image_height": int(intr["height"]),
                "fx": float(intr["fx"]),
                "fy": float(intr["fy"]),
                "cx": float(intr["cx"]),
                "cy": float(intr["cy"]),
                "hfov_deg": float(intr["horizontal_fov_deg"]),
                "vfov_deg": float(intr["vertical_fov_deg"]),
                "marker_length_m": MARKER_LENGTH_M,
                "marker_id": int(float(r["marker_id"])),
                "center_u": float(r["center_u"]),
                "center_v": float(r["center_v"]),
                "corner0_u": float(r["corner0_u"]),
                "corner0_v": float(r["corner0_v"]),
                "corner1_u": float(r["corner1_u"]),
                "corner1_v": float(r["corner1_v"]),
                "corner2_u": float(r["corner2_u"]),
                "corner2_v": float(r["corner2_v"]),
                "corner3_u": float(r["corner3_u"]),
                "corner3_v": float(r["corner3_v"]),
                "pnp_success": ok,
                "tvec_x_m": tvec[0],
                "tvec_y_m": tvec[1],
                "tvec_z_m": tvec[2],
                "distance_m": distance,
                "rvec_x": rvec[0],
                "rvec_y": rvec[1],
                "rvec_z": rvec[2],
            })

        detected_ids_sorted = sorted(set(detected_ids))
        summary_rows.append({
            "camera": cam,
            "source_camera_info": intr.get("source_camera_info", ""),
            "num_detected": len(detected_ids_sorted),
            "detected_ids": ";".join(str(v) for v in detected_ids_sorted),
            "raw_image": str(raw_path) if detected_ids_sorted or raw_path.exists() else "",
            "debug_image": str(dbg_path) if detected_ids_sorted or dbg_path.exists() else "",
            "status": "ok" if detected_ids_sorted else "no_markers",
        })

    detection_fields = [
        "camera", "source_camera_info", "image_width", "image_height",
        "fx", "fy", "cx", "cy", "hfov_deg", "vfov_deg",
        "marker_length_m", "marker_id",
        "center_u", "center_v",
        "corner0_u", "corner0_v", "corner1_u", "corner1_v",
        "corner2_u", "corner2_v", "corner3_u", "corner3_v",
        "pnp_success",
        "tvec_x_m", "tvec_y_m", "tvec_z_m", "distance_m",
        "rvec_x", "rvec_y", "rvec_z",
    ]

    summary_fields = [
        "camera", "source_camera_info", "num_detected", "detected_ids",
        "raw_image", "debug_image", "status",
    ]

    write_csv(static_out / "detections.csv", detection_rows, detection_fields)
    write_csv(static_out / "summary_by_camera.csv", summary_rows, summary_fields)

    (static_out / "README.txt").write_text(
        "AP01 static detections exported from shared ArUco baseline.\n"
        "Marker corners come from shared baseline; PnP is recomputed with AP01-compatible SOLVEPNP_ITERATIVE.\n"
    )

    print(f"[OK] AP01 static detections from shared: {static_out / 'detections.csv'}")
    print(f"[OK] AP01 static summary from shared: {static_out / 'summary_by_camera.csv'}")


def export_moving(shared_rows, shared_raw, shared_debug, sequence_dir, route_csv, link_images):
    sequence_dir = Path(sequence_dir)
    img_dir = sequence_dir / "images"
    dbg_dir = sequence_dir / "debug_images"
    img_dir.mkdir(parents=True, exist_ok=True)
    dbg_dir.mkdir(parents=True, exist_ok=True)

    route_by_frame = load_route_commanded(route_csv)
    expected_ids = load_expected_ids()

    moving_info_json = Path(shared_raw) / "camera_info" / "moving_calib_camera.json"
    if moving_info_json.exists():
        K, D, fx, fy, cx, cy = make_moving_camera_matrix_from_json(moving_info_json)
        print(f"[OK] AP01 moving intrinsics loaded from {moving_info_json}")
    else:
        K, D, fx, fy, cx, cy = make_moving_camera_matrix()
        print("[WARN] AP01 moving intrinsics fallback: hardcoded baseline intrinsics")

    rows_by_frame = defaultdict(list)
    marker_to_frames = defaultdict(list)

    for r in shared_rows:
        try:
            frame = int(float(r["frame_id"]))
        except Exception:
            frame = int(str(r["observer_id"]).split("_")[-1])
        rows_by_frame[frame].append(r)

    shared_moving_dir = Path(shared_raw) / "moving"
    all_frames = []
    if shared_moving_dir.exists():
        for p in sorted(shared_moving_dir.glob("frame_*.png")):
            try:
                all_frames.append(int(p.stem.split("_")[-1]))
            except Exception:
                pass

    if not all_frames:
        all_frames = sorted(rows_by_frame)

    detection_rows = []
    summary_rows = []

    for frame in sorted(all_frames):
        image_name = f"frame_{frame:04d}.png"
        image_path = img_dir / image_name
        dbg_path = dbg_dir / f"frame_{frame:04d}_debug.png"

        if link_images:
            copy_if_exists(shared_moving_dir / image_name, image_path)
            copy_if_exists(Path(shared_debug) / "moving" / f"frame_{frame:04d}_detections.png", dbg_path)

        detected_ids = []
        route = route_by_frame.get(frame, {})

        for r in rows_by_frame.get(frame, []):
            ok, rvec, tvec, distance = solve_ap1_iterative_pnp(r, K, D)
            marker_id = int(float(r["marker_id"]))
            detected_ids.append(marker_id)
            marker_to_frames[marker_id].append(frame)

            detection_rows.append({
                "frame": frame,
                "image": str(image_path),
                "marker_id": marker_id,
                "center_u": float(r["center_u"]),
                "center_v": float(r["center_v"]),
                "corner0_u": float(r["corner0_u"]),
                "corner0_v": float(r["corner0_v"]),
                "corner1_u": float(r["corner1_u"]),
                "corner1_v": float(r["corner1_v"]),
                "corner2_u": float(r["corner2_u"]),
                "corner2_v": float(r["corner2_v"]),
                "corner3_u": float(r["corner3_u"]),
                "corner3_v": float(r["corner3_v"]),
                "pnp_success": ok,
                "tvec_x_m": tvec[0],
                "tvec_y_m": tvec[1],
                "tvec_z_m": tvec[2],
                "distance_m": distance,
                "rvec_x": rvec[0],
                "rvec_y": rvec[1],
                "rvec_z": rvec[2],
                "route_x": route.get("x", ""),
                "route_y": route.get("y", ""),
                "route_z": route.get("z", ""),
                "route_roll": route.get("roll", ""),
                "route_pitch": route.get("pitch", ""),
                "route_yaw": route.get("yaw", ""),
            })

        detected_ids_sorted = sorted(set(detected_ids))
        summary_rows.append({
            "frame": frame,
            "num_detected": len(detected_ids_sorted),
            "detected_ids": ";".join(str(v) for v in detected_ids_sorted),
            "image": str(image_path),
            "debug_image": str(dbg_path),
            "route_x": route.get("x", ""),
            "route_y": route.get("y", ""),
            "route_z": route.get("z", ""),
            "route_roll": route.get("roll", ""),
            "route_pitch": route.get("pitch", ""),
            "route_yaw": route.get("yaw", ""),
        })

    coverage_rows = []
    for marker_id in sorted(set(expected_ids) | set(marker_to_frames)):
        frames = sorted(set(marker_to_frames.get(marker_id, [])))
        coverage_rows.append({
            "marker_id": marker_id,
            "num_frames": len(frames),
            "first_frame": frames[0] if frames else "",
            "last_frame": frames[-1] if frames else "",
            "frames": ";".join(str(v) for v in frames),
        })

    det_fields = [
        "frame", "image", "marker_id",
        "center_u", "center_v",
        "corner0_u", "corner0_v", "corner1_u", "corner1_v",
        "corner2_u", "corner2_v", "corner3_u", "corner3_v",
        "pnp_success", "tvec_x_m", "tvec_y_m", "tvec_z_m",
        "distance_m", "rvec_x", "rvec_y", "rvec_z",
        "route_x", "route_y", "route_z", "route_roll", "route_pitch", "route_yaw",
    ]

    summary_fields = [
        "frame", "num_detected", "detected_ids", "image", "debug_image",
        "route_x", "route_y", "route_z", "route_roll", "route_pitch", "route_yaw",
    ]

    coverage_fields = ["marker_id", "num_frames", "first_frame", "last_frame", "frames"]

    write_csv(sequence_dir / "moving_detections.csv", detection_rows, det_fields)
    write_csv(sequence_dir / "moving_summary_by_frame.csv", summary_rows, summary_fields)
    write_csv(sequence_dir / "moving_coverage_by_marker.csv", coverage_rows, coverage_fields)

    route_csv = Path(route_csv)
    if route_csv.exists() and route_csv.resolve() != (sequence_dir / "route_commanded.csv").resolve():
        copy_if_exists(route_csv, sequence_dir / "route_commanded.csv")

    report = [
        "AP01 moving detections exported from shared ArUco baseline",
        "==========================================================",
        "",
        "Marker corners come from shared baseline.",
        "PnP is recomputed with AP01-compatible SOLVEPNP_ITERATIVE.",
        "",
        f"Frames: {len(summary_rows)}",
        f"Detection rows: {len(detection_rows)}",
        f"Markers with detections: {len([r for r in coverage_rows if int(r['num_frames']) > 0])}",
        "",
        f"Detections CSV: {sequence_dir / 'moving_detections.csv'}",
        f"Summary CSV: {sequence_dir / 'moving_summary_by_frame.csv'}",
        f"Coverage CSV: {sequence_dir / 'moving_coverage_by_marker.csv'}",
    ]
    (sequence_dir / "moving_detection_report.txt").write_text("\n".join(report) + "\n")

    print(f"[OK] AP01 moving detections from shared: {sequence_dir / 'moving_detections.csv'}")
    print(f"[OK] AP01 moving summary from shared: {sequence_dir / 'moving_summary_by_frame.csv'}")
    print(f"[OK] AP01 moving coverage from shared: {sequence_dir / 'moving_coverage_by_marker.csv'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shared-obs", default="results/bus_real_data/00_shared_baseline/bus_real_data_ref_marker_v1/aruco_observations")
    ap.add_argument("--shared-raw", default="results/bus_real_data/00_shared_baseline/bus_real_data_ref_marker_v1/raw_images")
    ap.add_argument("--intrinsics", default="src/calib_lab/bus_real_data/config/camera_intrinsics_by_camera.yaml")
    ap.add_argument("--ap1-root", default="results/bus_real_data/01_marker_direct_relay_multimarker_multichain")
    ap.add_argument("--static-out", default=None)
    ap.add_argument("--sequence", default=None)
    ap.add_argument("--route-csv", default=None)
    ap.add_argument("--link-images", action="store_true")
    args = ap.parse_args()

    shared_obs = Path(args.shared_obs)
    shared_raw = Path(args.shared_raw)

    static_shared = read_csv(shared_obs / "shared_static_aruco_observations.csv")
    moving_shared = read_csv(shared_obs / "shared_moving_aruco_observations.csv")

    if not static_shared:
        raise RuntimeError(f"No shared static observations found in {shared_obs}")
    if not moving_shared:
        raise RuntimeError(f"No shared moving observations found in {shared_obs}")

    ap1_root = Path(args.ap1_root)
    static_out = Path(args.static_out) if args.static_out else ap1_root / ".ap01_compat_cache" / "static_observations"
    sequence_dir = Path(args.sequence) if args.sequence else ap1_root / ".ap01_compat_cache" / "moving_observations"

    route_csv = Path(args.route_csv) if args.route_csv else sequence_dir / "route_commanded.csv"
    if not route_csv.exists():
        candidate = shared_raw / "ap1_metadata" / "route_commanded.csv"
        if candidate.exists():
            route_csv = candidate

    intrinsics = load_intrinsics(args.intrinsics)

    export_static(
        shared_rows=static_shared,
        shared_raw=shared_raw,
        shared_debug=shared_obs / "debug_images",
        static_out=static_out,
        intrinsics=intrinsics,
        link_images=args.link_images,
    )

    export_moving(
        shared_rows=moving_shared,
        shared_raw=shared_raw,
        shared_debug=shared_obs / "debug_images",
        sequence_dir=sequence_dir,
        route_csv=route_csv,
        link_images=args.link_images,
    )

    print("[OK] AP01 observation export from shared baseline complete.")


if __name__ == "__main__":
    main()
