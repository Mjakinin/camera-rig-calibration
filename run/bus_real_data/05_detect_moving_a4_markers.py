#!/usr/bin/env python3

import argparse
import csv
import json
import math
import shutil
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


ARUCO_DICT_NAME = "DICT_4X4_50"
MARKER_LENGTH_M = 0.170

WIDTH = 1280
HEIGHT = 720
HFOV_DEG = 69.1


def get_aruco_detector():
    aruco = cv2.aruco
    dictionary = aruco.getPredefinedDictionary(getattr(aruco, ARUCO_DICT_NAME))

    if hasattr(aruco, "DetectorParameters"):
        params = aruco.DetectorParameters()
    else:
        params = aruco.DetectorParameters_create()

    if hasattr(aruco, "ArucoDetector"):
        return ("new", aruco.ArucoDetector(dictionary, params), dictionary, params)

    return ("old", None, dictionary, params)


def detect_markers(gray, detector_pack):
    mode, detector, dictionary, params = detector_pack

    if mode == "new":
        corners, ids, rejected = detector.detectMarkers(gray)
    else:
        corners, ids, rejected = cv2.aruco.detectMarkers(gray, dictionary, parameters=params)

    if ids is None:
        ids = np.empty((0, 1), dtype=np.int32)
        corners = []

    return corners, ids


def make_camera_matrix(width, height, hfov_deg):
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


def marker_object_points(marker_length):
    s = marker_length / 2.0
    return np.array([
        [-s,  s, 0.0],
        [ s,  s, 0.0],
        [ s, -s, 0.0],
        [-s, -s, 0.0],
    ], dtype=np.float32)


def load_route_commanded(sequence_dir):
    p = sequence_dir / "route_commanded.csv"
    out = {}

    if not p.exists():
        return out

    with p.open() as f:
        for r in csv.DictReader(f):
            frame = int(r["frame"])
            out[frame] = r

    return out


def load_expected_ids():
    p = Path("src/calib_lab/bus_real_data/config/a4_marker_placements.json")
    if not p.exists():
        return list(range(14))

    data = json.loads(p.read_text())
    ids = []
    for item in data:
        try:
            ids.append(int(item["name"].split("_")[1]))
        except Exception:
            pass

    return sorted(set(ids))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sequence", default="results/bus_real_data/03_moving_camera_sequence")
    ap.add_argument("--clean-debug", action="store_true")
    args = ap.parse_args()

    seq_dir = Path(args.sequence)
    img_dir = seq_dir / "images"
    dbg_dir = seq_dir / "debug_images"

    if args.clean_debug and dbg_dir.exists():
        shutil.rmtree(dbg_dir)

    dbg_dir.mkdir(parents=True, exist_ok=True)

    images = sorted(img_dir.glob("frame_*.png"))
    if not images:
        raise RuntimeError(f"No images found in {img_dir}")

    route_by_frame = load_route_commanded(seq_dir)
    expected_ids = load_expected_ids()

    detector_pack = get_aruco_detector()
    K, D, fx, fy, cx, cy = make_camera_matrix(WIDTH, HEIGHT, HFOV_DEG)
    obj_pts = marker_object_points(MARKER_LENGTH_M)

    detection_rows = []
    summary_rows = []
    marker_to_frames = defaultdict(list)

    max_empty_run = 0
    current_empty_run = 0

    for img_path in images:
        frame = int(img_path.stem.split("_")[1])
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            print("[WARN] could not read", img_path)
            continue

        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        corners, ids = detect_markers(gray, detector_pack)

        debug = bgr.copy()
        detected_ids = []

        if len(corners) > 0:
            cv2.aruco.drawDetectedMarkers(debug, corners, ids)

        for idx, marker_id_arr in enumerate(ids):
            marker_id = int(marker_id_arr[0])
            c = corners[idx].reshape(4, 2).astype(np.float32)
            center = c.mean(axis=0)

            ok, rvec, tvec = cv2.solvePnP(
                obj_pts,
                c,
                K,
                D,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )

            if ok:
                try:
                    cv2.drawFrameAxes(debug, K, D, rvec, tvec, MARKER_LENGTH_M * 0.5)
                except Exception:
                    pass

                tx, ty, tz = [float(v) for v in tvec.reshape(3)]
                rx, ry, rz = [float(v) for v in rvec.reshape(3)]
                distance = math.sqrt(tx * tx + ty * ty + tz * tz)
            else:
                tx = ty = tz = rx = ry = rz = distance = float("nan")

            detected_ids.append(marker_id)
            marker_to_frames[marker_id].append(frame)

            route = route_by_frame.get(frame, {})

            detection_rows.append({
                "frame": frame,
                "image": str(img_path),
                "marker_id": marker_id,
                "center_u": float(center[0]),
                "center_v": float(center[1]),
                "corner0_u": float(c[0, 0]),
                "corner0_v": float(c[0, 1]),
                "corner1_u": float(c[1, 0]),
                "corner1_v": float(c[1, 1]),
                "corner2_u": float(c[2, 0]),
                "corner2_v": float(c[2, 1]),
                "corner3_u": float(c[3, 0]),
                "corner3_v": float(c[3, 1]),
                "pnp_success": bool(ok),
                "tvec_x_m": tx,
                "tvec_y_m": ty,
                "tvec_z_m": tz,
                "distance_m": distance,
                "rvec_x": rx,
                "rvec_y": ry,
                "rvec_z": rz,
                "route_x": route.get("x", ""),
                "route_y": route.get("y", ""),
                "route_z": route.get("z", ""),
                "route_roll": route.get("roll", ""),
                "route_pitch": route.get("pitch", ""),
                "route_yaw": route.get("yaw", ""),
            })

        detected_ids_sorted = sorted(set(detected_ids))

        if len(detected_ids_sorted) == 0:
            current_empty_run += 1
            max_empty_run = max(max_empty_run, current_empty_run)
        else:
            current_empty_run = 0

        dbg_path = dbg_dir / f"frame_{frame:04d}_debug.png"
        cv2.imwrite(str(dbg_path), debug)

        route = route_by_frame.get(frame, {})

        summary_rows.append({
            "frame": frame,
            "num_detected": len(detected_ids_sorted),
            "detected_ids": ";".join(str(v) for v in detected_ids_sorted),
            "image": str(img_path),
            "debug_image": str(dbg_path),
            "route_x": route.get("x", ""),
            "route_y": route.get("y", ""),
            "route_z": route.get("z", ""),
            "route_roll": route.get("roll", ""),
            "route_pitch": route.get("pitch", ""),
            "route_yaw": route.get("yaw", ""),
        })

        print(f"[frame {frame:04d}] IDs: {detected_ids_sorted}")

    detections_csv = seq_dir / "moving_detections.csv"
    summary_csv = seq_dir / "moving_summary_by_frame.csv"
    coverage_csv = seq_dir / "moving_coverage_by_marker.csv"

    with detections_csv.open("w", newline="") as f:
        fields = [
            "frame", "image", "marker_id",
            "center_u", "center_v",
            "corner0_u", "corner0_v", "corner1_u", "corner1_v",
            "corner2_u", "corner2_v", "corner3_u", "corner3_v",
            "pnp_success", "tvec_x_m", "tvec_y_m", "tvec_z_m",
            "distance_m", "rvec_x", "rvec_y", "rvec_z",
            "route_x", "route_y", "route_z", "route_roll", "route_pitch", "route_yaw",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(detection_rows)

    with summary_csv.open("w", newline="") as f:
        fields = [
            "frame", "num_detected", "detected_ids", "image", "debug_image",
            "route_x", "route_y", "route_z", "route_roll", "route_pitch", "route_yaw",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary_rows)

    coverage_rows = []
    for marker_id in sorted(marker_to_frames):
        frames = sorted(marker_to_frames[marker_id])
        coverage_rows.append({
            "marker_id": marker_id,
            "num_frames": len(frames),
            "first_frame": frames[0],
            "last_frame": frames[-1],
            "frames": ";".join(str(v) for v in frames),
        })

    with coverage_csv.open("w", newline="") as f:
        fields = ["marker_id", "num_frames", "first_frame", "last_frame", "frames"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(coverage_rows)

    all_ids = sorted(marker_to_frames.keys())
    missing = sorted(set(expected_ids) - set(all_ids))

    report = seq_dir / "moving_detection_report.txt"
    report.write_text(
        "Moving camera ArUco detection report\n"
        "====================================\n\n"
        f"Frames: {len(summary_rows)}\n"
        f"Total detections: {len(detection_rows)}\n"
        f"Unique detected IDs: {all_ids}\n"
        f"Expected IDs: {expected_ids}\n"
        f"Missing IDs: {missing}\n"
        f"Max consecutive frames without marker: {max_empty_run}\n\n"
        f"Debug images: {dbg_dir}\n"
        f"Summary CSV: {summary_csv}\n"
        f"Detections CSV: {detections_csv}\n"
        f"Coverage CSV: {coverage_csv}\n"
    )

    print()
    print("=== MOVING ROUTE DETECTION SUMMARY ===")
    print("frames:", len(summary_rows))
    print("total detections:", len(detection_rows))
    print("unique detected IDs:", all_ids)
    print("expected IDs:", expected_ids)
    print("missing IDs:", missing)
    print("max consecutive frames without marker:", max_empty_run)
    print()
    print("[OK] debug images:", dbg_dir)
    print("[OK] summary:", summary_csv)
    print("[OK] detections:", detections_csv)
    print("[OK] coverage:", coverage_csv)
    print("[OK] report:", report)


if __name__ == "__main__":
    main()
