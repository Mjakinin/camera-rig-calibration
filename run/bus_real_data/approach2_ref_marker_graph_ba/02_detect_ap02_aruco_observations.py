#!/usr/bin/env python3

import argparse
from pathlib import Path

import cv2
import numpy as np

from ap02_common import (
    AP02_ROOT,
    SHARED_RAW_ROOT,
    STATIC_CAMERAS,
    MOVING_CAMERA,
    DEFAULT_MARKER_LENGTH_M,
    ensure_dir,
    load_camera_info_json,
    write_csv,
)


def get_aruco_detector(dictionary_name):
    aruco = cv2.aruco
    dictionary_id = getattr(aruco, dictionary_name)
    dictionary = aruco.getPredefinedDictionary(dictionary_id)

    if hasattr(aruco, "ArucoDetector"):
        params = aruco.DetectorParameters()
        detector = aruco.ArucoDetector(dictionary, params)

        def detect(gray):
            return detector.detectMarkers(gray)

        return detect

    params = aruco.DetectorParameters_create()

    def detect(gray):
        return aruco.detectMarkers(gray, dictionary, parameters=params)

    return detect


def marker_object_points(marker_length_m):
    s = marker_length_m / 2.0
    return np.array([
        [-s,  s, 0.0],
        [ s,  s, 0.0],
        [ s, -s, 0.0],
        [-s, -s, 0.0],
    ], dtype=np.float32)


def polygon_area(pts):
    pts = np.asarray(pts, dtype=np.float64)
    x = pts[:, 0]
    y = pts[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def detect_image(image_path, cam_name, observer_type, frame_id, K, dist, marker_length_m, detect_fn, debug_path):
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Could not read image: {image_path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detect_fn(gray)
    rows = []
    debug = image.copy()

    if ids is not None and len(ids) > 0:
        cv2.aruco.drawDetectedMarkers(debug, corners, ids)
        objp = marker_object_points(marker_length_m)

        for marker_corners, marker_id_arr in zip(corners, ids):
            marker_id = int(marker_id_arr[0])
            pts = marker_corners.reshape(-1, 2).astype(np.float32)

            ok, rvec, tvec = cv2.solvePnP(objp, pts, K, dist, flags=cv2.SOLVEPNP_IPPE_SQUARE)
            if not ok:
                ok, rvec, tvec = cv2.solvePnP(objp, pts, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)

            center = pts.mean(axis=0)
            area = polygon_area(pts)
            distance = float(np.linalg.norm(tvec.reshape(3))) if ok else float("nan")

            row = {
                "observer_type": observer_type,
                "observer_id": cam_name if observer_type == "static" else f"moving_frame_{int(frame_id):06d}",
                "camera_name": cam_name,
                "frame_id": frame_id,
                "image_path": str(image_path),
                "marker_id": marker_id,
                "marker_length_m": marker_length_m,
                "fx": float(K[0, 0]),
                "fy": float(K[1, 1]),
                "cx": float(K[0, 2]),
                "cy": float(K[1, 2]),
                "pnp_success": bool(ok),
                "rvec_x": float(rvec[0, 0]) if ok else "",
                "rvec_y": float(rvec[1, 0]) if ok else "",
                "rvec_z": float(rvec[2, 0]) if ok else "",
                "tvec_x_m": float(tvec[0, 0]) if ok else "",
                "tvec_y_m": float(tvec[1, 0]) if ok else "",
                "tvec_z_m": float(tvec[2, 0]) if ok else "",
                "distance_m": distance,
                "center_u": float(center[0]),
                "center_v": float(center[1]),
                "area_px2": area,
            }

            for i in range(4):
                row[f"corner{i}_u"] = float(pts[i, 0])
                row[f"corner{i}_v"] = float(pts[i, 1])

            rows.append(row)

    ensure_dir(debug_path.parent)
    cv2.imwrite(str(debug_path), debug)
    return rows


def fields():
    base = [
        "observer_type", "observer_id", "camera_name", "frame_id", "image_path",
        "marker_id", "marker_length_m",
        "fx", "fy", "cx", "cy",
        "pnp_success",
        "rvec_x", "rvec_y", "rvec_z",
        "tvec_x_m", "tvec_y_m", "tvec_z_m",
        "distance_m", "center_u", "center_v", "area_px2",
    ]
    for i in range(4):
        base += [f"corner{i}_u", f"corner{i}_v"]
    return base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=str(SHARED_RAW_ROOT))
    ap.add_argument("--out", default=str(AP02_ROOT / "02_aruco_observations"))
    ap.add_argument("--marker-length-m", type=float, default=DEFAULT_MARKER_LENGTH_M)
    ap.add_argument("--dictionary", default="DICT_4X4_50")
    args = ap.parse_args()

    dataset = Path(args.dataset)
    if not dataset.exists():
        raise RuntimeError(f"Shared raw dataset not found: {dataset}")

    out = ensure_dir(Path(args.out))
    debug_dir = ensure_dir(out / "debug_images")
    detect_fn = get_aruco_detector(args.dictionary)

    all_rows, static_rows, moving_rows = [], [], []

    for cam in STATIC_CAMERAS:
        image_path = dataset / "static" / f"{cam}.png"
        info_path = dataset / "camera_info" / f"{cam}.json"
        if not image_path.exists():
            print(f"[WARN] missing static image: {image_path}")
            continue

        info = load_camera_info_json(info_path)
        K = np.array(info["k"], dtype=np.float64).reshape(3, 3)
        dist = np.array(info.get("d", []), dtype=np.float64)

        rows = detect_image(
            image_path=image_path,
            cam_name=cam,
            observer_type="static",
            frame_id="static",
            K=K,
            dist=dist,
            marker_length_m=args.marker_length_m,
            detect_fn=detect_fn,
            debug_path=debug_dir / "static" / f"{cam}_detections.png",
        )
        static_rows.extend(rows)
        all_rows.extend(rows)

    moving_dir = dataset / "moving"
    moving_info_path = dataset / "camera_info" / f"{MOVING_CAMERA}.json"
    if moving_dir.exists() and moving_info_path.exists():
        info = load_camera_info_json(moving_info_path)
        K = np.array(info["k"], dtype=np.float64).reshape(3, 3)
        dist = np.array(info.get("d", []), dtype=np.float64)

        for image_path in sorted(moving_dir.glob("frame_*.png")):
            try:
                frame_id = int(image_path.stem.split("_")[-1])
            except Exception:
                frame_id = 0

            rows = detect_image(
                image_path=image_path,
                cam_name=MOVING_CAMERA,
                observer_type="moving",
                frame_id=frame_id,
                K=K,
                dist=dist,
                marker_length_m=args.marker_length_m,
                detect_fn=detect_fn,
                debug_path=debug_dir / "moving" / f"{image_path.stem}_detections.png",
            )
            moving_rows.extend(rows)
            all_rows.extend(rows)

    write_csv(out / "ap02_static_aruco_observations.csv", static_rows, fields())
    write_csv(out / "ap02_moving_aruco_observations.csv", moving_rows, fields())
    write_csv(out / "ap02_all_aruco_observations.csv", all_rows, fields())

    summary = [
        "AP02 ArUco detection summary",
        "============================",
        "",
        f"Dataset: {dataset}",
        f"Dictionary: {args.dictionary}",
        f"Marker length [m]: {args.marker_length_m}",
        "",
        f"Static observations: {len(static_rows)}",
        f"Moving observations: {len(moving_rows)}",
        f"All observations: {len(all_rows)}",
        "",
        "Output files:",
        "- ap02_static_aruco_observations.csv",
        "- ap02_moving_aruco_observations.csv",
        "- ap02_all_aruco_observations.csv",
        "- debug_images/",
        "",
    ]
    (out / "ap02_detection_summary.txt").write_text("\n".join(summary))

    print("[OK] wrote", out)
    print("[OK] dataset:", dataset)
    print("[OK] static observations:", len(static_rows))
    print("[OK] moving observations:", len(moving_rows))
    print("[OK] all observations:", len(all_rows))


if __name__ == "__main__":
    main()
