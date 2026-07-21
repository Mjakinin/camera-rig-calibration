#!/usr/bin/env python3
import argparse
from pathlib import Path
import sys

import cv2
import numpy as np

BUS_RUN = Path(__file__).resolve().parents[2]
if str(BUS_RUN) not in sys.path:
    sys.path.insert(0, str(BUS_RUN))

from _shared.common.aruco_utils import (
    detect_markers_in_image,
    draw_marker_debug,
    marker_object_points,
)
from _shared.common.camera_io import load_camera_info_json
from _shared.common.constants import STATIC_CAMERAS, SHARED_RAW_ROOT, MARKER_LENGTH_M
from _shared.common.io_utils import ensure_dir, write_csv


MOVING_CAMERA = "moving_calib_camera"


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


def polygon_area(pts):
    pts = np.asarray(pts, dtype=np.float64)
    x = pts[:, 0]
    y = pts[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def camera_KD(info_path):
    info = load_camera_info_json(info_path)
    return info["K"], info["D"]


def detect_image(image_path, cam_name, observer_type, frame_id, K, dist, marker_length_m, dictionary, debug_path):
    image, detections = detect_markers_in_image(image_path, dictionary)
    rows = []

    objp = marker_object_points(marker_length_m).astype(np.float32)

    for det in detections:
        marker_id = int(det["marker_id"])
        pts = np.asarray(det["corners_px"], dtype=np.float64).reshape(4, 2)
        pts32 = pts.astype(np.float32)

        ok, rvec, tvec = cv2.solvePnP(
            objp,
            pts32,
            K,
            dist,
            flags=cv2.SOLVEPNP_IPPE_SQUARE,
        )
        if not ok:
            ok, rvec, tvec = cv2.solvePnP(
                objp,
                pts32,
                K,
                dist,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )

        if ok:
            rvec = rvec.reshape(3)
            tvec = tvec.reshape(3)
            distance = float(np.linalg.norm(tvec))
        else:
            rvec = np.full(3, np.nan, dtype=np.float64)
            tvec = np.full(3, np.nan, dtype=np.float64)
            distance = float("nan")

        center = pts.mean(axis=0)
        area = float(det.get("area_px2", polygon_area(pts)))

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
            "rvec_x": float(rvec[0]) if ok else "",
            "rvec_y": float(rvec[1]) if ok else "",
            "rvec_z": float(rvec[2]) if ok else "",
            "tvec_x_m": float(tvec[0]) if ok else "",
            "tvec_y_m": float(tvec[1]) if ok else "",
            "tvec_z_m": float(tvec[2]) if ok else "",
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
    debug = draw_marker_debug(image, detections)
    cv2.imwrite(str(debug_path), debug)

    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=str(SHARED_RAW_ROOT))
    ap.add_argument("--out", default="results/bus_real_data/00_shared_baseline/bus_real_data_ref_marker_v1/aruco_observations")
    ap.add_argument("--marker-length-m", type=float, default=MARKER_LENGTH_M)
    ap.add_argument("--dictionary", default="DICT_4X4_50")
    args = ap.parse_args()

    dataset = Path(args.dataset)
    if not dataset.exists():
        raise RuntimeError(f"Shared raw dataset not found: {dataset}")

    out = ensure_dir(Path(args.out))
    debug_dir = ensure_dir(out / "debug_images")

    all_rows, static_rows, moving_rows = [], [], []

    for cam in STATIC_CAMERAS:
        image_path = dataset / "static" / f"{cam}.png"
        info_path = dataset / "camera_info" / f"{cam}.json"

        if not image_path.exists():
            print(f"[WARN] missing static image: {image_path}")
            continue
        if not info_path.exists():
            print(f"[WARN] missing static camera_info: {info_path}")
            continue

        K, dist = camera_KD(info_path)

        rows = detect_image(
            image_path=image_path,
            cam_name=cam,
            observer_type="static",
            frame_id="static",
            K=K,
            dist=dist,
            marker_length_m=args.marker_length_m,
            dictionary=args.dictionary,
            debug_path=debug_dir / "static" / f"{cam}_detections.png",
        )
        static_rows.extend(rows)
        all_rows.extend(rows)

    moving_dir = dataset / "moving"
    moving_info_path = dataset / "camera_info" / f"{MOVING_CAMERA}.json"

    if moving_dir.exists() and moving_info_path.exists():
        K, dist = camera_KD(moving_info_path)

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
                dictionary=args.dictionary,
                debug_path=debug_dir / "moving" / f"{image_path.stem}_detections.png",
            )
            moving_rows.extend(rows)
            all_rows.extend(rows)
    else:
        print(f"[WARN] moving data incomplete: {moving_dir}, {moving_info_path}")

    write_csv(out / "shared_static_aruco_observations.csv", static_rows, fields())
    write_csv(out / "shared_moving_aruco_observations.csv", moving_rows, fields())
    write_csv(out / "shared_all_aruco_observations.csv", all_rows, fields())

    pnp_ok = sum(1 for r in all_rows if str(r["pnp_success"]) == "True")

    summary = [
        "Shared ArUco detection summary",
        "==============================",
        "",
        f"Dataset: {dataset}",
        f"Dictionary: {args.dictionary}",
        f"Marker length [m]: {args.marker_length_m}",
        "",
        f"Static observations: {len(static_rows)}",
        f"Moving observations: {len(moving_rows)}",
        f"Total observations: {len(all_rows)}",
        f"PnP success: {pnp_ok} / {len(all_rows)}",
        "",
        "Output files:",
        f"- {out / 'shared_static_aruco_observations.csv'}",
        f"- {out / 'shared_moving_aruco_observations.csv'}",
        f"- {out / 'shared_all_aruco_observations.csv'}",
        f"- {out / 'debug_images'}",
    ]

    summary_text = "\n".join(summary) + "\n"
    (out / "SHARED_ARUCO_DETECTION_SUMMARY.txt").write_text(summary_text)

    print(summary_text)


if __name__ == "__main__":
    main()
