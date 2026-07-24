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
        "fx", "fy", "cx", "cy", "distortion_model",
        "detection_success", "pnp_success", "pnp_reprojection_rmse_px",
        "rvec_x", "rvec_y", "rvec_z",
        "tvec_x_m", "tvec_y_m", "tvec_z_m",
        "distance_m", "center_u", "center_v", "area_px2",
    ]
    base += [f"d{i}" for i in range(8)]
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
    return (
        info["K"],
        info["D"],
        str(info.get("raw", {}).get("distortion_model", "plumb_bob")),
    )


def pnp_reprojection_rmse(objp, corners, rvec, tvec, K, dist, distortion_model):
    if distortion_model in {"equidistant", "fisheye"}:
        projected, _ = cv2.fisheye.projectPoints(
            np.asarray(objp, dtype=np.float64).reshape(-1, 1, 3),
            np.asarray(rvec, dtype=np.float64).reshape(3, 1),
            np.asarray(tvec, dtype=np.float64).reshape(3, 1),
            np.asarray(K, dtype=np.float64),
            np.asarray(dist, dtype=np.float64).reshape(-1, 1)[:4],
        )
    else:
        projected, _ = cv2.projectPoints(
            np.asarray(objp, dtype=np.float64),
            np.asarray(rvec, dtype=np.float64).reshape(3, 1),
            np.asarray(tvec, dtype=np.float64).reshape(3, 1),
            np.asarray(K, dtype=np.float64),
            np.asarray(dist, dtype=np.float64),
        )
    residuals = projected.reshape(4, 2) - np.asarray(corners).reshape(4, 2)
    return float(np.sqrt(np.mean(np.sum(residuals * residuals, axis=1))))


def detect_image(
    image_path,
    cam_name,
    observer_type,
    frame_id,
    K,
    dist,
    distortion_model,
    marker_length_m,
    dictionary,
    debug_path,
    allowed_marker_ids=None,
    minimum_area_px2=0.0,
):
    image, detections = detect_markers_in_image(image_path, dictionary)
    rows = []

    objp = marker_object_points(marker_length_m).astype(np.float32)

    for det in detections:
        marker_id = int(det["marker_id"])
        pts = np.asarray(det["corners_px"], dtype=np.float64).reshape(4, 2)
        pts32 = pts.astype(np.float32)
        area = float(det.get("area_px2", polygon_area(pts)))

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
            reprojection_rmse = pnp_reprojection_rmse(
                objp, pts, rvec, tvec, K, dist, distortion_model
            )
        else:
            rvec = np.full(3, np.nan, dtype=np.float64)
            tvec = np.full(3, np.nan, dtype=np.float64)
            distance = float("nan")
            reprojection_rmse = float("nan")

        center = pts.mean(axis=0)
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
            "distortion_model": distortion_model,
            "detection_success": True,
            "pnp_success": bool(ok),
            "pnp_reprojection_rmse_px": reprojection_rmse,
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

        coefficients = list(np.asarray(dist, dtype=np.float64).reshape(-1))
        coefficients += [0.0] * (8 - len(coefficients))
        for index in range(8):
            row[f"d{index}"] = float(coefficients[index])

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
    ap.add_argument(
        "--cameras",
        default=",".join(STATIC_CAMERAS),
        help="Comma-separated static camera IDs, or 'auto' to use PNG basenames.",
    )
    ap.add_argument("--moving-camera-id", default=MOVING_CAMERA)
    ap.add_argument("--allowed-marker-ids", default="auto")
    ap.add_argument("--minimum-area-px2", type=float, default=0.0)
    args = ap.parse_args()

    if args.allowed_marker_ids != "auto" or args.minimum_area_px2 > 0:
        print(
            "[WARN] Detection-time marker filters are deprecated and ignored. "
            "Raw observations are always complete; rigcal applies quality "
            "settings independently for each job."
        )

    dataset = Path(args.dataset)
    if not dataset.exists():
        raise RuntimeError(f"Shared raw dataset not found: {dataset}")

    out = ensure_dir(Path(args.out))
    debug_dir = ensure_dir(out / "debug_images")

    all_rows, static_rows, moving_rows = [], [], []

    cameras = (
        sorted(path.stem for path in (dataset / "static").glob("*.png"))
        if args.cameras == "auto"
        else [value.strip() for value in args.cameras.split(",") if value.strip()]
    )
    if not cameras:
        raise RuntimeError("No static cameras were configured or detected")
    allowed_marker_ids = None
    if args.allowed_marker_ids != "auto":
        allowed_marker_ids = {
            int(value.strip())
            for value in args.allowed_marker_ids.split(",")
            if value.strip()
        }

    for cam in cameras:
        info_path = dataset / "camera_info" / f"{cam}.json"
        multi_dir = dataset / "static_multi" / cam
        image_paths = sorted(multi_dir.glob("*.png")) if multi_dir.is_dir() else []
        if not image_paths:
            image_paths = [dataset / "static" / f"{cam}.png"]

        if not any(path.exists() for path in image_paths):
            print(f"[WARN] missing static image for camera: {cam}")
            continue
        if not info_path.exists():
            print(f"[WARN] missing static camera_info: {info_path}")
            continue

        K, dist, distortion_model = camera_KD(info_path)

        available_images = [path for path in image_paths if path.exists()]
        for index, image_path in enumerate(available_images):
            rows = detect_image(
                image_path=image_path,
                cam_name=cam,
                observer_type="static",
                frame_id=("static" if len(available_images) == 1 else f"static_{index:06d}"),
                K=K,
                dist=dist,
                distortion_model=distortion_model,
                marker_length_m=args.marker_length_m,
                dictionary=args.dictionary,
                debug_path=(
                    debug_dir / "static" / cam / f"{image_path.stem}_detections.png"
                ),
                allowed_marker_ids=allowed_marker_ids,
                minimum_area_px2=args.minimum_area_px2,
            )
            static_rows.extend(rows)
            all_rows.extend(rows)

    moving_dir = dataset / "moving"
    moving_info_path = dataset / "camera_info" / f"{args.moving_camera_id}.json"

    if moving_dir.exists() and moving_info_path.exists():
        K, dist, distortion_model = camera_KD(moving_info_path)

        for image_path in sorted(moving_dir.glob("frame_*.png")):
            try:
                frame_id = int(image_path.stem.split("_")[-1])
            except Exception:
                frame_id = 0

            rows = detect_image(
                image_path=image_path,
                cam_name=args.moving_camera_id,
                observer_type="moving",
                frame_id=frame_id,
                K=K,
                dist=dist,
                distortion_model=distortion_model,
                marker_length_m=args.marker_length_m,
                dictionary=args.dictionary,
                debug_path=debug_dir / "moving" / f"{image_path.stem}_detections.png",
                allowed_marker_ids=allowed_marker_ids,
                minimum_area_px2=args.minimum_area_px2,
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
