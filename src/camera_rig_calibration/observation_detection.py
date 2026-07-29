#!/usr/bin/env python3
"""Create the canonical, unfiltered ArUco observation contract."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from camera_rig_calibration.methods.common.aruco_utils import (
    DETECTOR_CONTRACT,
    detect_markers_in_image,
    draw_marker_debug,
    effective_detector_config,
    marker_object_points,
)
from camera_rig_calibration.methods.common.camera_io import load_camera_info_json
from camera_rig_calibration.methods.common.io_utils import ensure_dir, write_csv


MOVING_CAMERA = "moving_calib_camera"


def fields():
    base = [
        "observer_type", "observer_id", "camera_name", "frame_id", "image_path",
        "marker_id", "marker_length_m",
        "detection_mode", "detection_source", "detection_support",
        "detector_contract", "opencv_version", "detector_parameters_json",
        "fx", "fy", "cx", "cy", "distortion_model",
        "detection_success", "pnp_success", "pnp_reprojection_rmse_px",
        "rvec_x", "rvec_y", "rvec_z",
        "tvec_x_m", "tvec_y_m", "tvec_z_m",
        "distance_m", "center_u", "center_v",
        "image_width_px", "image_height_px", "area_px2",
        "marker_area_ratio",
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
    detection_mode,
    debug_path,
    allowed_marker_ids=None,
    minimum_area_px2=0.0,
):
    image, detections, diagnostics = detect_markers_in_image(
        image_path,
        dictionary,
        detection_mode,
        return_diagnostics=True,
    )
    rows = []
    detector_config = effective_detector_config(detection_mode, dictionary)
    detector_parameters_json = json.dumps(
        {
            key: detector_config[key]
            for key in (
                "parameters",
                "gamma_passes",
                "support_rule",
            )
            if key in detector_config
        },
        sort_keys=True,
        separators=(",", ":"),
    )

    objp = marker_object_points(marker_length_m).astype(np.float32)
    image_height_px, image_width_px = image.shape[:2]

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
            "detection_mode": detection_mode,
            "detection_source": det["detection_source"],
            "detection_support": det["detection_support"],
            "detector_contract": DETECTOR_CONTRACT,
            "opencv_version": cv2.__version__,
            "detector_parameters_json": detector_parameters_json,
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
            "image_width_px": int(image_width_px),
            "image_height_px": int(image_height_px),
            "area_px2": area,
            "marker_area_ratio": area / float(image_width_px * image_height_px),
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

    return rows, [
        {
            "image_path": str(image_path),
            "observer_type": observer_type,
            "camera_name": cam_name,
            "frame_id": frame_id,
            "detection_mode": detection_mode,
            **item,
        }
        for item in diagnostics
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--marker-length-m", type=float, default=0.17)
    ap.add_argument("--dictionary", default="DICT_4X4_50")
    ap.add_argument(
        "--detection-mode",
        choices=("baseline", "subpixel_refined", "high_sensitivity"),
        default="baseline",
    )
    ap.add_argument(
        "--cameras",
        default="auto",
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
    debug_dir = ensure_dir(out / "debug_images" / args.detection_mode)

    all_rows, static_rows, moving_rows = [], [], []
    diagnostic_rows = []

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
            rows, diagnostics = detect_image(
                image_path=image_path,
                cam_name=cam,
                observer_type="static",
                frame_id=("static" if len(available_images) == 1 else f"static_{index:06d}"),
                K=K,
                dist=dist,
                distortion_model=distortion_model,
                marker_length_m=args.marker_length_m,
                dictionary=args.dictionary,
                detection_mode=args.detection_mode,
                debug_path=(
                    debug_dir / "static" / cam / f"{image_path.stem}_detections.png"
                ),
                allowed_marker_ids=allowed_marker_ids,
                minimum_area_px2=args.minimum_area_px2,
            )
            static_rows.extend(rows)
            all_rows.extend(rows)
            diagnostic_rows.extend(diagnostics)

    moving_dir = dataset / "moving"
    moving_info_path = dataset / "camera_info" / f"{args.moving_camera_id}.json"

    if moving_dir.exists() and moving_info_path.exists():
        K, dist, distortion_model = camera_KD(moving_info_path)
        moving_paths = sorted(moving_dir.glob("frame_*.png"))
        progress_interval = max(1, len(moving_paths) // 20)
        print(
            f"[INFO] ArUco observation scan: {len(moving_paths)} moving frames",
            flush=True,
        )
        for moving_index, image_path in enumerate(moving_paths, 1):
            try:
                frame_id = int(image_path.stem.split("_")[-1])
            except Exception:
                frame_id = 0

            rows, diagnostics = detect_image(
                image_path=image_path,
                cam_name=args.moving_camera_id,
                observer_type="moving",
                frame_id=frame_id,
                K=K,
                dist=dist,
                distortion_model=distortion_model,
                marker_length_m=args.marker_length_m,
                dictionary=args.dictionary,
                detection_mode=args.detection_mode,
                debug_path=debug_dir / "moving" / f"{image_path.stem}_detections.png",
                allowed_marker_ids=allowed_marker_ids,
                minimum_area_px2=args.minimum_area_px2,
            )
            moving_rows.extend(rows)
            all_rows.extend(rows)
            diagnostic_rows.extend(diagnostics)
            if (
                moving_index == 1
                or moving_index % progress_interval == 0
                or moving_index == len(moving_paths)
            ):
                print(
                    "RIGCAL_PROGRESS "
                    f"current={moving_index} total={len(moving_paths)} "
                    "unit=frames label=ArUco observations",
                    flush=True,
                )
    else:
        print(f"[WARN] moving data incomplete: {moving_dir}, {moving_info_path}")

    write_csv(out / "shared_static_aruco_observations.csv", static_rows, fields())
    write_csv(out / "shared_moving_aruco_observations.csv", moving_rows, fields())
    write_csv(out / "shared_all_aruco_observations.csv", all_rows, fields())
    diagnostic_fields = [
        "image_path",
        "observer_type",
        "camera_name",
        "frame_id",
        "detection_mode",
        "pass",
        "marker_id",
        "area_px2",
        "center_u",
        "center_v",
        "accepted",
        "reason",
    ]
    write_csv(
        out / "detection_candidates.csv",
        diagnostic_rows,
        diagnostic_fields,
    )
    (out / "detection_candidates.json").write_text(
        json.dumps(diagnostic_rows, indent=2) + "\n",
        encoding="utf-8",
    )
    (out / "effective_detection_config.json").write_text(
        json.dumps(
            effective_detector_config(args.detection_mode, args.dictionary),
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    pnp_ok = sum(1 for r in all_rows if str(r["pnp_success"]) == "True")

    summary = [
        "Shared ArUco detection summary",
        "==============================",
        "",
        f"Dataset: {dataset}",
        f"Dictionary: {args.dictionary}",
        f"Detection mode: {args.detection_mode}",
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
