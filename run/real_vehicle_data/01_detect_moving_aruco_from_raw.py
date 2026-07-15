#!/usr/bin/env python3
"""Regenerate real moving-camera ArUco observations from canonical raw images.

This stage is intentionally independent from AP01/AP02/AP03. It reads every
``raw_images/moving/frame_*.png`` image, applies the calibrated moving-camera
intrinsics, writes the canonical moving/all observation CSVs, and produces one
annotated debug image per input frame plus a compact contact sheet.

The existing static observation CSV and reference-marker file are preserved.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
from collections import Counter
from pathlib import Path

import cv2
import numpy as np


FIELDS = [
    "observer_type",
    "observer_id",
    "camera_name",
    "frame_id",
    "image_path",
    "marker_id",
    "marker_length_m",
    "fx",
    "fy",
    "cx",
    "cy",
    "pnp_success",
    "rvec_x",
    "rvec_y",
    "rvec_z",
    "tvec_x_m",
    "tvec_y_m",
    "tvec_z_m",
    "distance_m",
    "center_u",
    "center_v",
    "area_px2",
    "corner0_u",
    "corner0_v",
    "corner1_u",
    "corner1_v",
    "corner2_u",
    "corner2_v",
    "corner3_u",
    "corner3_v",
    "distortion_model",
    "d0",
    "d1",
    "d2",
    "d3",
    "d4",
    "d5",
    "d6",
    "d7",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Detect ArUco markers in every real moving-camera raw image and "
            "write canonical observations plus annotated debug images."
        )
    )
    parser.add_argument(
        "--dataset",
        default=(
            "results/real_vehicle_data/real_05x_4k_3hz_v1/00_shared_input"
        ),
    )
    parser.add_argument("--observations-root", default="")
    parser.add_argument("--dictionary", default="DICT_4X4_50")
    parser.add_argument("--marker-length-m", type=float, default=0.17)
    parser.add_argument("--min-marker-id", type=int, default=0)
    parser.add_argument("--max-marker-id", type=int, default=20)
    parser.add_argument(
        "--contact-sheet-count",
        type=int,
        default=24,
        help="Maximum number of evenly spaced debug images in the contact sheet.",
    )
    return parser.parse_args()


def repository_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / ".git").exists():
            return parent
    return Path.cwd().resolve()


def portable_path(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise RuntimeError(f"Missing CSV: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv_atomic(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
            extrasaction="ignore",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})
    temporary.replace(path)


def load_camera_info(path: Path) -> dict:
    if not path.is_file():
        raise RuntimeError(f"Missing moving-camera CameraInfo: {path}")

    data = json.loads(path.read_text(encoding="utf-8"))
    flat_k = data.get("K", data.get("k"))
    if flat_k is None and isinstance(data.get("camera_matrix"), dict):
        flat_k = data["camera_matrix"].get("data")
    if not isinstance(flat_k, list) or len(flat_k) != 9:
        raise RuntimeError(f"CameraInfo K must contain 9 values: {path}")

    distortion = data.get("D", data.get("d"))
    if distortion is None and isinstance(data.get("distortion_coefficients"), dict):
        distortion = data["distortion_coefficients"].get("data")
    if distortion is None:
        distortion = []

    K = np.asarray(flat_k, dtype=np.float64).reshape(3, 3)
    D = np.asarray(distortion, dtype=np.float64).reshape(-1)
    if not np.all(np.isfinite(K)) or K[0, 0] <= 0 or K[1, 1] <= 0:
        raise RuntimeError(f"Invalid moving-camera matrix K: {path}")
    if not np.all(np.isfinite(D)):
        raise RuntimeError(f"Invalid moving-camera distortion D: {path}")

    width = int(data.get("width", data.get("image_width", 0)) or 0)
    height = int(data.get("height", data.get("image_height", 0)) or 0)
    if width <= 0 or height <= 0:
        raise RuntimeError(f"Invalid moving-camera image dimensions: {path}")

    return {
        "K": K,
        "D": D,
        "width": width,
        "height": height,
        "distortion_model": str(data.get("distortion_model", "plumb_bob")),
    }


def make_detector(dictionary_name: str):
    if not hasattr(cv2.aruco, dictionary_name):
        raise RuntimeError(f"Unknown OpenCV ArUco dictionary: {dictionary_name}")

    dictionary = cv2.aruco.getPredefinedDictionary(
        getattr(cv2.aruco, dictionary_name)
    )

    if hasattr(cv2.aruco, "ArucoDetector"):
        parameters = cv2.aruco.DetectorParameters()
        parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        return cv2.aruco.ArucoDetector(dictionary, parameters)

    parameters = cv2.aruco.DetectorParameters_create()
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    return dictionary, parameters


def detect_markers(gray: np.ndarray, detector):
    if hasattr(cv2.aruco, "ArucoDetector"):
        return detector.detectMarkers(gray)
    dictionary, parameters = detector
    return cv2.aruco.detectMarkers(
        gray,
        dictionary,
        parameters=parameters,
    )


def marker_object_points(length_m: float) -> np.ndarray:
    half = length_m / 2.0
    return np.asarray(
        [
            [-half, half, 0.0],
            [half, half, 0.0],
            [half, -half, 0.0],
            [-half, -half, 0.0],
        ],
        dtype=np.float32,
    )


def frame_number(path: Path) -> int:
    match = re.fullmatch(r"frame_(\d+)", path.stem)
    if match is None:
        raise RuntimeError(f"Unexpected moving-frame name: {path.name}")
    return int(match.group(1))


def polygon_area(points: np.ndarray) -> float:
    return abs(
        float(
            cv2.contourArea(
                np.asarray(points, dtype=np.float32).reshape(4, 2)
            )
        )
    )


def solve_marker_pose(
    points: np.ndarray,
    object_points: np.ndarray,
    K: np.ndarray,
    D: np.ndarray,
) -> tuple[bool, np.ndarray, np.ndarray, float]:
    ok, rvec, tvec = cv2.solvePnP(
        object_points,
        points.astype(np.float32),
        K,
        D,
        flags=cv2.SOLVEPNP_IPPE_SQUARE,
    )
    if not ok:
        ok, rvec, tvec = cv2.solvePnP(
            object_points,
            points.astype(np.float32),
            K,
            D,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )

    if not ok:
        return (
            False,
            np.full(3, np.nan, dtype=np.float64),
            np.full(3, np.nan, dtype=np.float64),
            float("nan"),
        )

    rvec = np.asarray(rvec, dtype=np.float64).reshape(3)
    tvec = np.asarray(tvec, dtype=np.float64).reshape(3)
    projected, _ = cv2.projectPoints(
        object_points,
        rvec.reshape(3, 1),
        tvec.reshape(3, 1),
        K,
        D,
    )
    projected = projected.reshape(4, 2)
    rmse = math.sqrt(float(np.mean(np.sum((projected - points) ** 2, axis=1))))
    return True, rvec, tvec, rmse


def put_header(
    image: np.ndarray,
    frame_id: int,
    marker_ids: list[int],
    successful: int,
) -> None:
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (image.shape[1], 92), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.60, image, 0.40, 0.0, image)
    cv2.putText(
        image,
        f"moving frame {frame_id:06d} | detections={len(marker_ids)} | PnP={successful}",
        (24, 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        "marker IDs: " + (", ".join(map(str, marker_ids)) if marker_ids else "none"),
        (24, 73),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.78,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def write_contact_sheet(
    debug_paths: list[Path],
    destination: Path,
    maximum: int,
) -> None:
    if not debug_paths or maximum <= 0:
        return

    sample_count = min(maximum, len(debug_paths))
    positions = np.linspace(0, len(debug_paths) - 1, sample_count, dtype=int)
    thumbnails = []

    for position in positions:
        image = cv2.imread(str(debug_paths[int(position)]), cv2.IMREAD_COLOR)
        if image is None:
            continue
        thumb = cv2.resize(image, (480, 270), interpolation=cv2.INTER_AREA)
        thumbnails.append(thumb)

    if not thumbnails:
        return

    columns = 4
    rows = math.ceil(len(thumbnails) / columns)
    blank = np.zeros_like(thumbnails[0])
    while len(thumbnails) < rows * columns:
        thumbnails.append(blank.copy())

    row_images = [
        np.hstack(thumbnails[index * columns : (index + 1) * columns])
        for index in range(rows)
    ]
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(destination), np.vstack(row_images)):
        raise RuntimeError(f"Could not write contact sheet: {destination}")


def main() -> None:
    args = parse_args()
    repo = repository_root()
    dataset = Path(args.dataset).resolve()
    observations_root = (
        Path(args.observations_root).resolve()
        if args.observations_root
        else dataset / "aruco_observations"
    )

    moving_dir = dataset / "raw_images" / "moving"
    camera_info_path = (
        dataset
        / "raw_images"
        / "camera_info"
        / "moving_calib_camera.json"
    )
    static_csv = observations_root / "shared_static_aruco_observations.csv"
    moving_csv = observations_root / "shared_moving_aruco_observations.csv"
    all_csv = observations_root / "shared_all_aruco_observations.csv"
    frame_summary_csv = observations_root / "MOVING_ARUCO_FRAME_SUMMARY.csv"
    summary_txt = observations_root / "MOVING_ARUCO_DETECTION_SUMMARY.txt"
    debug_root = observations_root / "debug_images" / "moving"
    contact_sheet = (
        observations_root
        / "debug_images"
        / "MOVING_ARUCO_DEBUG_CONTACT_SHEET.jpg"
    )

    if not moving_dir.is_dir():
        raise RuntimeError(f"Missing moving raw-image directory: {moving_dir}")
    if not static_csv.is_file():
        raise RuntimeError(
            "The canonical static observations must exist before refreshing the "
            f"moving observations: {static_csv}"
        )

    camera = load_camera_info(camera_info_path)
    K = camera["K"]
    D = camera["D"]
    distortion_values = list(float(value) for value in D) + [0.0] * 8
    object_points = marker_object_points(args.marker_length_m)
    detector = make_detector(args.dictionary)

    moving_files = sorted(moving_dir.glob("frame_*.png"))
    if not moving_files:
        raise RuntimeError(f"No moving raw images under {moving_dir}")

    numbers = [frame_number(path) for path in moving_files]
    expected = list(range(numbers[0], numbers[0] + len(numbers)))
    if numbers[0] != 0 or numbers != expected:
        raise RuntimeError(
            "Moving raw-image frames must be contiguous from zero: "
            f"first={numbers[0]}, last={numbers[-1]}, count={len(numbers)}"
        )

    shutil.rmtree(debug_root, ignore_errors=True)
    debug_root.mkdir(parents=True, exist_ok=True)
    contact_sheet.unlink(missing_ok=True)

    rows: list[dict] = []
    frame_rows: list[dict] = []
    marker_counts: Counter[int] = Counter()
    successful_pnp = 0
    failed_pnp = 0
    frames_with_markers = 0
    unexpected_ids: set[int] = set()
    debug_paths: list[Path] = []

    print("[INFO] moving raw images:", len(moving_files))
    print("[INFO] source:", moving_dir)
    print(
        "[INFO] camera model:",
        f"{camera['width']}x{camera['height']}",
        camera["distortion_model"],
    )

    for index, image_path in enumerate(moving_files):
        frame_id = frame_number(image_path)
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Could not decode moving raw image: {image_path}")
        if (image.shape[1], image.shape[0]) != (
            camera["width"],
            camera["height"],
        ):
            raise RuntimeError(
                f"{image_path}: size {image.shape[1]}x{image.shape[0]} does not "
                f"match intrinsics {camera['width']}x{camera['height']}"
            )

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = detect_markers(gray, detector)
        ids_flat = [] if ids is None else [int(value) for value in ids.reshape(-1)]

        accepted_corners = []
        accepted_ids = []
        debug = image.copy()
        frame_success = 0
        frame_failure = 0
        frame_rmse_values = []

        for marker_corners, marker_id in zip(corners, ids_flat):
            if not (args.min_marker_id <= marker_id <= args.max_marker_id):
                unexpected_ids.add(marker_id)
                continue

            points = np.asarray(marker_corners, dtype=np.float64).reshape(4, 2)
            ok, rvec, tvec, reprojection_rmse = solve_marker_pose(
                points,
                object_points,
                K,
                D,
            )
            center = points.mean(axis=0)
            area = polygon_area(points)
            distance = float(np.linalg.norm(tvec)) if ok else float("nan")

            row = {
                "observer_type": "moving",
                "observer_id": f"moving_frame_{frame_id:06d}",
                "camera_name": "moving_calib_camera",
                "frame_id": frame_id,
                "image_path": portable_path(image_path, repo),
                "marker_id": marker_id,
                "marker_length_m": args.marker_length_m,
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
                "distance_m": distance if ok else "",
                "center_u": float(center[0]),
                "center_v": float(center[1]),
                "area_px2": area,
                "distortion_model": camera["distortion_model"],
            }
            for corner_index in range(4):
                row[f"corner{corner_index}_u"] = float(points[corner_index, 0])
                row[f"corner{corner_index}_v"] = float(points[corner_index, 1])
            for distortion_index in range(8):
                row[f"d{distortion_index}"] = distortion_values[distortion_index]
            rows.append(row)

            accepted_corners.append(points.astype(np.float32).reshape(1, 4, 2))
            accepted_ids.append(marker_id)
            marker_counts[marker_id] += 1

            if ok:
                successful_pnp += 1
                frame_success += 1
                frame_rmse_values.append(reprojection_rmse)
                cv2.drawFrameAxes(
                    debug,
                    K,
                    D,
                    rvec.reshape(3, 1),
                    tvec.reshape(3, 1),
                    args.marker_length_m * 0.5,
                    3,
                )
                label = f"id={marker_id} z={tvec[2]:.2f}m e={reprojection_rmse:.2f}px"
                color = (0, 255, 0)
            else:
                failed_pnp += 1
                frame_failure += 1
                label = f"id={marker_id} PnP FAILED"
                color = (0, 0, 255)

            text_position = (
                max(5, int(center[0]) - 100),
                max(110, int(center[1]) - 15),
            )
            cv2.putText(
                debug,
                label,
                text_position,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                color,
                2,
                cv2.LINE_AA,
            )

        if accepted_ids:
            frames_with_markers += 1
            cv2.aruco.drawDetectedMarkers(
                debug,
                accepted_corners,
                np.asarray(accepted_ids, dtype=np.int32).reshape(-1, 1),
            )

        accepted_ids = sorted(accepted_ids)
        put_header(debug, frame_id, accepted_ids, frame_success)
        debug_path = debug_root / image_path.name
        if not cv2.imwrite(str(debug_path), debug):
            raise RuntimeError(f"Could not write debug image: {debug_path}")
        debug_paths.append(debug_path)

        frame_rows.append(
            {
                "frame_id": frame_id,
                "image_path": portable_path(image_path, repo),
                "debug_image": portable_path(debug_path, repo),
                "detected_marker_count": len(accepted_ids),
                "detected_marker_ids": " ".join(map(str, accepted_ids)),
                "pnp_success_count": frame_success,
                "pnp_failure_count": frame_failure,
                "median_pnp_reprojection_rmse_px": (
                    float(np.median(frame_rmse_values))
                    if frame_rmse_values
                    else ""
                ),
            }
        )

        if index % 25 == 0 or index + 1 == len(moving_files):
            print(
                "[SCAN]",
                f"{index + 1}/{len(moving_files)}",
                f"frame={frame_id:06d}",
                f"markers={accepted_ids}",
            )

    if successful_pnp == 0:
        raise RuntimeError("No successful moving-camera ArUco PnP observations")

    static_rows = read_csv(static_csv)
    write_csv_atomic(moving_csv, rows, FIELDS)
    write_csv_atomic(all_csv, static_rows + rows, FIELDS)
    write_csv_atomic(
        frame_summary_csv,
        frame_rows,
        [
            "frame_id",
            "image_path",
            "debug_image",
            "detected_marker_count",
            "detected_marker_ids",
            "pnp_success_count",
            "pnp_failure_count",
            "median_pnp_reprojection_rmse_px",
        ],
    )
    write_contact_sheet(
        debug_paths,
        contact_sheet,
        args.contact_sheet_count,
    )

    marker_lines = [
        f"- marker {marker_id}: {marker_counts[marker_id]} observations"
        for marker_id in sorted(marker_counts)
    ]
    summary = [
        "REAL MOVING-CAMERA ARUCO DETECTION",
        "=" * 80,
        "",
        f"Dataset: {portable_path(dataset, repo)}",
        f"Raw moving directory: {portable_path(moving_dir, repo)}",
        f"CameraInfo: {portable_path(camera_info_path, repo)}",
        f"Dictionary: {args.dictionary}",
        f"Marker length [m]: {args.marker_length_m}",
        f"Allowed marker IDs: {args.min_marker_id}..{args.max_marker_id}",
        f"Image size: {camera['width']}x{camera['height']}",
        f"Distortion model: {camera['distortion_model']}",
        "",
        f"Moving frames scanned: {len(moving_files)}",
        f"Frames with accepted markers: {frames_with_markers}",
        f"Frames without accepted markers: {len(moving_files) - frames_with_markers}",
        f"Moving observations written: {len(rows)}",
        f"PnP successes: {successful_pnp}",
        f"PnP failures: {failed_pnp}",
        f"Observed marker IDs: {sorted(marker_counts)}",
        f"Unexpected marker IDs rejected: {sorted(unexpected_ids)}",
        "",
        "Observations per marker:",
        *marker_lines,
        "",
        "Outputs:",
        f"- {portable_path(moving_csv, repo)}",
        f"- {portable_path(all_csv, repo)}",
        f"- {portable_path(frame_summary_csv, repo)}",
        f"- {portable_path(debug_root, repo)}",
        f"- {portable_path(contact_sheet, repo)}",
        "",
        "Debug annotation:",
        "- green marker outline and coordinate axes: successful PnP",
        "- label z: estimated marker depth in metres",
        "- label e: four-corner PnP reprojection RMSE in pixels",
        "",
        "[OK] moving-camera ArUco observations and debug images regenerated",
    ]
    summary_txt.write_text("\n".join(summary) + "\n", encoding="utf-8")

    print()
    print(summary_txt.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
