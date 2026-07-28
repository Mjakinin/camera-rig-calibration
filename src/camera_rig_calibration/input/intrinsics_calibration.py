#!/usr/bin/env python3
"""Calibrate a managed camera-intrinsics profile from video or images."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import time
from pathlib import Path

import cv2
import numpy as np

try:
    from camera_rig_calibration.input.video_geometry import open_oriented_video
except ImportError:  # pragma: no cover - direct script fallback
    from video_geometry import open_oriented_video


def open_video_without_autorotation(path):
    """Compatibility name; returned frames now use the canonical display transform."""
    return open_oriented_video(path)


def balanced_candidate_indices(
    frame_count: int,
    source_fps: float,
    target_hz: float,
    *,
    tested: set[int] | None = None,
) -> list[int]:
    if frame_count <= 0:
        return []
    step = max(1, int(round(max(source_fps, target_hz) / target_hz)))
    excluded = tested or set()
    return [
        frame_index
        for frame_index in range(0, frame_count, step)
        if frame_index not in excluded
    ]


def detect_checkerboard_balanced(
    gray: np.ndarray,
    pattern: tuple[int, int],
    preview_max_dimension: int,
) -> tuple[bool, np.ndarray | None]:
    height, width = gray.shape[:2]
    scale = min(
        1.0,
        float(preview_max_dimension) / float(max(width, height)),
    )
    preview = (
        cv2.resize(
            gray,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_AREA,
        )
        if scale < 1.0
        else gray
    )
    found, corners = cv2.findChessboardCornersSB(
        preview,
        pattern,
        flags=cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_EXHAUSTIVE,
    )
    if not found or corners is None:
        return False, None
    full_resolution = (corners / scale).astype(np.float32)
    cv2.cornerSubPix(
        gray,
        full_resolution,
        (7, 7),
        (-1, -1),
        (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER,
            30,
            0.01,
        ),
    )
    return True, full_resolution


def _detect_candidates(
    video: Path,
    *,
    candidate_indices: list[int],
    pass_label: str,
    pattern: tuple[int, int],
    reported_frames: int,
    source_fps: float,
    columns: int,
    rows: int,
    preview_max_dimension: int,
    detections: list[dict],
) -> int:
    if not candidate_indices:
        return 0
    candidate_set = set(candidate_indices)
    capture = open_video_without_autorotation(video)
    tested = 0
    next_progress_frame = 0
    frame_index = 0
    while frame_index < reported_frames:
        ok = capture.grab()
        if not ok:
            break
        if frame_index not in candidate_set:
            frame_index += 1
            continue
        ok, frame = capture.retrieve()
        if not ok:
            frame_index += 1
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, corners = detect_checkerboard_balanced(
            gray, pattern, preview_max_dimension
        )
        if found and corners is not None:
            metrics = board_metrics(
                gray,
                corners,
                columns,
                rows,
                frame_index,
                max(reported_frames, frame_index + 1),
            )
            detections.append(
                {
                    "frame_index": frame_index,
                    "time_s": frame_index / max(source_fps, 1e-9),
                    "corners": corners.reshape(-1, 1, 2),
                    **metrics,
                }
            )
        tested += 1
        if frame_index >= next_progress_frame:
            print(
                "RIGCAL_PROGRESS "
                f"current={min(frame_index + 1, reported_frames)} "
                f"total={reported_frames} unit=frames "
                f"label=intrinsics_{pass_label}",
                flush=True,
            )
            next_progress_frame = frame_index + 50
        frame_index += 1
    capture.release()
    return tested



def board_metrics(
    gray: np.ndarray,
    corners: np.ndarray,
    cols: int,
    rows: int,
    frame_index: int,
    frame_count: int,
) -> dict:
    height, width = gray.shape[:2]
    points = corners.reshape(-1, 2).astype(np.float64)

    center = points.mean(axis=0)
    hull = cv2.convexHull(points.astype(np.float32))
    area = abs(float(cv2.contourArea(hull)))
    area_fraction = area / float(width * height)

    x, y, w, h = cv2.boundingRect(points.astype(np.float32))
    padding = 30

    x0 = max(0, x - padding)
    y0 = max(0, y - padding)
    x1 = min(width, x + w + padding)
    y1 = min(height, y + h + padding)

    roi = gray[y0:y1, x0:x1]

    sharpness = float(
        cv2.Laplacian(roi, cv2.CV_64F).var()
    )

    horizontal = points[cols - 1] - points[0]
    angle = math.atan2(horizontal[1], horizontal[0])

    top = np.linalg.norm(points[cols - 1] - points[0])
    bottom = np.linalg.norm(points[-1] - points[-cols])
    left = np.linalg.norm(points[(rows - 1) * cols] - points[0])
    right = np.linalg.norm(points[-1] - points[cols - 1])

    top_bottom = math.log(
        max(top, 1e-9) / max(bottom, 1e-9)
    )

    left_right = math.log(
        max(left, 1e-9) / max(right, 1e-9)
    )

    time_fraction = frame_index / max(frame_count - 1, 1)

    feature = np.array(
        [
            center[0] / width,
            center[1] / height,
            math.log(max(area_fraction, 1e-9)),
            math.sin(2.0 * angle),
            math.cos(2.0 * angle),
            top_bottom,
            left_right,
            0.25 * time_fraction,
        ],
        dtype=np.float64,
    )

    return {
        "center_x": float(center[0]),
        "center_y": float(center[1]),
        "area_fraction": float(area_fraction),
        "angle_deg": float(math.degrees(angle)),
        "top_bottom_log_ratio": float(top_bottom),
        "left_right_log_ratio": float(left_right),
        "sharpness": sharpness,
        "feature": feature,
    }


def select_diverse(
    detections: list[dict],
    maximum: int,
    minimum_frame_gap: int,
) -> list[int]:
    if len(detections) <= maximum:
        return list(range(len(detections)))

    features = np.vstack(
        [item["feature"] for item in detections]
    )

    mean = features.mean(axis=0)
    std = features.std(axis=0)
    std[std < 1e-9] = 1.0

    normalized = (features - mean) / std

    sharpness = np.asarray(
        [item["sharpness"] for item in detections],
        dtype=np.float64,
    )

    sharpness = (
        sharpness - sharpness.min()
    ) / max(float(np.ptp(sharpness)), 1e-9)

    selected = [int(np.argmax(sharpness))]

    while len(selected) < maximum:
        selected_features = normalized[selected]

        distances = np.linalg.norm(
            normalized[:, None, :]
            - selected_features[None, :, :],
            axis=2,
        )

        score = distances.min(axis=1) + 0.15 * sharpness
        score[selected] = -np.inf

        order = np.argsort(score)[::-1]
        chosen = None

        for candidate in order:
            candidate = int(candidate)
            candidate_frame = detections[candidate]["frame_index"]

            sufficiently_separated = all(
                abs(
                    candidate_frame
                    - detections[index]["frame_index"]
                ) >= minimum_frame_gap
                for index in selected
            )

            if sufficiently_separated:
                chosen = candidate
                break

        if chosen is None:
            chosen = int(order[0])

        selected.append(chosen)

    return sorted(
        selected,
        key=lambda index: detections[index]["frame_index"],
    )


def object_points(cols: int, rows: int) -> np.ndarray:
    points = np.zeros(
        (cols * rows, 3),
        dtype=np.float32,
    )

    points[:, :2] = np.mgrid[
        0:cols,
        0:rows,
    ].T.reshape(-1, 2)

    return points


def calibrate(
    detections: list[dict],
    indices: list[int],
    object_template: np.ndarray,
    image_size: tuple[int, int],
    rational: bool,
) -> dict:
    objects = [
        object_template.copy()
        for _ in indices
    ]

    images = [
        detections[index]["corners"].astype(np.float32)
        for index in indices
    ]

    flags = cv2.CALIB_RATIONAL_MODEL if rational else 0

    rms, matrix, distortion, rvecs, tvecs = (
        cv2.calibrateCamera(
            objects,
            images,
            image_size,
            None,
            None,
            flags=flags,
        )
    )

    per_view = []

    for obj, img, rvec, tvec in zip(
        objects,
        images,
        rvecs,
        tvecs,
    ):
        projected, _ = cv2.projectPoints(
            obj,
            rvec,
            tvec,
            matrix,
            distortion,
        )

        difference = (
            img.reshape(-1, 2)
            - projected.reshape(-1, 2)
        )

        per_view.append(
            float(
                np.sqrt(
                    np.mean(
                        np.sum(
                            difference * difference,
                            axis=1,
                        )
                    )
                )
            )
        )

    return {
        "rms": float(rms),
        "K": matrix,
        "D": distortion.reshape(-1),
        "per_view": per_view,
        "rational": rational,
    }


def holdout_error(
    detections: list[dict],
    indices: list[int],
    object_template: np.ndarray,
    K: np.ndarray,
    D: np.ndarray,
) -> list[float]:
    errors = []

    for index in indices:
        corners = detections[index]["corners"].astype(
            np.float32
        )

        ok, rvec, tvec = cv2.solvePnP(
            object_template,
            corners,
            K,
            D,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )

        if not ok:
            continue

        projected, _ = cv2.projectPoints(
            object_template,
            rvec,
            tvec,
            K,
            D,
        )

        difference = (
            corners.reshape(-1, 2)
            - projected.reshape(-1, 2)
        )

        errors.append(
            float(
                np.sqrt(
                    np.mean(
                        np.sum(
                            difference * difference,
                            axis=1,
                        )
                    )
                )
            )
        )

    return errors


def model_comparison(
    detections: list[dict],
    selected: list[int],
    object_template: np.ndarray,
    image_size: tuple[int, int],
) -> tuple[bool, dict]:
    holdout = selected[::5]
    holdout_set = set(holdout)

    training = [
        index
        for index in selected
        if index not in holdout_set
    ]

    if len(training) < 20 or len(holdout) < 5:
        training = selected
        holdout = selected

    standard = calibrate(
        detections,
        training,
        object_template,
        image_size,
        rational=False,
    )

    rational = calibrate(
        detections,
        training,
        object_template,
        image_size,
        rational=True,
    )

    standard_errors = holdout_error(
        detections,
        holdout,
        object_template,
        standard["K"],
        standard["D"],
    )

    rational_errors = holdout_error(
        detections,
        holdout,
        object_template,
        rational["K"],
        rational["D"],
    )

    standard_median = float(
        np.median(standard_errors)
    )

    rational_median = float(
        np.median(rational_errors)
    )

    use_rational = rational_median < 0.98 * standard_median

    comparison = {
        "training_views": len(training),
        "holdout_views": len(holdout),
        "standard_holdout_median_rmse_px": standard_median,
        "rational_holdout_median_rmse_px": rational_median,
        "selected_model": (
            "rational_polynomial"
            if use_rational
            else "plumb_bob"
        ),
    }

    return use_rational, comparison


def calibrate_with_outlier_filter(
    detections: list[dict],
    selected: list[int],
    object_template: np.ndarray,
    image_size: tuple[int, int],
    rational: bool,
) -> tuple[list[int], dict, list[dict]]:
    active = list(selected)
    removed = []

    while True:
        result = calibrate(
            detections,
            active,
            object_template,
            image_size,
            rational=rational,
        )

        errors = np.asarray(
            result["per_view"],
            dtype=np.float64,
        )

        median = float(np.median(errors))
        mad = float(
            np.median(
                np.abs(errors - median)
            )
        )

        robust_limit = median + 3.0 * max(
            1.4826 * mad,
            0.05,
        )

        threshold = max(1.5, robust_limit)

        worst_position = int(np.argmax(errors))
        worst_error = float(errors[worst_position])

        if (
            worst_error <= threshold
            or len(active) <= 30
            or len(removed) >= 15
        ):
            result["median_view_error"] = median
            result["maximum_view_error"] = worst_error
            result["outlier_threshold"] = threshold
            return active, result, removed

        removed_index = active.pop(worst_position)

        removed.append(
            {
                "detection_index": removed_index,
                "frame_index": detections[
                    removed_index
                ]["frame_index"],
                "reprojection_rmse_px": worst_error,
            }
        )

        print(
            "[FILTER]",
            f"frame={detections[removed_index]['frame_index']}",
            f"rmse={worst_error:.4f}px",
        )


def write_contact_sheet(
    debug_paths: list[Path],
    destination: Path,
) -> None:
    if not debug_paths:
        return

    sample_count = min(20, len(debug_paths))

    positions = np.linspace(
        0,
        len(debug_paths) - 1,
        sample_count,
        dtype=int,
    )

    thumbs = []

    for position in positions:
        image = cv2.imread(str(debug_paths[position]))

        if image is None:
            continue

        source_height, source_width = image.shape[:2]
        scale = min(480.0 / source_width, 480.0 / source_height)
        target_width = max(1, int(round(source_width * scale)))
        target_height = max(1, int(round(source_height * scale)))
        resized = cv2.resize(
            image,
            (target_width, target_height),
            interpolation=cv2.INTER_AREA,
        )
        thumbnail = np.zeros((480, 480, 3), dtype=np.uint8)
        x0 = (480 - target_width) // 2
        y0 = (480 - target_height) // 2
        thumbnail[y0:y0 + target_height, x0:x0 + target_width] = resized
        thumbs.append(thumbnail)

    if not thumbs:
        return

    columns = 4
    rows = math.ceil(len(thumbs) / columns)

    blank = np.zeros_like(thumbs[0])

    while len(thumbs) < rows * columns:
        thumbs.append(blank.copy())

    row_images = []

    for row in range(rows):
        start = row * columns
        row_images.append(
            np.hstack(
                thumbs[start:start + columns]
            )
        )

    sheet = np.vstack(row_images)
    cv2.imwrite(str(destination), sheet)


def main() -> None:
    parser = argparse.ArgumentParser()

    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--video")
    source_group.add_argument("--images")
    parser.add_argument("--out", required=True)
    parser.add_argument("--cols", type=int, default=8)
    parser.add_argument("--rows", type=int, default=6)
    parser.add_argument("--max-views", type=int, default=80)
    parser.add_argument("--minimum-frame-gap", type=int, default=5)
    parser.add_argument("--minimum-detections", type=int, default=20)
    parser.add_argument(
        "--scan-mode",
        choices=("balanced", "full_frame"),
        default="balanced",
    )
    parser.add_argument("--scan-target-hz", type=float, default=3.0)
    parser.add_argument("--preview-max-dimension", type=int, default=1920)

    args = parser.parse_args()

    video = Path(args.video).resolve() if args.video else None
    image_root = Path(args.images).resolve() if args.images else None
    out = Path(args.out)

    if video is not None and not video.is_file():
        raise RuntimeError(
            f"Video not found: {video}"
        )
    image_paths = (
        [
            path
            for path in sorted(image_root.iterdir())
            if path.is_file()
            and path.suffix.lower()
            in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
        ]
        if image_root is not None and image_root.is_dir()
        else []
    )
    if image_root is not None and not image_paths:
        raise RuntimeError(
            f"Checkerboard image folder contains no supported images: {image_root}"
        )

    shutil.rmtree(out, ignore_errors=True)

    selected_dir = out / "selected_frames"
    debug_dir = out / "debug_selected"
    undistorted_dir = out / "undistorted_samples"

    selected_dir.mkdir(parents=True)
    debug_dir.mkdir(parents=True)
    undistorted_dir.mkdir(parents=True)

    video_geometry = None
    if video is not None:
        capture = open_video_without_autorotation(video)
        if not capture.isOpened():
            raise RuntimeError(
                f"Could not open video: {video}"
            )
        reported_frames = int(
            capture.get(cv2.CAP_PROP_FRAME_COUNT)
        )
        source_fps = float(
            capture.get(cv2.CAP_PROP_FPS)
        )
        width = int(
            capture.get(cv2.CAP_PROP_FRAME_WIDTH)
        )
        height = int(
            capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
        )
        video_geometry = capture.geometry
        capture.release()
        source = video
    else:
        first_image = cv2.imread(str(image_paths[0]), cv2.IMREAD_COLOR)
        if first_image is None:
            raise RuntimeError(f"Could not read checkerboard image: {image_paths[0]}")
        height, width = first_image.shape[:2]
        reported_frames = len(image_paths)
        source_fps = 0.0
        source = image_root

    if width <= 0 or height <= 0:
        raise RuntimeError(f"Invalid source dimensions: {width}x{height}")

    pattern = (args.cols, args.rows)

    detections = []
    scanned_frames = 0

    print("[INFO] source:", source, flush=True)
    print(
        "[INFO] source type:",
        "video" if video is not None else "checkerboard image folder",
        flush=True,
    )
    print("[INFO] resolution:", width, "x", height, flush=True)
    if video is not None:
        print("[INFO] source FPS:", source_fps, flush=True)
    print("[INFO] reported frames:", reported_frames, flush=True)
    print("[INFO] checkerboard:", pattern, flush=True)
    print("[INFO] scan mode:", args.scan_mode, flush=True)
    if image_paths and args.scan_mode == "balanced":
        print(
            "[INFO] image-folder scan: every supplied image,",
            f"preview max dimension={args.preview_max_dimension},",
            "full-resolution corner refinement",
            flush=True,
        )
    elif args.scan_mode == "balanced":
        print(
            "[INFO] balanced scan:",
            f"{args.scan_target_hz:g}/"
            f"{2 * args.scan_target_hz:g}/"
            f"{4 * args.scan_target_hz:g} Hz adaptive passes,",
            f"preview max dimension={args.preview_max_dimension}",
            flush=True,
        )
    else:
        print(
            "[INFO] full-frame scan: every original frame at full resolution",
            flush=True,
        )
    scan_started = time.monotonic()
    print("RIGCAL_STAGE_START intrinsic_checkerboard_scan", flush=True)

    tested_indices: set[int] = set()
    if image_paths:
        detector_flags = (
            cv2.CALIB_CB_NORMALIZE_IMAGE
            | cv2.CALIB_CB_EXHAUSTIVE
            | cv2.CALIB_CB_ACCURACY
        )
        for frame_index, image_path in enumerate(image_paths):
            frame = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if frame is None:
                raise RuntimeError(f"Could not read checkerboard image: {image_path}")
            frame_height, frame_width = frame.shape[:2]
            if (frame_width, frame_height) != (width, height):
                raise RuntimeError(
                    "All checkerboard images must have the same resolution: "
                    f"expected {width}x{height}, got {frame_width}x{frame_height} "
                    f"for {image_path}"
                )
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if args.scan_mode == "balanced":
                found, corners = detect_checkerboard_balanced(
                    gray,
                    pattern,
                    preview_max_dimension=args.preview_max_dimension,
                )
            else:
                found, corners = cv2.findChessboardCornersSB(
                    gray, pattern, flags=detector_flags
                )
            if found:
                metrics = board_metrics(
                    gray,
                    corners,
                    args.cols,
                    args.rows,
                    frame_index,
                    reported_frames,
                )
                detections.append(
                    {
                        "frame_index": frame_index,
                        "time_s": "",
                        "source_name": image_path.name,
                        "corners": corners.reshape(-1, 1, 2),
                        **metrics,
                    }
                )
            if frame_index % 10 == 0 or frame_index + 1 == reported_frames:
                print(
                    "RIGCAL_PROGRESS "
                    f"current={frame_index + 1} total={reported_frames} "
                    "unit=frames label=intrinsics_image_folder",
                    flush=True,
                )
            tested_indices.add(frame_index)
            scanned_frames += 1
    elif args.scan_mode == "balanced":
        assert video is not None
        desired_detections = min(
            reported_frames,
            max(
                args.minimum_detections,
                args.max_views + max(10, args.max_views // 4),
            ),
        )
        for pass_hz in (
            args.scan_target_hz,
            2.0 * args.scan_target_hz,
            4.0 * args.scan_target_hz,
        ):
            candidates = balanced_candidate_indices(
                reported_frames,
                source_fps,
                pass_hz,
                tested=tested_indices,
            )
            scanned_frames += _detect_candidates(
                video,
                candidate_indices=candidates,
                pass_label=f"{pass_hz:g}Hz",
                pattern=pattern,
                reported_frames=reported_frames,
                source_fps=source_fps,
                columns=args.cols,
                rows=args.rows,
                preview_max_dimension=args.preview_max_dimension,
                detections=detections,
            )
            tested_indices.update(candidates)
            if len(detections) >= desired_detections:
                break
    else:
        assert video is not None
        detector_flags = (
            cv2.CALIB_CB_NORMALIZE_IMAGE
            | cv2.CALIB_CB_EXHAUSTIVE
            | cv2.CALIB_CB_ACCURACY
        )
        capture = open_video_without_autorotation(video)
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            found, corners = cv2.findChessboardCornersSB(
                gray, pattern, flags=detector_flags
            )
            if found:
                metrics = board_metrics(
                    gray,
                    corners,
                    args.cols,
                    args.rows,
                    scanned_frames,
                    max(reported_frames, scanned_frames + 1),
                )
                detections.append(
                    {
                        "frame_index": scanned_frames,
                        "time_s": scanned_frames / max(source_fps, 1e-9),
                        "corners": corners.reshape(-1, 1, 2),
                        **metrics,
                    }
                )
            if scanned_frames % 50 == 0:
                print(
                    "RIGCAL_PROGRESS "
                    f"current={min(scanned_frames + 1, reported_frames)} "
                    f"total={reported_frames} unit=frames "
                    "label=intrinsics_exhaustive",
                    flush=True,
                )
            tested_indices.add(scanned_frames)
            scanned_frames += 1
        capture.release()
    print(
        "RIGCAL_STAGE_END intrinsic_checkerboard_scan "
        f"elapsed_seconds={time.monotonic() - scan_started:.3f}",
        flush=True,
    )

    if len(detections) < args.minimum_detections:
        raise RuntimeError(
            "Insufficient checkerboard detections: "
            f"{len(detections)} < {args.minimum_detections}"
        )

    calibration_started = time.monotonic()
    print("RIGCAL_STAGE_START intrinsic_model_calibration", flush=True)
    selected = select_diverse(
        detections,
        maximum=min(args.max_views, len(detections)),
        minimum_frame_gap=args.minimum_frame_gap,
    )

    object_template = object_points(
        args.cols,
        args.rows,
    )

    use_rational, comparison = model_comparison(
        detections,
        selected,
        object_template,
        (width, height),
    )

    final_indices, result, removed = (
        calibrate_with_outlier_filter(
            detections,
            selected,
            object_template,
            (width, height),
            rational=use_rational,
        )
    )
    print(
        "RIGCAL_STAGE_END intrinsic_model_calibration "
        f"elapsed_seconds={time.monotonic() - calibration_started:.3f}",
        flush=True,
    )

    K = result["K"]
    raw_D = result["D"]

    if use_rational:
        D = np.zeros(8, dtype=np.float64)
        D[:min(8, len(raw_D))] = raw_D[:8]
        distortion_model = "rational_polynomial"
    else:
        D = np.zeros(5, dtype=np.float64)
        D[:min(5, len(raw_D))] = raw_D[:5]
        distortion_model = "plumb_bob"

    final_detection_by_frame = {
        detections[index]["frame_index"]: index
        for index in final_indices
    }

    per_view_error = {
        index: error
        for index, error in zip(
            final_indices,
            result["per_view"],
        )
    }

    export_started = time.monotonic()
    print("RIGCAL_STAGE_START intrinsic_artifact_export", flush=True)
    debug_paths = []

    sample_frames = set(
        detections[index]["frame_index"]
        for index in (
            final_indices[::max(1, len(final_indices) // 5)]
        )[:5]
    )

    def export_frame(frame_index, frame):
        detection_index = final_detection_by_frame.get(
            frame_index
        )
        if detection_index is not None:
            filename = f"frame_{frame_index:06d}.png"
            selected_path = selected_dir / filename
            debug_path = debug_dir / filename
            cv2.imwrite(
                str(selected_path),
                frame,
            )
            debug = frame.copy()
            cv2.drawChessboardCorners(
                debug,
                pattern,
                detections[
                    detection_index
                ]["corners"],
                True,
            )
            cv2.putText(
                debug,
                (
                    f"frame={frame_index} "
                    f"rmse={per_view_error[detection_index]:.3f}px"
                ),
                (50, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                (0, 255, 0),
                3,
                cv2.LINE_AA,
            )
            cv2.imwrite(
                str(debug_path),
                debug,
            )
            debug_paths.append(debug_path)
            if frame_index in sample_frames:
                undistorted = cv2.undistort(
                    frame,
                    K,
                    D,
                )

                cv2.imwrite(
                    str(
                        undistorted_dir
                        / filename
                    ),
                    undistorted,
                )

    if image_paths:
        for frame_index in sorted(final_detection_by_frame):
            frame = cv2.imread(str(image_paths[frame_index]), cv2.IMREAD_COLOR)
            if frame is None:
                raise RuntimeError(
                    f"Could not read checkerboard image: {image_paths[frame_index]}"
                )
            export_frame(frame_index, frame)
            print(
                "[EXPORT]",
                f"selected={len(debug_paths)}/{len(final_detection_by_frame)}",
                f"source_image={image_paths[frame_index].name}",
                flush=True,
            )
    else:
        assert video is not None
        capture = open_video_without_autorotation(video)
        frame_index = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            export_frame(frame_index, frame)
            frame_index += 1
            if frame_index % 250 == 0:
                progress = (
                    100.0 * frame_index / reported_frames
                    if reported_frames > 0
                    else 0.0
                )
                print(
                    "[EXPORT]",
                    (
                        f"frame={frame_index}/{reported_frames}"
                        if reported_frames > 0
                        else f"frame={frame_index}"
                    ),
                    f"progress={progress:.1f}%",
                    flush=True,
                )
        capture.release()
    print(
        "RIGCAL_STAGE_END intrinsic_artifact_export "
        f"elapsed_seconds={time.monotonic() - export_started:.3f}",
        flush=True,
    )

    write_contact_sheet(
        debug_paths,
        out / "SELECTED_VIEWS_CONTACT_SHEET.jpg",
    )

    fields = [
        "frame_index",
        "source_name",
        "time_s",
        "sharpness",
        "center_x",
        "center_y",
        "area_fraction",
        "angle_deg",
        "top_bottom_log_ratio",
        "left_right_log_ratio",
        "selected_for_calibration",
        "view_reprojection_rmse_px",
    ]

    with (
        out / "checkerboard_detections.csv"
    ).open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
        )

        writer.writeheader()

        final_set = set(final_indices)

        for detection_index, item in enumerate(detections):
            row = {
                field: item.get(field, "")
                for field in fields
            }

            row["selected_for_calibration"] = (
                detection_index in final_set
            )

            row["view_reprojection_rmse_px"] = (
                per_view_error.get(
                    detection_index,
                    "",
                )
            )

            writer.writerow(row)

    camera_info = {
        "camera_name": "moving_calib_camera",
        "width": width,
        "height": height,
        "image_width": width,
        "image_height": height,
        "distortion_model": distortion_model,
        "K": K.reshape(-1).tolist(),
        "k": K.reshape(-1).tolist(),
        "D": D.tolist(),
        "d": D.tolist(),
        "R": np.eye(3).reshape(-1).tolist(),
        "r": np.eye(3).reshape(-1).tolist(),
        "P": [
            float(K[0, 0]), 0.0, float(K[0, 2]), 0.0,
            0.0, float(K[1, 1]), float(K[1, 2]), 0.0,
            0.0, 0.0, 1.0, 0.0,
        ],
        "p": [
            float(K[0, 0]), 0.0, float(K[0, 2]), 0.0,
            0.0, float(K[1, 1]), float(K[1, 2]), 0.0,
            0.0, 0.0, 1.0, 0.0,
        ],
        "fx": float(K[0, 0]),
        "fy": float(K[1, 1]),
        "cx": float(K[0, 2]),
        "cy": float(K[1, 2]),
        "source_type": "video" if video is not None else "image_directory",
        "intrinsic_scan": {
            "mode": args.scan_mode,
            "target_hz": args.scan_target_hz,
            "preview_max_dimension": args.preview_max_dimension,
            "tested_frame_count": scanned_frames,
            "tested_frame_indices": sorted(tested_indices),
        },
        "checkerboard_inner_corners": {
            "columns": args.cols,
            "rows": args.rows,
        },
        "scanned_original_frames": scanned_frames,
        "successful_checkerboard_detections": len(detections),
        "selected_calibration_views": len(final_indices),
        "removed_outlier_views": removed,
        "opencv_calibration_rms_px": result["rms"],
        "median_view_reprojection_rmse_px": (
            result["median_view_error"]
        ),
        "maximum_view_reprojection_rmse_px": (
            result["maximum_view_error"]
        ),
        "model_comparison": comparison,
    }
    if video is not None:
        assert video_geometry is not None
        camera_info.update(
            {
                "source_video": str(video),
                "source_fps": source_fps,
                "video_geometry": video_geometry.as_dict(),
            }
        )
    else:
        camera_info.update(
            {
                "source_images": str(image_root),
                "source_image_count": len(image_paths),
            }
        )

    json_path = out / "moving_calib_camera.json"

    json_path.write_text(
        json.dumps(camera_info, indent=2) + "\n"
    )

    report = [
        "MOVING-CAMERA INTRINSIC CALIBRATION",
        "=" * 76,
        "",
        f"Source: {source}",
        f"Source type: {'video' if video is not None else 'checkerboard image folder'}",
        f"Resolution: {width}x{height}",
        *(
            [
                (
                    "Encoded resolution: "
                    f"{video_geometry.encoded_width}x"
                    f"{video_geometry.encoded_height}"
                ),
                (
                    "Display rotation: "
                    f"{video_geometry.display_rotation_degrees:+d} deg"
                ),
                f"Orientation policy: {video_geometry.orientation_policy}",
            ]
            if video_geometry is not None
            else []
        ),
        *(
            [f"Source FPS: {source_fps:.6f}"]
            if video is not None
            else []
        ),
        f"Source frames: {reported_frames}",
        f"Checkerboard frames tested: {scanned_frames}",
        f"Scan mode: {args.scan_mode}",
        f"Checkerboard: {args.cols}x{args.rows} inner corners",
        f"Successful detections: {len(detections)}",
        f"Initially selected views: {len(selected)}",
        f"Final calibration views: {len(final_indices)}",
        f"Removed outliers: {len(removed)}",
        "",
        "Model comparison:",
        (
            "Standard holdout median RMSE: "
            f"{comparison['standard_holdout_median_rmse_px']:.6f} px"
        ),
        (
            "Rational holdout median RMSE: "
            f"{comparison['rational_holdout_median_rmse_px']:.6f} px"
        ),
        f"Selected model: {distortion_model}",
        "",
        f"OpenCV calibration RMS: {result['rms']:.6f} px",
        (
            "Median per-view RMSE: "
            f"{result['median_view_error']:.6f} px"
        ),
        (
            "Maximum per-view RMSE: "
            f"{result['maximum_view_error']:.6f} px"
        ),
        "",
        "Camera matrix K:",
        np.array2string(K, precision=10),
        "",
        "Distortion D:",
        np.array2string(D, precision=12),
        "",
        f"CameraInfo JSON: {json_path}",
        (
            "Contact sheet: "
            f"{out / 'SELECTED_VIEWS_CONTACT_SHEET.jpg'}"
        ),
    ]

    report_path = out / "INTRINSICS_REPORT.txt"

    report_path.write_text(
        "\n".join(report) + "\n"
    )

    print()
    print(report_path.read_text())
    print("[OK] intrinsic calibration completed")


if __name__ == "__main__":
    main()
