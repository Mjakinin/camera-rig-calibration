#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from pathlib import Path

import cv2
import numpy as np

def open_video_without_autorotation(path):
    capture = cv2.VideoCapture(str(path))

    if hasattr(cv2, "CAP_PROP_ORIENTATION_AUTO"):
        capture.set(
            cv2.CAP_PROP_ORIENTATION_AUTO,
            0,
        )

    return capture



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

        thumbnail = cv2.resize(
            image,
            (480, 270),
            interpolation=cv2.INTER_AREA,
        )

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

    parser.add_argument("--video", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--cols", type=int, default=8)
    parser.add_argument("--rows", type=int, default=6)
    parser.add_argument("--max-views", type=int, default=80)
    parser.add_argument("--minimum-frame-gap", type=int, default=5)

    args = parser.parse_args()

    video = Path(args.video).resolve()
    out = Path(args.out)

    if not video.is_file():
        raise RuntimeError(
            f"Video not found: {video}"
        )

    shutil.rmtree(out, ignore_errors=True)

    selected_dir = out / "selected_frames"
    debug_dir = out / "debug_selected"
    undistorted_dir = out / "undistorted_samples"

    selected_dir.mkdir(parents=True)
    debug_dir.mkdir(parents=True)
    undistorted_dir.mkdir(parents=True)

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

    valid_4k_orientations = {
        (3840, 2160),
        (2160, 3840),
    }

    if (width, height) not in valid_4k_orientations:
        raise RuntimeError(
            "Expected native 4K video in landscape or portrait "
            f"orientation, got {width}x{height}"
        )

    pattern = (args.cols, args.rows)

    detector_flags = (
        cv2.CALIB_CB_NORMALIZE_IMAGE
        | cv2.CALIB_CB_EXHAUSTIVE
        | cv2.CALIB_CB_ACCURACY
    )

    detections = []
    scanned_frames = 0

    print("[INFO] source:", video)
    print("[INFO] resolution:", width, "x", height)
    print("[INFO] source FPS:", source_fps)
    print("[INFO] reported frames:", reported_frames)
    print("[INFO] checkerboard:", pattern)
    print("[INFO] scanning every original frame")

    while True:
        ok, frame = capture.read()

        if not ok:
            break

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY,
        )

        found, corners = cv2.findChessboardCornersSB(
            gray,
            pattern,
            flags=detector_flags,
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
                    "time_s": (
                        scanned_frames
                        / max(source_fps, 1e-9)
                    ),
                    "corners": corners.reshape(-1, 1, 2),
                    **metrics,
                }
            )

        if scanned_frames % 100 == 0:
            print(
                "[SCAN]",
                f"frame={scanned_frames}",
                f"detections={len(detections)}",
            )

        scanned_frames += 1

    capture.release()

    if len(detections) < 20:
        raise RuntimeError(
            "Insufficient checkerboard detections: "
            f"{len(detections)}"
        )

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

    capture = open_video_without_autorotation(video)
    frame_index = 0
    debug_paths = []

    sample_frames = set(
        detections[index]["frame_index"]
        for index in (
            final_indices[::max(1, len(final_indices) // 5)]
        )[:5]
    )

    while True:
        ok, frame = capture.read()

        if not ok:
            break

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

        frame_index += 1

    capture.release()

    write_contact_sheet(
        debug_paths,
        out / "SELECTED_VIEWS_CONTACT_SHEET.jpg",
    )

    fields = [
        "frame_index",
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
        "source_video": str(video),
        "source_fps": source_fps,
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

    json_path = out / "moving_calib_camera.json"

    json_path.write_text(
        json.dumps(camera_info, indent=2) + "\n"
    )

    report = [
        "0.5x MOVING-CAMERA INTRINSIC CALIBRATION",
        "=" * 76,
        "",
        f"Source: {video}",
        f"Resolution: {width}x{height}",
        f"Source FPS: {source_fps:.6f}",
        f"Original frames scanned: {scanned_frames}",
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
