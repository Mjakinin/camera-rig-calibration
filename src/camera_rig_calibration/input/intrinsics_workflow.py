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



from .intrinsics_detection import (
    _detect_candidates,
    balanced_candidate_indices,
    board_metrics,
    detect_checkerboard_balanced,
    open_video_without_autorotation,
    select_diverse,
)
from .intrinsics_reporting import write_contact_sheet
from .intrinsics_solver import (
    calibrate_with_outlier_filter,
    model_comparison,
    object_points,
)
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
