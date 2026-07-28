#!/usr/bin/env python3
"""Normalize a moving-camera video into timestamp-selected dataset frames."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
from pathlib import Path

import cv2

try:
    from camera_rig_calibration.input.video_geometry import open_oriented_video
except ImportError:  # pragma: no cover - direct script fallback
    from video_geometry import open_oriented_video


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Extract a moving-camera video at an exact target cadence without resizing. "
            "Selection uses source-frame timestamps derived from the reported source FPS."
        )
    )
    parser.add_argument("--video", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--target-fps", type=float, default=3.0)
    parser.add_argument("--start-s", type=float, default=0.0)
    parser.add_argument("--end-s", type=float)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--skip-sha256",
        action="store_true",
        help="Skip hashing the source video when only a fast extraction is required.",
    )
    args = parser.parse_args()

    video = Path(args.video).resolve()
    dataset = Path(args.dataset).resolve()
    if not video.is_file():
        raise RuntimeError(f"Video not found: {video}")
    if not math.isfinite(args.target_fps) or args.target_fps <= 0:
        raise RuntimeError("--target-fps must be positive")
    if args.start_s < 0:
        raise RuntimeError("--start-s must be non-negative")
    if args.end_s is not None and args.end_s <= args.start_s:
        raise RuntimeError("--end-s must be greater than --start-s")
    if args.max_frames is not None and args.max_frames <= 0:
        raise RuntimeError("--max-frames must be positive")

    output = dataset / "raw_images/moving"
    metadata = dataset / "metadata/moving_video_extraction"
    existing = sorted(output.glob("frame_*.png")) if output.is_dir() else []
    if existing and not args.overwrite:
        raise RuntimeError(
            f"{output} already contains {len(existing)} frames. "
            "Use --overwrite to replace them."
        )

    shutil.rmtree(output, ignore_errors=True)
    shutil.rmtree(metadata, ignore_errors=True)
    output.mkdir(parents=True)
    metadata.mkdir(parents=True)

    capture = open_oriented_video(video)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video}")

    geometry = capture.geometry
    source_fps = float(capture.get(cv2.CAP_PROP_FPS))
    source_frames_reported = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if not math.isfinite(source_fps) or source_fps <= 0:
        raise RuntimeError(f"Invalid source FPS reported by OpenCV: {source_fps}")
    if width <= 0 or height <= 0:
        raise RuntimeError(f"Invalid source dimensions: {width}x{height}")
    if args.target_fps > source_fps + 1e-9:
        raise RuntimeError(
            f"Target FPS {args.target_fps} exceeds source FPS {source_fps}"
        )

    interval = 1.0 / args.target_fps
    next_target_s = args.start_s
    source_index = 0
    output_index = 0
    rows: list[dict] = []
    tolerance = 0.5 / source_fps
    source_last_time_s = max(0.0, (source_frames_reported - 1) / source_fps)
    effective_end_s = min(
        source_last_time_s,
        args.end_s if args.end_s is not None else source_last_time_s,
    )
    expected_output_frames = max(
        0,
        int(
            math.floor(
                (effective_end_s - args.start_s + tolerance)
                * args.target_fps
            )
        )
        + 1,
    )
    if args.max_frames is not None:
        expected_output_frames = min(
            expected_output_frames, args.max_frames
        )
    progress_interval = max(1, expected_output_frames // 20)
    print(
        f"[INFO] video extraction target: {expected_output_frames} frames "
        f"from {effective_end_s - args.start_s:.1f} s of source video",
        flush=True,
    )

    while True:
        ok, frame = capture.read()
        if not ok:
            break

        time_s = source_index / source_fps
        if args.end_s is not None and time_s > args.end_s + tolerance:
            break

        if time_s + tolerance >= next_target_s:
            if frame.shape[1] != width or frame.shape[0] != height:
                raise RuntimeError(
                    f"Frame {source_index} changed dimensions to "
                    f"{frame.shape[1]}x{frame.shape[0]}"
                )
            destination = output / f"frame_{output_index:06d}.png"
            if not cv2.imwrite(str(destination), frame):
                raise RuntimeError(f"Could not write {destination}")
            rows.append(
                {
                    "output_frame": output_index,
                    "output_file": destination.name,
                    "source_frame": source_index,
                    "source_time_s": f"{time_s:.9f}",
                    "target_time_s": f"{next_target_s:.9f}",
                    "time_error_ms": f"{1000.0 * (time_s - next_target_s):.6f}",
                }
            )
            output_index += 1
            next_target_s = args.start_s + output_index * interval
            if (
                output_index == 1
                or output_index % progress_interval == 0
                or output_index == expected_output_frames
            ):
                print(
                    "RIGCAL_PROGRESS "
                    f"current={output_index} "
                    f"total={expected_output_frames} "
                    "unit=frames "
                    "label=video extraction | "
                    f"source {time_s:.1f}/{effective_end_s:.1f} s",
                    flush=True,
                )
            if args.max_frames is not None and output_index >= args.max_frames:
                break

        source_index += 1

    capture.release()
    if not rows:
        raise RuntimeError("No frames were extracted")

    manifest = metadata / "frame_manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    source_hash = None if args.skip_sha256 else sha256_file(video)
    payload = {
        "source_video": str(video),
        "source_video_sha256": source_hash,
        "source_fps": source_fps,
        "source_frames_reported": source_frames_reported,
        "source_width": width,
        "source_height": height,
        "encoded_width": geometry.encoded_width,
        "encoded_height": geometry.encoded_height,
        "display_rotation_degrees": geometry.display_rotation_degrees,
        "output_width": geometry.output_width,
        "output_height": geometry.output_height,
        "orientation_policy": geometry.orientation_policy,
        "video_geometry_contract": geometry.contract,
        "target_fps": args.target_fps,
        "selection_method": (
            "sequential source-frame timestamps; first frame at or after each "
            "target timestamp within half a source-frame tolerance"
        ),
        "start_s": args.start_s,
        "end_s": args.end_s,
        "output_frame_count": len(rows),
        "output_pattern": "raw_images/moving/frame_%06d.png",
        "resize_applied": False,
        "autorotation_applied": geometry.display_rotation_degrees != 0,
        "manifest": str(manifest),
    }
    json_path = metadata / "MOVING_VIDEO_EXTRACTION.json"
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    report = metadata / "MOVING_VIDEO_EXTRACTION.txt"
    report.write_text(
        "\n".join(
            [
                "REAL MOVING-CAMERA VIDEO EXTRACTION",
                "=" * 72,
                "",
                f"Source: {video}",
                f"SHA-256: {source_hash or 'SKIPPED'}",
                (
                    "Encoded source: "
                    f"{geometry.encoded_width}x{geometry.encoded_height} "
                    f"at {source_fps:.9f} FPS"
                ),
                (
                    "Display geometry: "
                    f"rotation={geometry.display_rotation_degrees:+d} deg -> "
                    f"{geometry.output_width}x{geometry.output_height}"
                ),
                f"Target FPS: {args.target_fps:.9f}",
                f"Trim: [{args.start_s}, {args.end_s if args.end_s is not None else 'EOF'}] s",
                f"Extracted frames: {len(rows)}",
                "Resize: no",
                f"Orientation policy: {geometry.orientation_policy}",
                f"Manifest: {manifest}",
                "",
                "[OK] extraction complete",
                "",
            ]
        ),
        encoding="utf-8",
    )

    print(report.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
