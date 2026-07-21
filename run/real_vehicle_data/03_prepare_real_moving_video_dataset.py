#!/usr/bin/env python3
"""Prepare one real moving-camera video as an independent 3 Hz dataset.

The static-camera images, camera models, static ArUco observations and reference
marker are copied from an existing canonical template dataset.  The supplied
moving-camera intrinsics are installed for the new dataset and every selected
moving frame is written with contiguous numbering.

Video decoding deliberately disables metadata autorotation, matching the
intrinsic calibration engine.  The raw decoded video dimensions must therefore
match the dimensions stored in the intrinsic CameraInfo JSON exactly.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import cv2


CAMERAS = ("cam_edge_0", "cam_edge_1", "cam_edge_3", "cam_edge_5")


def repository_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / ".git").exists():
            return parent
    raise RuntimeError("Repository root not found")


def open_video_without_autorotation(path: Path) -> cv2.VideoCapture:
    capture = cv2.VideoCapture(str(path))
    if hasattr(cv2, "CAP_PROP_ORIENTATION_AUTO"):
        capture.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)
    return capture


def resolve_from_repo(repo: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (repo / path).resolve()


def read_camera_info(path: Path) -> dict:
    if not path.is_file():
        raise RuntimeError(f"Missing moving-camera intrinsics: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    width = int(data.get("width", data.get("image_width", 0)) or 0)
    height = int(data.get("height", data.get("image_height", 0)) or 0)
    K = data.get("K", data.get("k"))
    D = data.get("D", data.get("d"))
    if width <= 0 or height <= 0:
        raise RuntimeError(f"Invalid intrinsic dimensions in {path}: {width}x{height}")
    if not isinstance(K, list) or len(K) != 9:
        raise RuntimeError(f"Camera matrix K must contain 9 values: {path}")
    if not isinstance(D, list):
        raise RuntimeError(f"Distortion vector D must be a list: {path}")
    return data


def copy_tree_if_present(source: Path, destination: Path) -> None:
    if source.is_dir():
        shutil.copytree(source, destination, dirs_exist_ok=True)


def rewrite_static_observation_paths(csv_path: Path, shared_root: Path) -> None:
    if not csv_path.is_file():
        raise RuntimeError(f"Missing copied static observation CSV: {csv_path}")

    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fields = list(reader.fieldnames or [])

    for row in rows:
        camera = str(row.get("camera_name", ""))
        old_path = Path(str(row.get("image_path", "")))
        filename = old_path.name
        static_multi = shared_root / "raw_images" / "static_multi" / camera / filename
        static_single = shared_root / "raw_images" / "static" / f"{camera}.png"
        if static_multi.is_file():
            row["image_path"] = str(static_multi.resolve())
        elif static_single.is_file():
            row["image_path"] = str(static_single.resolve())

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def copy_static_contract(template: Path, shared_root: Path) -> None:
    template_raw = template / "raw_images"
    template_obs = template / "aruco_observations"
    destination_raw = shared_root / "raw_images"
    destination_obs = shared_root / "aruco_observations"

    for required in (
        template_raw / "static",
        template_raw / "camera_info",
        template_obs / "shared_static_aruco_observations.csv",
        template_obs / "REFERENCE_MARKER_ID.txt",
    ):
        if not required.exists():
            raise RuntimeError(f"Template dataset is incomplete: {required}")

    copy_tree_if_present(template_raw / "static", destination_raw / "static")
    copy_tree_if_present(template_raw / "static_multi", destination_raw / "static_multi")
    copy_tree_if_present(template_raw / "camera_info", destination_raw / "camera_info")

    destination_obs.mkdir(parents=True, exist_ok=True)
    shutil.copy2(
        template_obs / "shared_static_aruco_observations.csv",
        destination_obs / "shared_static_aruco_observations.csv",
    )
    shutil.copy2(
        template_obs / "REFERENCE_MARKER_ID.txt",
        destination_obs / "REFERENCE_MARKER_ID.txt",
    )

    copy_tree_if_present(
        template / "metadata" / "static_extraction",
        shared_root / "metadata" / "static_extraction",
    )
    rewrite_static_observation_paths(
        destination_obs / "shared_static_aruco_observations.csv",
        shared_root,
    )


def extract_at_rate(
    video: Path,
    destination: Path,
    target_hz: float,
    expected_size: tuple[int, int],
) -> dict:
    capture = open_video_without_autorotation(video)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open moving video: {video}")

    source_fps = float(capture.get(cv2.CAP_PROP_FPS))
    source_frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if source_fps <= 0:
        raise RuntimeError(f"Invalid source FPS for {video}: {source_fps}")
    if (width, height) != expected_size:
        raise RuntimeError(
            "Moving-video raw decoded dimensions do not match the 1x intrinsic "
            f"calibration: video={width}x{height}, intrinsics={expected_size[0]}x{expected_size[1]}. "
            "This usually indicates an orientation/crop mismatch; do not scale K blindly."
        )

    destination.mkdir(parents=True, exist_ok=True)
    frame_map = []
    source_index = 0
    output_index = 0
    next_time = 0.0
    interval = 1.0 / target_hz
    half_source_period = 0.5 / source_fps

    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if (int(frame.shape[1]), int(frame.shape[0])) != expected_size:
            raise RuntimeError(
                f"Frame {source_index} changed dimensions to "
                f"{frame.shape[1]}x{frame.shape[0]}"
            )

        source_time = source_index / source_fps
        if source_time + half_source_period >= next_time:
            filename = f"frame_{output_index:06d}.png"
            output_path = destination / filename
            if not cv2.imwrite(str(output_path), frame):
                raise RuntimeError(f"Could not write extracted frame: {output_path}")
            frame_map.append(
                {
                    "output_frame": output_index,
                    "source_frame": source_index,
                    "source_time_s": source_time,
                    "output_path": str(output_path.resolve()),
                }
            )
            output_index += 1
            next_time += interval

        source_index += 1

    capture.release()
    if not frame_map:
        raise RuntimeError(f"No frames were extracted from {video}")

    return {
        "source_video": str(video.resolve()),
        "autorotation_disabled": True,
        "source_width": width,
        "source_height": height,
        "source_fps": source_fps,
        "reported_source_frames": source_frame_count,
        "decoded_source_frames": source_index,
        "target_sampling_hz": target_hz,
        "extracted_frames": len(frame_map),
        "first_source_time_s": frame_map[0]["source_time_s"],
        "last_source_time_s": frame_map[-1]["source_time_s"],
        "frame_map": frame_map,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare one independent real moving-camera video dataset at 3 Hz."
    )
    parser.add_argument("--video", required=True)
    parser.add_argument("--intrinsics-json", required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument(
        "--template-dataset",
        default=(
            "results/real_vehicle_data/real_05x_4k_3hz_v1/00_shared_input"
        ),
    )
    parser.add_argument("--results-root", default="results/real_vehicle_data")
    parser.add_argument("--sampling-hz", type=float, default=3.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.sampling_hz <= 0:
        raise RuntimeError("--sampling-hz must be positive")

    repo = repository_root()
    video = resolve_from_repo(repo, args.video)
    intrinsics_path = resolve_from_repo(repo, args.intrinsics_json)
    template = resolve_from_repo(repo, args.template_dataset)
    results_root = resolve_from_repo(repo, args.results_root)
    result_root = results_root / args.dataset_name
    shared_root = result_root / "00_shared_input"

    if not video.is_file():
        raise RuntimeError(f"Moving video not found: {video}")
    if not template.is_dir():
        raise RuntimeError(f"Template shared dataset not found: {template}")
    if shared_root.exists():
        if not args.overwrite:
            raise RuntimeError(
                f"Shared dataset already exists: {shared_root}; pass --overwrite to replace it"
            )
        shutil.rmtree(shared_root)

    shared_root.mkdir(parents=True)
    copy_static_contract(template, shared_root)

    intrinsics = read_camera_info(intrinsics_path)
    moving_info_destination = (
        shared_root / "raw_images" / "camera_info" / "moving_calib_camera.json"
    )
    moving_info_destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(intrinsics_path, moving_info_destination)

    expected_size = (
        int(intrinsics.get("width", intrinsics.get("image_width"))),
        int(intrinsics.get("height", intrinsics.get("image_height"))),
    )
    moving_root = shared_root / "raw_images" / "moving"
    metadata = extract_at_rate(
        video,
        moving_root,
        target_hz=float(args.sampling_hz),
        expected_size=expected_size,
    )
    metadata.update(
        {
            "dataset_name": args.dataset_name,
            "template_dataset": str(template),
            "moving_intrinsics_json": str(moving_info_destination.resolve()),
            "intrinsic_source_json": str(intrinsics_path),
        }
    )

    metadata_root = shared_root / "metadata" / "moving_video_extraction"
    metadata_root.mkdir(parents=True, exist_ok=True)
    metadata_json = metadata_root / "MOVING_VIDEO_EXTRACTION.json"
    metadata_json.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    frame_map_csv = metadata_root / "MOVING_VIDEO_FRAME_MAP.csv"
    with frame_map_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("output_frame", "source_frame", "source_time_s", "output_path"),
        )
        writer.writeheader()
        writer.writerows(metadata["frame_map"])

    print("[OK] independent real-video dataset prepared")
    print(" result root:", result_root)
    print(" shared input:", shared_root)
    print(" source video:", video)
    print(" source raw dimensions:", f"{metadata['source_width']}x{metadata['source_height']}")
    print(" source FPS:", metadata["source_fps"])
    print(" target sampling:", f"{args.sampling_hz} Hz")
    print(" extracted moving frames:", metadata["extracted_frames"])
    print(" metadata:", metadata_json)


if __name__ == "__main__":
    main()
