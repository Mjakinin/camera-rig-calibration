#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path

import cv2

CAMERAS = ("cam_edge_0", "cam_edge_1", "cam_edge_3", "cam_edge_5")
EXPECTED_MAPPING = {
    "cam_edge_0": ("center_left", "/edge_5/"),
    "cam_edge_1": ("front_right", "/edge_1/"),
    "cam_edge_3": ("front_left", "/edge_3/"),
    "cam_edge_5": ("back_right", "/edge_0/"),
}
OBSERVATION_FILES = (
    "shared_static_aruco_observations.csv",
    "shared_moving_aruco_observations.csv",
    "shared_all_aruco_observations.csv",
)


def repository_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / ".git").exists():
            return parent
    raise RuntimeError("Repository root not found")


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"Invalid JSON {path}: {exc}") from exc


def camera_parameters(path: Path) -> dict:
    data = read_json(path)
    flat = data.get("K", data.get("k"))
    if not isinstance(flat, list) or len(flat) != 9:
        raise RuntimeError(f"Camera matrix K must contain 9 values: {path}")
    values = [float(value) for value in flat]
    if not all(math.isfinite(value) for value in values):
        raise RuntimeError(f"Non-finite camera matrix in {path}")
    if values[0] <= 0 or values[4] <= 0:
        raise RuntimeError(f"Non-positive focal length in {path}")

    distortion = data.get("D", data.get("d", []))
    if not isinstance(distortion, list):
        raise RuntimeError(f"Distortion coefficients must be a list: {path}")
    distortion = [float(value) for value in distortion]
    if not all(math.isfinite(value) for value in distortion):
        raise RuntimeError(f"Non-finite distortion in {path}")

    width = int(data.get("width", data.get("image_width", 0)) or 0)
    height = int(data.get("height", data.get("image_height", 0)) or 0)
    if width <= 0 or height <= 0:
        raise RuntimeError(f"Invalid image dimensions in {path}")

    return {
        "data": data,
        "width": width,
        "height": height,
        "fx": values[0],
        "fy": values[4],
        "cx": values[2],
        "cy": values[5],
        "D": distortion,
        "distortion_model": str(data.get("distortion_model", "")),
    }


def image_size(path: Path) -> tuple[int, int]:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError(f"Could not decode image: {path}")
    return int(image.shape[1]), int(image.shape[0])


def assert_close(actual: float, expected: float, label: str, tol: float = 1e-6) -> None:
    if not math.isclose(actual, expected, rel_tol=tol, abs_tol=tol):
        raise RuntimeError(f"{label}: {actual} != {expected}")


def frame_number(path: Path) -> int:
    match = re.fullmatch(r"frame_(\d+)", path.stem)
    if match is None:
        raise RuntimeError(f"Unexpected moving-frame name: {path.name}")
    return int(match.group(1))


def validate_observations(obs_root: Path, infos: dict[str, dict]) -> dict:
    all_path = obs_root / "shared_all_aruco_observations.csv"
    with all_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"No observations in {all_path}")

    marker_ids: set[int] = set()
    static_observers: set[str] = set()
    moving_observers: set[str] = set()
    checked = 0

    for row_number, row in enumerate(rows, start=2):
        success = str(row.get("pnp_success", "")).strip().lower() in {"true", "1", "yes"}
        if not success:
            continue

        camera_name = str(row.get("camera_name", "")).strip()
        if camera_name not in infos:
            raise RuntimeError(
                f"{all_path}:{row_number}: unknown camera_name={camera_name!r}"
            )

        info = infos[camera_name]
        for key in ("fx", "fy", "cx", "cy"):
            assert_close(
                float(row[key]),
                float(info[key]),
                f"{all_path}:{row_number}:{camera_name}:{key}",
            )

        row_model = str(row.get("distortion_model", "")).strip().lower()
        expected_model = str(info["distortion_model"]).strip().lower()
        if row_model != expected_model:
            raise RuntimeError(
                f"{all_path}:{row_number}: distortion model {row_model!r} "
                f"does not match {expected_model!r} for {camera_name}"
            )

        expected_d = list(info["D"]) + [0.0] * 8
        for index in range(8):
            value = float(row.get(f"d{index}", 0.0) or 0.0)
            assert_close(
                value,
                float(expected_d[index]),
                f"{all_path}:{row_number}:{camera_name}:d{index}",
            )

        marker_length = float(row.get("marker_length_m", 0.0) or 0.0)
        assert_close(marker_length, 0.17, f"{all_path}:{row_number}:marker_length_m")

        marker_ids.add(int(float(row["marker_id"])))
        observer_type = row.get("observer_type")
        observer_id = str(row.get("observer_id", ""))
        if observer_type == "static":
            static_observers.add(observer_id)
        elif observer_type == "moving":
            moving_observers.add(observer_id)
        checked += 1

    if static_observers != set(CAMERAS):
        raise RuntimeError(
            f"Static observers mismatch: got {sorted(static_observers)}, "
            f"expected {sorted(CAMERAS)}"
        )
    if not moving_observers:
        raise RuntimeError("No successful moving-camera observations")

    ref_path = obs_root / "REFERENCE_MARKER_ID.txt"
    if not ref_path.is_file():
        raise RuntimeError(f"Missing reference marker file: {ref_path}")
    ref_marker = int(ref_path.read_text(encoding="utf-8").strip())
    if ref_marker not in marker_ids:
        raise RuntimeError(
            f"Reference marker {ref_marker} is absent from successful observations"
        )

    return {
        "rows": len(rows),
        "successful_rows_checked": checked,
        "marker_ids": sorted(marker_ids),
        "reference_marker": ref_marker,
        "static_observers": sorted(static_observers),
        "moving_observer_count": len(moving_observers),
    }


def validate_extraction_metadata(shared: Path) -> None:
    path = shared / "metadata/static_extraction/selected_static_frames.csv"
    if not path.is_file():
        print("[WARN] static extraction metadata is missing:", path)
        return

    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    mapping = {
        row.get("canonical_camera", ""): (
            row.get("physical_camera", ""),
            row.get("source_topic", ""),
        )
        for row in rows
    }
    for camera, (physical, prefix) in EXPECTED_MAPPING.items():
        actual = mapping.get(camera)
        if actual is None:
            raise RuntimeError(f"Extraction metadata missing {camera}")
        if actual[0] != physical or prefix not in actual[1]:
            raise RuntimeError(
                f"Extraction metadata mismatch for {camera}: {actual}; "
                f"expected physical={physical}, topic containing {prefix}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate the canonical real 0.5x shared input and mappings."
    )
    parser.add_argument("--dataset")
    parser.add_argument(
        "--deep-images",
        action="store_true",
        help="Decode every moving image instead of only first/last.",
    )
    args = parser.parse_args()

    root = repository_root()
    shared = (
        Path(args.dataset).resolve()
        if args.dataset
        else root
        / "results/real_vehicle_data/real_05x_4k_3hz_v1/00_shared_input"
    )
    raw = shared / "raw_images"
    info_root = raw / "camera_info"
    obs_root = shared / "aruco_observations"

    required = [
        raw / "moving",
        raw / "static",
        info_root,
        obs_root,
        info_root / "moving_calib_camera.json",
        *(info_root / f"{camera}.json" for camera in CAMERAS),
        *(raw / "static" / f"{camera}.png" for camera in CAMERAS),
        *(obs_root / name for name in OBSERVATION_FILES),
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit(
            "[ERROR] incomplete canonical real shared input:\n- "
            + "\n- ".join(missing)
        )

    infos: dict[str, dict] = {}
    for camera in (*CAMERAS, "moving_calib_camera"):
        path = info_root / f"{camera}.json"
        infos[camera] = camera_parameters(path)

    for camera, (physical, topic_prefix) in EXPECTED_MAPPING.items():
        data = infos[camera]["data"]
        if data.get("canonical_physical_camera") != physical:
            raise RuntimeError(
                f"{camera}: canonical_physical_camera mismatch: "
                f"{data.get('canonical_physical_camera')!r} != {physical!r}"
            )
        if data.get("current_rosbag_topic_prefix") != topic_prefix:
            raise RuntimeError(
                f"{camera}: current_rosbag_topic_prefix mismatch: "
                f"{data.get('current_rosbag_topic_prefix')!r} != {topic_prefix!r}"
            )
        width, height = image_size(raw / "static" / f"{camera}.png")
        if (width, height) != (infos[camera]["width"], infos[camera]["height"]):
            raise RuntimeError(
                f"{camera}: image size {(width, height)} does not match "
                f"intrinsics {(infos[camera]['width'], infos[camera]['height'])}"
            )

    moving_files = sorted((raw / "moving").glob("frame_*.png"))
    if not moving_files:
        raise RuntimeError("No moving frames")
    numbers = [frame_number(path) for path in moving_files]
    expected = list(range(numbers[0], numbers[0] + len(numbers)))
    if numbers != expected or numbers[0] != 0:
        raise RuntimeError(
            "Moving frames must be contiguous and start at frame_000000.png; "
            f"got first={numbers[0]}, last={numbers[-1]}, count={len(numbers)}"
        )

    moving_to_check = moving_files if args.deep_images else [moving_files[0], moving_files[-1]]
    expected_size = (
        infos["moving_calib_camera"]["width"],
        infos["moving_calib_camera"]["height"],
    )
    for path in moving_to_check:
        if image_size(path) != expected_size:
            raise RuntimeError(
                f"{path}: image dimensions do not match moving-camera intrinsics "
                f"{expected_size}"
            )

    validate_extraction_metadata(shared)
    observation_summary = validate_observations(obs_root, infos)

    provenance = shared / "metadata/moving_video_extraction/MOVING_VIDEO_EXTRACTION.json"
    if provenance.is_file():
        provenance_state = f"tracked: {provenance}"
    else:
        provenance_state = (
            "missing for the existing 232-frame dataset; reruns from tracked images are "
            "reproducible, but the exact original video-to-3Hz trim is not yet proven"
        )

    print("[OK] canonical real shared input validated")
    print(" root:", shared)
    print(" static cameras:", ", ".join(CAMERAS))
    print(" moving frames:", len(moving_files))
    print(" moving dimensions:", f"{expected_size[0]}x{expected_size[1]}")
    print(" marker IDs observed:", observation_summary["marker_ids"])
    print(" reference marker:", observation_summary["reference_marker"])
    print(" successful observations checked:", observation_summary["successful_rows_checked"])
    print(" extraction provenance:", provenance_state)


if __name__ == "__main__":
    main()
