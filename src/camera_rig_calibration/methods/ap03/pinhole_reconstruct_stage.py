from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from camera_rig_calibration.pipeline import StageResult, run_stage


POLICY_ID = "moving_undistorted_pinhole_v1"


def _distortion_snapshot(data: dict[str, Any]) -> dict[str, Any]:
    snapshot: dict[str, Any] = {}
    for key in (
        "distortion_model",
        "D",
        "d",
        "distortion",
        "distortion_coefficients",
    ):
        if key in data:
            snapshot[key] = data[key]
    return snapshot


def _camera_matrix_and_distortion(data: dict[str, Any]):
    import numpy as np

    flat_k = data.get("K", data.get("k"))
    if flat_k is None and "camera_matrix" in data:
        value = data["camera_matrix"]
        flat_k = value.get("data") if isinstance(value, dict) else value
    if flat_k is None:
        fx = float(data["fx"])
        fy = float(data.get("fy", fx))
        cx = float(data["cx"])
        cy = float(data["cy"])
        flat_k = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
    matrix = np.asarray(flat_k, dtype=float).reshape(3, 3)

    distortion = data.get("D", data.get("d"))
    if distortion is None and "distortion_coefficients" in data:
        value = data["distortion_coefficients"]
        distortion = value.get("data") if isinstance(value, dict) else value
    if distortion is None:
        distortion = data.get("distortion", [])
    coefficients = np.asarray(list(distortion), dtype=float).reshape(-1)
    return matrix, coefficients


def _zero_distortion(data: dict[str, Any]) -> dict[str, Any]:
    """Return camera-info for images already rectified to the same K matrix."""
    result = json.loads(json.dumps(data))
    result["distortion_model"] = "none"
    result["D"] = []
    if "d" in result:
        result["d"] = []
    if "distortion" in result:
        result["distortion"] = []
    if "distortion_coefficients" in result:
        value = result["distortion_coefficients"]
        if isinstance(value, dict):
            value = dict(value)
            value["data"] = []
            result["distortion_coefficients"] = value
        else:
            result["distortion_coefficients"] = []
    return result


def _link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        destination.unlink()
    try:
        os.link(source, destination)
    except OSError:
        try:
            destination.symlink_to(source.resolve())
        except OSError:
            shutil.copy2(source, destination)


def prepare_pinhole_camera_info(
    *,
    shared_raw: Path,
    destination: Path,
    moving_camera_id: str,
) -> Path:
    """Copy camera-info and make only the rectified moving-camera copy PINHOLE."""
    source_camera_info = shared_raw / "camera_info"
    if not source_camera_info.is_dir():
        raise RuntimeError(
            f"AP03 sensitivity input has no camera_info directory: {source_camera_info}"
        )

    shutil.rmtree(destination, ignore_errors=True)
    destination_camera_info = destination / "camera_info"
    destination_camera_info.mkdir(parents=True, exist_ok=True)
    for source in sorted(source_camera_info.glob("*.json")):
        shutil.copy2(source, destination_camera_info / source.name)

    moving_path = destination_camera_info / f"{moving_camera_id}.json"
    source_moving_path = source_camera_info / f"{moving_camera_id}.json"
    if not moving_path.is_file():
        raise RuntimeError(
            "AP03 sensitivity could not find moving-camera intrinsics: "
            f"{moving_path}"
        )
    original = json.loads(moving_path.read_text(encoding="utf-8"))
    modified = _zero_distortion(original)
    moving_path.write_text(
        json.dumps(modified, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    metadata = {
        "schema_version": 5,
        "policy": POLICY_ID,
        "scope": "ap03_colmap_moving_camera_only",
        "moving_camera_id": moving_camera_id,
        "image_preprocessing": "opencv_undistort_same_intrinsic_matrix",
        "intrinsic_terms_preserved": ["fx", "fy", "cx", "cy"],
        "distortion_used_for_preprocessing": True,
        "distortion_used_by_colmap": False,
        "original_distortion": _distortion_snapshot(original),
        "diagnostic_camera_info": str(moving_path),
        "original_camera_info": str(source_moving_path),
        "original_files_modified": False,
        "ground_truth_used": False,
    }
    (destination / "AP03_CAMERA_MODEL_SENSITIVITY.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return destination


def prepare_undistorted_colmap_dataset(
    *,
    source_dataset: Path,
    shared_raw: Path,
    destination: Path,
    moving_camera_id: str,
) -> Path:
    """Rectify only AP03 moving images; link static images unchanged."""
    import cv2

    manifest = source_dataset / "image_manifest.csv"
    source_images = source_dataset / "images"
    if not manifest.is_file() or not source_images.is_dir():
        raise RuntimeError(
            f"AP03 prepared COLMAP dataset is incomplete: {source_dataset}"
        )

    moving_info_path = shared_raw / "camera_info" / f"{moving_camera_id}.json"
    if not moving_info_path.is_file():
        raise RuntimeError(
            f"Missing moving-camera intrinsics for undistortion: {moving_info_path}"
        )
    moving_info = json.loads(moving_info_path.read_text(encoding="utf-8"))
    matrix, distortion = _camera_matrix_and_distortion(moving_info)

    shutil.rmtree(destination, ignore_errors=True)
    destination_images = destination / "images"
    destination_images.mkdir(parents=True, exist_ok=True)

    with manifest.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"Empty AP03 image manifest: {manifest}")

    moving_count = 0
    static_count = 0
    for row in rows:
        image_name = str(row["image_name"])
        source = source_images / image_name
        destination_image = destination_images / image_name
        if not source.is_file():
            raise RuntimeError(f"Missing AP03 source image: {source}")
        if str(row.get("source_type", "")) != "moving":
            _link_or_copy(source, destination_image)
            static_count += 1
            continue

        image = cv2.imread(str(source), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise RuntimeError(f"Could not read AP03 moving image: {source}")
        if distortion.size and float(abs(distortion).max()) > 1e-15:
            image = cv2.undistort(image, matrix, distortion, None, matrix)
        if not cv2.imwrite(str(destination_image), image):
            raise RuntimeError(
                f"Could not write undistorted AP03 moving image: {destination_image}"
            )
        moving_count += 1

    shutil.copy2(manifest, destination / "image_manifest.csv")
    (destination / "AP03_UNDISTORTION_SUMMARY.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "policy": POLICY_ID,
                "moving_images_undistorted": moving_count,
                "static_images_linked_unchanged": static_count,
                "new_camera_matrix_policy": "preserve_original_fx_fy_cx_cy",
                "original_files_modified": False,
                "ground_truth_used": False,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return destination


def run(
    *,
    dataset: Path,
    output_root: Path,
    camera_ids: tuple[str, ...],
    moving_camera_id: str,
    matcher: str,
    use_gpu: bool,
    maximum_image_size: int | None,
    maximum_features: int | None,
    sequential_overlap: int,
    loop_detection: bool | None,
    mapper_minimum_matches: int,
    colmap_executable: str,
    reuse: bool,
) -> StageResult:
    stage_root = output_root / "colmap" / "reconstruction"
    prepared_colmap_dataset = output_root / "colmap" / "dataset"
    diagnostic_colmap_dataset = output_root / "colmap" / "undistorted_pinhole_dataset"
    diagnostic_shared_raw = output_root / "colmap" / "undistorted_pinhole_camera_info"

    def action() -> dict[str, Path]:
        prepared_shared_raw = prepare_pinhole_camera_info(
            shared_raw=dataset / "raw_images",
            destination=diagnostic_shared_raw,
            moving_camera_id=moving_camera_id,
        )
        prepared_dataset = prepare_undistorted_colmap_dataset(
            source_dataset=prepared_colmap_dataset,
            shared_raw=dataset / "raw_images",
            destination=diagnostic_colmap_dataset,
            moving_camera_id=moving_camera_id,
        )
        command = [
            sys.executable,
            "-m",
            "camera_rig_calibration.methods.ap03.reconstruct",
            "--dataset-root",
            str(prepared_dataset),
            "--shared-raw",
            str(prepared_shared_raw),
            "--run-root",
            str(stage_root),
            "--cameras",
            ",".join(camera_ids),
            "--moving-camera",
            moving_camera_id,
            "--colmap",
            colmap_executable,
            "--use-gpu",
            "1" if use_gpu else "0",
            "--matcher",
            matcher,
            "--sequential-overlap",
            str(sequential_overlap),
            "--mapper-min-matches",
            str(mapper_minimum_matches),
        ]
        if maximum_image_size is not None:
            command.extend(["--max-image-size", str(maximum_image_size)])
        if maximum_features is not None:
            command.extend(["--max-features", str(maximum_features)])
        if loop_detection is not None:
            command.extend(
                ["--loop-detection", "1" if loop_detection else "0"]
            )
        if reuse:
            command.append("--reuse")
        subprocess.run(command, check=True)

        policy_source = diagnostic_shared_raw / "AP03_CAMERA_MODEL_SENSITIVITY.json"
        policy_destination = stage_root / "AP03_CAMERA_MODEL_SENSITIVITY.json"
        if policy_source.is_file():
            shutil.copy2(policy_source, policy_destination)
        return {
            "database": stage_root / "database.db",
            "sparse_text": stage_root / "sparse_txt",
            "camera_model_policy": policy_destination,
            "undistorted_dataset": diagnostic_colmap_dataset,
        }

    return run_stage(
        "ap03.reconstruct",
        stage_root,
        action,
        inputs={
            "colmap_dataset": prepared_colmap_dataset,
            "moving_camera_info": (
                dataset / "raw_images" / "camera_info" / f"{moving_camera_id}.json"
            ),
        },
        parameters={
            "matcher": matcher,
            "gpu": use_gpu,
            "maximum_image_size": maximum_image_size,
            "maximum_features": maximum_features,
            "sequential_overlap": sequential_overlap,
            "loop_detection": loop_detection,
            "mapper_minimum_matches": mapper_minimum_matches,
            "camera_model_policy": POLICY_ID,
            "camera_model_scope": "moving_camera_only",
            "moving_image_preprocessing": "undistort_same_K",
            "original_camera_info_modified": False,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cameras", required=True)
    parser.add_argument("--moving-camera-id", required=True)
    parser.add_argument(
        "--matcher", choices=["exhaustive", "sequential"], required=True
    )
    parser.add_argument("--use-gpu", type=int, choices=[0, 1], required=True)
    parser.add_argument("--max-image-size", type=int)
    parser.add_argument("--max-features", type=int)
    parser.add_argument("--sequential-overlap", type=int, required=True)
    parser.add_argument("--loop-detection", type=int, choices=[0, 1])
    parser.add_argument("--mapper-min-matches", type=int, required=True)
    parser.add_argument("--colmap-executable", required=True)
    parser.add_argument("--reuse-colmap", action="store_true")
    args = parser.parse_args()
    run(
        dataset=args.dataset.resolve(),
        output_root=args.out.resolve(),
        camera_ids=tuple(
            item.strip() for item in args.cameras.split(",") if item.strip()
        ),
        moving_camera_id=args.moving_camera_id,
        matcher=args.matcher,
        use_gpu=bool(args.use_gpu),
        maximum_image_size=args.max_image_size,
        maximum_features=args.max_features,
        sequential_overlap=args.sequential_overlap,
        loop_detection=(
            bool(args.loop_detection)
            if args.loop_detection is not None
            else None
        ),
        mapper_minimum_matches=args.mapper_min_matches,
        colmap_executable=args.colmap_executable,
        reuse=args.reuse_colmap,
    )


if __name__ == "__main__":
    main()
