from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from camera_rig_calibration.pipeline import StageResult, run_stage


POLICY_ID = "moving_pinhole_intrinsics_only_v1"


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


def _zero_distortion(data: dict[str, Any]) -> dict[str, Any]:
    """Return a camera-info copy that serializes as a COLMAP PINHOLE model.

    Focal lengths and principal point stay untouched. Only distortion metadata
    is removed in the diagnostic copy consumed by AP03 COLMAP.
    """
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


def prepare_pinhole_camera_info(
    *,
    shared_raw: Path,
    destination: Path,
    moving_camera_id: str,
) -> Path:
    """Copy camera-info metadata and pinhole only the moving-camera copy."""
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
        "intrinsic_terms_preserved": ["fx", "fy", "cx", "cy"],
        "distortion_used_by_colmap": False,
        "original_distortion": _distortion_snapshot(original),
        "diagnostic_camera_info": str(moving_path),
        "original_camera_info": str(
            source_camera_info / f"{moving_camera_id}.json"
        ),
        "original_files_modified": False,
        "ground_truth_used": False,
    }
    (destination / "AP03_CAMERA_MODEL_SENSITIVITY.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
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
    colmap_dataset = output_root / "colmap" / "dataset"
    diagnostic_shared_raw = output_root / "colmap" / "camera_model_input"

    def action() -> dict[str, Path]:
        prepared_shared_raw = prepare_pinhole_camera_info(
            shared_raw=dataset / "raw_images",
            destination=diagnostic_shared_raw,
            moving_camera_id=moving_camera_id,
        )
        command = [
            sys.executable,
            "-m",
            "camera_rig_calibration.methods.ap03.reconstruct",
            "--dataset-root",
            str(colmap_dataset),
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

        policy_source = (
            diagnostic_shared_raw / "AP03_CAMERA_MODEL_SENSITIVITY.json"
        )
        policy_destination = stage_root / "AP03_CAMERA_MODEL_SENSITIVITY.json"
        if policy_source.is_file():
            shutil.copy2(policy_source, policy_destination)
        return {
            "database": stage_root / "database.db",
            "sparse_text": stage_root / "sparse_txt",
            "camera_model_policy": policy_destination,
        }

    return run_stage(
        "ap03.reconstruct",
        stage_root,
        action,
        inputs={"colmap_dataset": colmap_dataset},
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
