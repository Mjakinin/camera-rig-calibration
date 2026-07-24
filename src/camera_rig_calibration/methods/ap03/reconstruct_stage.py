from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from camera_rig_calibration.pipeline import StageResult, run_stage


def run(
    *,
    dataset: Path,
    output_root: Path,
    camera_ids: tuple[str, ...],
    moving_camera_id: str,
    matcher: str,
    use_gpu: bool,
    maximum_image_size: int,
    maximum_features: int,
    sequential_overlap: int,
    loop_detection: bool,
    mapper_minimum_matches: int,
    colmap_executable: str,
    reuse: bool,
) -> StageResult:
    stage_root = output_root / "colmap" / "reconstruction"
    colmap_dataset = output_root / "colmap" / "dataset"

    def action() -> dict[str, Path]:
        command = [
            sys.executable,
            "-m",
            "camera_rig_calibration.methods.ap03.reconstruct",
            "--dataset-root",
            str(colmap_dataset),
            "--shared-raw",
            str(dataset / "raw_images"),
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
            "--max-image-size",
            str(maximum_image_size),
            "--max-features",
            str(maximum_features),
            "--sequential-overlap",
            str(sequential_overlap),
            "--loop-detection",
            "1" if loop_detection else "0",
            "--mapper-min-matches",
            str(mapper_minimum_matches),
        ]
        if reuse:
            command.append("--reuse")
        subprocess.run(command, check=True)
        return {
            "database": stage_root / "database.db",
            "sparse_text": stage_root / "sparse_txt",
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
    parser.add_argument("--max-image-size", type=int, required=True)
    parser.add_argument("--max-features", type=int, required=True)
    parser.add_argument("--sequential-overlap", type=int, required=True)
    parser.add_argument("--loop-detection", type=int, choices=[0, 1], required=True)
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
        loop_detection=bool(args.loop_detection),
        mapper_minimum_matches=args.mapper_min_matches,
        colmap_executable=args.colmap_executable,
        reuse=args.reuse_colmap,
    )


if __name__ == "__main__":
    main()
