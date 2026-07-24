from __future__ import annotations

from pathlib import Path

from camera_rig_calibration.pipeline import StageResult, run_stage

from . import core
from ._shared import cameras, parser


def run(
    *,
    dataset: Path,
    observations_root: Path,
    output_root: Path,
    camera_ids: tuple[str, ...],
    root_camera: str,
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
    stage_root = output_root / "01_moving_colmap"
    moving = dataset / "raw_images" / "moving"
    camera_info_path = (
        dataset / "raw_images" / "camera_info" / f"{moving_camera_id}.json"
    )

    def action() -> dict[str, Path | int]:
        info = core.load_camera_info(camera_info_path)
        images = core.run_colmap(
            image_dir=moving,
            camera_info=info,
            out_dir=stage_root,
            matcher=matcher,
            use_gpu=int(use_gpu),
            max_image_size=maximum_image_size,
            max_features=maximum_features,
            sequential_overlap=sequential_overlap,
            loop_detection=int(loop_detection),
            mapper_min_matches=mapper_minimum_matches,
            colmap_executable=colmap_executable,
            reuse=reuse,
        )
        poses = core.parse_colmap_poses(images)
        return {"images_txt": images, "registered_images": len(poses)}

    return run_stage(
        "ap01.reconstruct_moving",
        stage_root,
        action,
        inputs={"moving_images": moving, "camera_info": camera_info_path},
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
    cli = parser(__doc__ or "Reconstruct AP01 moving-camera poses")
    cli.add_argument(
        "--matcher", choices=["exhaustive", "sequential"], default="exhaustive"
    )
    cli.add_argument("--use-gpu", type=int, choices=[0, 1], default=0)
    cli.add_argument("--max-image-size", type=int, default=2400)
    cli.add_argument("--max-features", type=int, default=8192)
    cli.add_argument("--sequential-overlap", type=int, default=20)
    cli.add_argument("--loop-detection", type=int, choices=[0, 1], default=1)
    cli.add_argument("--mapper-min-matches", type=int, default=8)
    cli.add_argument("--colmap-executable", default="colmap")
    cli.add_argument("--reuse-colmap", action="store_true")
    args = cli.parse_args()
    camera_values = cameras(args)
    run(
        dataset=args.dataset.resolve(),
        observations_root=args.observations_root.resolve(),
        output_root=args.out.resolve(),
        camera_ids=camera_values,
        root_camera=args.root_camera,
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
