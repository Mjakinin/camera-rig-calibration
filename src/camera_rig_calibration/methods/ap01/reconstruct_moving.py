from __future__ import annotations

from pathlib import Path

from camera_rig_calibration.pipeline import StageResult, run_stage

from . import core
from ._shared import cameras, parser
from .contracts import resolve_ap01_method_contract


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
    method_contract: str = "baseline_v1",
    direct_target_camera: str = "cam_edge_1",
    top_moving_per_marker: int | None = 8,
    scale_top_per_marker: int | None = 30,
) -> StageResult:
    stage_root = output_root / "01_moving_colmap"
    moving = dataset / "raw_images" / "moving"
    camera_info_path = (
        dataset / "raw_images" / "camera_info" / f"{moving_camera_id}.json"
    )
    contract = resolve_ap01_method_contract(
        method_contract,
        direct_target_camera=direct_target_camera,
        top_moving_per_marker=top_moving_per_marker,
        scale_top_per_marker=scale_top_per_marker,
        colmap_matcher=matcher,
        colmap_use_gpu=use_gpu,
        colmap_maximum_image_size=maximum_image_size,
        colmap_maximum_features=maximum_features,
        colmap_sequential_overlap=sequential_overlap,
        colmap_loop_detection=loop_detection,
        colmap_mapper_minimum_matches=mapper_minimum_matches,
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
            contract=contract,
        )
        poses = core.parse_colmap_poses(images)
        return {"images_txt": images, "registered_images": len(poses)}

    return run_stage(
        "ap01.reconstruct_moving",
        stage_root,
        action,
        inputs={"moving_images": moving, "camera_info": camera_info_path},
        parameters={
            "matcher": contract.colmap_matching_mode,
            "gpu": contract.colmap_matcher_use_gpu,
            "maximum_image_size": contract.colmap_sift_maximum_image_size,
            "maximum_features": contract.colmap_sift_max_features,
            "extraction_threads": contract.colmap_sift_extraction_threads,
            "sequential_overlap": contract.colmap_sequential_overlap,
            "loop_detection": contract.colmap_loop_detection,
            "mapper_minimum_matches": (
                contract.colmap_mapper_minimum_matches
            ),
            "sfm_mode": contract.sfm_execution_policy,
            "method_contract": contract.fingerprint_payload(),
            "method_contract_sha256": contract.scientific_fingerprint(),
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
        method_contract=args.method_contract,
        direct_target_camera=args.direct_target_camera,
        top_moving_per_marker=args.top_moving_per_marker,
        scale_top_per_marker=args.scale_top_per_marker,
    )


if __name__ == "__main__":
    main()
