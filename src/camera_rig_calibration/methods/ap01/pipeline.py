"""AP01 runtime adapter: requirements, stage order and result collection."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Any, Sequence

from pydantic import BaseModel

from ...components.common import calibration_requirements, read_method_status
from ...config.models import AP01Settings
from ...contracts import CommandSpec, RequirementResult, RunContext


@dataclass(frozen=True)
class AP01Method:
    """Connect the AP01 stage modules to the generic rigcal runtime."""

    id: str = "ap01"
    display_name: str = "AP01 Baseline"
    config_model: type[BaseModel] = AP01Settings

    def requirements(self, context: RunContext) -> RequirementResult:
        result = calibration_requirements(context)
        if not result.compatible:
            return result
        if (
            context.resolved_root_camera is None
            and context.config.methods.ap01.root_camera == "auto"
        ):
            return RequirementResult.unavailable("root camera cannot be resolved")
        return RequirementResult.ok()

    def commands(self, context: RunContext) -> Sequence[CommandSpec]:
        root = context.resolved_root_camera or context.config.methods.ap01.root_camera
        if root == "auto":
            raise RuntimeError("AP01 root camera has not been resolved")

        output = context.run_directory / "02_AP01"
        config = context.config
        python_module = [sys.executable, "-m"]
        arguments = [
            "--dataset",
            str(context.dataset_root),
            "--observations-root",
            str(context.observations_root),
            "--out",
            str(output),
            "--cameras",
            ",".join(camera.id for camera in config.static_cameras),
            "--root-camera",
            root,
            "--moving-camera-id",
            config.moving_camera.id,
        ]
        if config.methods.ap01.top_moving_per_marker is not None:
            arguments.extend(
                [
                    "--top-moving-per-marker",
                    str(config.methods.ap01.top_moving_per_marker),
                ]
            )
        if config.methods.ap01.scale_top_per_marker is not None:
            arguments.extend(
                [
                    "--scale-top-per-marker",
                    str(config.methods.ap01.scale_top_per_marker),
                ]
            )
        reconstruct = [
            *python_module,
            "camera_rig_calibration.methods.ap01.reconstruct_moving",
            *arguments,
            "--matcher",
            config.colmap.matcher,
            "--use-gpu",
            "1" if config.colmap.use_gpu else "0",
            "--max-image-size",
            str(config.colmap.maximum_image_size),
            "--colmap-executable",
            config.colmap.executable,
            "--max-features",
            str(config.colmap.maximum_features),
            "--sequential-overlap",
            str(config.colmap.sequential_overlap),
            "--loop-detection",
            "1" if config.colmap.loop_detection else "0",
            "--mapper-min-matches",
            str(config.colmap.mapper_minimum_matches),
        ]
        if config.colmap.reuse or context.reuse_colmap_artifact:
            reconstruct.append("--reuse-colmap")

        stages = (
            (
                "ap01_reconstruct_moving",
                "AP01: reconstruct moving camera",
                reconstruct,
                output / "01_moving_colmap",
                (),
            ),
            (
                "ap01_estimate_scale",
                "AP01: estimate metric scale",
                [
                    *python_module,
                    "camera_rig_calibration.methods.ap01.estimate_scale",
                    *arguments,
                ],
                output / "02_metric_scale",
                ("ap01_reconstruct_moving",),
            ),
            (
                "ap01_build_candidates",
                "AP01: build transform candidates",
                [
                    *python_module,
                    "camera_rig_calibration.methods.ap01.build_candidates",
                    *arguments,
                ],
                output / "03_candidates",
                ("ap01_estimate_scale",),
            ),
            (
                "ap01_solve_extrinsics",
                "AP01: solve static extrinsics",
                [
                    *python_module,
                    "camera_rig_calibration.methods.ap01.solve_extrinsics",
                    *arguments,
                ],
                output / "03_static_extrinsics",
                ("ap01_build_candidates",),
            ),
            (
                "ap01_report",
                "AP01: write method report",
                [
                    *python_module,
                    "camera_rig_calibration.methods.ap01.report",
                    *arguments,
                ],
                output / "05_report",
                ("ap01_solve_extrinsics",),
            ),
        )
        return tuple(
            CommandSpec(
                stage_id,
                display_name,
                tuple(argv),
                context.repository_root,
                directory,
                depends_on=dependencies,
            )
            for stage_id, display_name, argv, directory, dependencies in stages
        )

    def collect(self, context: RunContext) -> dict[str, Any]:
        return read_method_status(context.run_directory / "02_AP01")
