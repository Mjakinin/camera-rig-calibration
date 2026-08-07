"""AP01 runtime adapter: requirements, stage order and result collection."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Any, Sequence

from pydantic import BaseModel

from ...components.common import calibration_requirements, read_method_status
from ...config.models import AP01Settings
from ...contracts import CommandSpec, RequirementResult, RunContext
from .contracts import (
    ap01_execution_contract_name,
    resolve_ap01_method_contract,
)


@dataclass(frozen=True)
class AP01Method:
    """Connect the AP01 stage modules to the generic rigcal runtime."""

    id: str = "ap01"
    display_name: str = "AP01"
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
        execution_contract_name = ap01_execution_contract_name(
            config.methods.ap01.method_contract,
            historical_reproduction=(
                config.methods.ap01.historical_reproduction
            ),
            advanced_strategy=config.methods.ap01.advanced_strategy,
        )
        contract = resolve_ap01_method_contract(
            execution_contract_name,
            direct_target_camera=config.methods.ap01.direct_target_camera,
            top_moving_per_marker=(
                config.methods.ap01.top_moving_per_marker
            ),
            scale_top_per_marker=config.methods.ap01.scale_top_per_marker,
            colmap_matcher=config.colmap.matcher,
            colmap_use_gpu=config.colmap.use_gpu,
            colmap_maximum_image_size=config.colmap.maximum_image_size,
            colmap_maximum_features=config.colmap.maximum_features,
            colmap_sequential_overlap=config.colmap.sequential_overlap,
            colmap_loop_detection=config.colmap.loop_detection,
            colmap_mapper_minimum_matches=(
                config.colmap.mapper_minimum_matches
            ),
        )
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
            "--method-contract",
            execution_contract_name,
            "--direct-target-camera",
            config.methods.ap01.direct_target_camera,
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
        direct_gate = config.methods.ap01.direct_quality_gate
        relay_gate = config.methods.ap01.relay_quality_gate
        consistency = config.methods.ap01.direct_relay_consistency
        arguments.extend(
            [
                "--direct-minimum-independent-markers",
                str(direct_gate.minimum_independent_markers),
                "--direct-minimum-inlier-ratio",
                str(direct_gate.minimum_inlier_ratio),
                "--direct-maximum-translation-dispersion-m",
                str(direct_gate.maximum_translation_dispersion_m),
                "--direct-maximum-rotation-dispersion-deg",
                str(direct_gate.maximum_rotation_dispersion_deg),
                "--relay-minimum-inlier-ratio",
                str(relay_gate.minimum_inlier_ratio),
                "--relay-maximum-translation-dispersion-m",
                str(relay_gate.maximum_translation_dispersion_m),
                "--relay-maximum-rotation-dispersion-deg",
                str(relay_gate.maximum_rotation_dispersion_deg),
                "--maximum-path-translation-disagreement-m",
                str(consistency.maximum_translation_disagreement_m),
                "--maximum-path-rotation-disagreement-deg",
                str(consistency.maximum_rotation_disagreement_deg),
            ]
        )
        reconstruct = [
            *python_module,
            "camera_rig_calibration.methods.ap01.reconstruct_moving",
            *arguments,
            "--matcher",
            contract.colmap_matching_mode,
            "--use-gpu",
            "1" if contract.colmap_matcher_use_gpu else "0",
            "--max-image-size",
            str(contract.colmap_sift_maximum_image_size),
            "--colmap-executable",
            config.colmap.executable,
            "--max-features",
            str(contract.colmap_sift_max_features),
            "--sequential-overlap",
            str(contract.colmap_sequential_overlap),
            "--loop-detection",
            "1" if contract.colmap_loop_detection else "0",
            "--mapper-min-matches",
            str(contract.colmap_mapper_minimum_matches),
        ]
        if config.colmap.reuse or context.reuse_colmap_artifact:
            reconstruct.append("--reuse-colmap")

        stages = [
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
        ]
        if contract.reproduction_validation_policy != "none":
            stages.append(
                (
                    "ap01_validate_reproduction",
                    "AP01: validate locked historical reproduction",
                    [
                        *python_module,
                        "camera_rig_calibration.methods.ap01.validate_reproduction",
                        *arguments,
                    ],
                    output / "06_reproduction_validation",
                    ("ap01_report",),
                )
            )
        if {
            "01_moving_colmap",
            "02_metric_scale",
        }.issubset(set(context.reused_method_stages)):
            stages = [
                (
                    stage_id,
                    display_name,
                    argv,
                    directory,
                    (
                        ()
                        if stage_id == "ap01_build_candidates"
                        else dependencies
                    ),
                )
                for (
                    stage_id,
                    display_name,
                    argv,
                    directory,
                    dependencies,
                ) in stages
                if stage_id
                not in {"ap01_reconstruct_moving", "ap01_estimate_scale"}
            ]
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
