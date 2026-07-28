"""AP03 runtime adapter: requirements, stage order and result collection."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Any, Sequence

from pydantic import BaseModel

from ...components.common import calibration_requirements, read_method_status
from ...config.models import AP03Settings
from ...contracts import CommandSpec, RequirementResult, RunContext


@dataclass(frozen=True)
class AP03Method:
    """Connect the AP03 shared-COLMAP stages to the generic rigcal runtime."""

    id: str = "ap03"
    display_name: str = "AP03 — shared COLMAP with single and multi scale"
    config_model: type[BaseModel] = AP03Settings

    def requirements(self, context: RunContext) -> RequirementResult:
        result = calibration_requirements(context)
        if not result.compatible:
            return result
        if not context.resolved_ap03_multi_marker_ids:
            return RequirementResult.unavailable(
                "no compatible multi-scale marker IDs were detected"
            )
        return RequirementResult.ok()

    def commands(self, context: RunContext) -> Sequence[CommandSpec]:
        single = context.config.methods.ap03.single
        multi = context.config.methods.ap03.multi
        scale = context.config.methods.ap03.scale
        single_marker = (
            context.resolved_ap03_single_scale_marker_id
            if single.scale_marker_id == "auto"
            else int(single.scale_marker_id)
        )
        multi_markers = (
            list(context.resolved_ap03_multi_marker_ids)
            if multi.marker_ids == "auto"
            else list(multi.marker_ids)
        )
        output = context.run_directory / "04_AP03"
        config = context.config
        cameras = ",".join(camera.id for camera in config.static_cameras)
        python_module = [sys.executable, "-m"]
        prepare = [
            *python_module,
            "camera_rig_calibration.methods.ap03.prepare_colmap",
            "--dataset",
            str(context.dataset_root),
            "--out",
            str(output),
            "--cameras",
            cameras,
            "--moving-camera-id",
            config.moving_camera.id,
        ]
        reconstruct = [
            *python_module,
            "camera_rig_calibration.methods.ap03.reconstruct_stage",
            "--dataset",
            str(context.dataset_root),
            "--out",
            str(output),
            "--cameras",
            cameras,
            "--moving-camera-id",
            config.moving_camera.id,
            "--matcher",
            config.colmap.matcher,
            "--use-gpu",
            "1" if config.colmap.use_gpu else "0",
            "--max-image-size",
            str(
                config.colmap.ap03_maximum_image_size
                or config.colmap.maximum_image_size
            ),
            "--max-features",
            str(
                config.colmap.ap03_maximum_features
                or config.colmap.maximum_features
            ),
            "--sequential-overlap",
            str(config.colmap.sequential_overlap),
            "--loop-detection",
            (
                "1"
                if (
                    config.colmap.ap03_loop_detection
                    if config.colmap.ap03_loop_detection is not None
                    else config.colmap.loop_detection
                )
                else "0"
            ),
            "--mapper-min-matches",
            str(config.colmap.mapper_minimum_matches),
            "--colmap-executable",
            config.colmap.executable,
        ]
        if config.colmap.reuse or context.reuse_colmap_artifact:
            reconstruct.append("--reuse-colmap")
        scale_common = [
            "--repository-root",
            str(context.repository_root),
            "--observations-root",
            str(context.observations_root),
            "--out",
            str(output),
            "--cameras",
            cameras,
            "--marker-length-m",
            str(config.markers.length_m),
            "--reprojection-threshold-px",
            str(scale.reprojection_threshold_px),
            "--ransac-iterations",
            str(scale.ransac_iterations),
            "--minimum-inliers",
            str(scale.minimum_inliers),
            "--dictionary",
            config.markers.dictionary,
            "--detection-mode",
            config.markers.detection_mode,
        ]
        stages = (
            (
                "ap03_prepare_colmap",
                "AP03: prepare grouped COLMAP input",
                prepare,
                output / "colmap/dataset",
                (),
            ),
            (
                "ap03_reconstruct",
                "AP03: grouped COLMAP reconstruction",
                reconstruct,
                output / "colmap/reconstruction",
                ("ap03_prepare_colmap",),
            ),
            (
                "ap03_inspect",
                "AP03: inspect reconstruction",
                [
                    *python_module,
                    "camera_rig_calibration.methods.ap03.inspect_stage",
                    "--out",
                    str(output),
                    "--cameras",
                    cameras,
                ],
                output / "colmap/inspection",
                ("ap03_reconstruct",),
            ),
            (
                "ap03_single_scale",
                "AP03: single-marker scale (diagnostic)",
                [
                    *python_module,
                    "camera_rig_calibration.methods.ap03.estimate_scale",
                    *scale_common,
                    "--mode",
                    "single",
                    "--marker-ids",
                    str(single_marker),
                ],
                output / "scale_single",
                ("ap03_inspect",),
            ),
            (
                "ap03_multi_scale",
                "AP03: multi-marker scale (primary)",
                [
                    *python_module,
                    "camera_rig_calibration.methods.ap03.estimate_scale",
                    *scale_common,
                    "--mode",
                    "multi",
                    "--marker-ids",
                    ",".join(str(value) for value in multi_markers),
                ],
                output / "scale_multi",
                ("ap03_inspect",),
            ),
            (
                "ap03_report",
                "AP03: write combined report",
                [
                    *python_module,
                    "camera_rig_calibration.methods.ap03.report",
                    "--out",
                    str(output),
                ],
                output / "report",
                ("ap03_single_scale", "ap03_multi_scale"),
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
                diagnostic=stage_id == "ap03_single_scale",
            )
            for stage_id, display_name, argv, directory, dependencies in stages
        )

    def collect(self, context: RunContext) -> dict[str, Any]:
        return read_method_status(context.run_directory / "04_AP03")
