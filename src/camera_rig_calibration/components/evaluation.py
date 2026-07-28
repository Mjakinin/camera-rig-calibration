"""Built-in evaluator adapters."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Sequence

from ..contracts import CommandSpec, RequirementResult, RunContext


@dataclass(frozen=True)
class MarkerConsistencyEvaluator:
    """Schedule the common marker-consistency evaluation."""

    id: str = "marker_consistency"
    display_name: str = "Common marker consistency evaluation"

    def requirements(self, context: RunContext) -> RequirementResult:
        if context.resolved_evaluation_anchor_marker_id is None:
            return RequirementResult.unavailable("evaluation scale anchor is unresolved")
        supported = {"ap01", "ap02", "ap03"}
        if not supported.intersection(context.config.methods.enabled):
            return RequirementResult.unavailable(
                "none of the enabled methods has a marker-consistency result parser"
            )
        return RequirementResult.ok()

    def commands(self, context: RunContext) -> Sequence[CommandSpec]:
        config = context.config
        directory_by_method = {
            "ap01": ("AP01", "02_AP01"),
            "ap02": ("AP02", "03_AP02"),
            "ap03": ("AP03_MULTI", "04_AP03/scale_multi"),
        }
        argv = [
            sys.executable,
            str(
                context.repository_root
                / "src/camera_rig_calibration/evaluation/marker_consistency.py"
            ),
            "--dataset",
            str(context.dataset_root),
            "--results-root",
            str(context.run_directory),
            "--observations-root",
            str(context.observations_root),
            "--output-root",
            str(context.run_directory / "06_EVALUATION"),
            "--cameras",
            ",".join(camera.id for camera in config.static_cameras),
            "--anchor-marker-id",
            str(context.resolved_evaluation_anchor_marker_id),
            "--marker-length-m",
            str(config.markers.length_m),
            "--reprojection-threshold-px",
            str(config.evaluation.reprojection_threshold_px),
            "--min-inliers",
            str(config.evaluation.minimum_inliers),
            "--ransac-iters",
            str(config.evaluation.ransac_iterations),
            "--min-triangulation-angle-deg",
            str(config.evaluation.minimum_triangulation_angle_deg),
            "--max-moving-observations-per-marker",
            str(config.evaluation.maximum_moving_observations_per_marker),
        ]
        for method_id in config.methods.enabled:
            if method_id not in directory_by_method:
                continue
            label, directory = directory_by_method[method_id]
            argv += ["--method", f"{label}={directory}"]
        return (
            CommandSpec(
                "evaluation",
                self.display_name,
                tuple(argv),
                context.repository_root,
                context.run_directory / "06_EVALUATION",
            ),
        )
