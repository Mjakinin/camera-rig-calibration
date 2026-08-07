"""AP02 runtime adapter: requirements, stage order and result collection."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Any, Sequence

from pydantic import BaseModel

from ...components.common import calibration_requirements, read_method_status
from ...config.models import AP02Settings
from ...contracts import CommandSpec, RequirementResult, RunContext
from .contracts import resolve_ap02_method_contract


@dataclass(frozen=True)
class AP02Method:
    """Connect the AP02 graph/BA stages to the generic rigcal runtime."""

    id: str = "ap02"
    display_name: str = "AP02"
    config_model: type[BaseModel] = AP02Settings

    def requirements(self, context: RunContext) -> RequirementResult:
        result = calibration_requirements(context)
        if not result.compatible:
            return result
        if context.resolved_ap02_reference_marker_id is None:
            return RequirementResult.unavailable("AP02 reference marker is unresolved")
        return RequirementResult.ok()

    def commands(self, context: RunContext) -> Sequence[CommandSpec]:
        settings = context.config.methods.ap02
        output = context.run_directory / "03_AP02"
        reference = str(context.resolved_ap02_reference_marker_id)
        contract = resolve_ap02_method_contract(
            settings.method_contract,
            reference_marker_selection_mode=(
                settings.reference_marker_selection_mode
            ),
            reference_marker_id=context.resolved_ap02_reference_marker_id,
            frame_selection_strategy=settings.frame_selection_strategy,
            initialization_strategy=settings.initialization_strategy,
            graph_edge_weight_strategy=(
                settings.graph_edge_weight_strategy
            ),
            reprojection_model=settings.reprojection_model,
            reference_marker_maximum_frames=(
                settings.reference_marker_maximum_frames
            ),
            top_per_marker=settings.top_per_marker,
            top_per_marker_pair=settings.top_per_marker_pair,
            maximum_total_frames=settings.maximum_total_frames,
            static_maximum_function_evaluations=(
                settings.static_only_ba_max_function_evaluations
            ),
            combined_maximum_function_evaluations=(
                settings.combined_ba_max_function_evaluations
            ),
            robust_loss=settings.ba_robust_loss,
            robust_loss_scale_px=settings.ba_robust_loss_scale_px,
        )
        cameras = ",".join(
            camera.id for camera in context.config.static_cameras
        )
        python_module = [sys.executable, "-m"]
        selection_arguments: list[str] = []
        for option, value in (
            (
                "--reference-marker-maximum-frames",
                settings.reference_marker_maximum_frames,
            ),
            ("--top-per-marker", settings.top_per_marker),
            ("--top-per-marker-pair", settings.top_per_marker_pair),
            ("--maximum-total-frames", settings.maximum_total_frames),
        ):
            if value is not None:
                selection_arguments.extend([option, str(value)])
        historical_arguments = (
            ["--historical-reproduction"]
            if settings.historical_reproduction
            else []
        )
        stages = [
            (
                "ap02_build_graph",
                "AP02: build unweighted observation graph",
                [
                    *python_module,
                    "camera_rig_calibration.methods.ap02.build_graph",
                    "--observations-root",
                    str(context.observations_root),
                    "--dataset",
                    str(context.dataset_root),
                    "--out",
                    str(output),
                    "--cameras",
                    cameras,
                    "--ref-marker-id",
                    reference,
                    "--graph-observation-policy",
                    contract.graph_observation_policy,
                    "--method-contract-sha256",
                    contract.scientific_fingerprint(),
                    *historical_arguments,
                    *selection_arguments,
                ],
                output / "02_aruco_observations",
                (),
            ),
            (
                "ap02_component_diagnostics",
                "AP02: calibrate disconnected components (diagnostic)",
                [
                    *python_module,
                    "camera_rig_calibration.methods.ap02.component_diagnostics",
                    "--out",
                    str(output),
                    "--max-nfev",
                    str(contract.combined_maximum_function_evaluations),
                    "--robust-loss",
                    contract.robust_loss,
                    "--robust-loss-scale-px",
                    str(contract.robust_loss_scale_px),
                ],
                output / "09_component_diagnostics",
                ("ap02_build_graph",),
            ),
            (
                "ap02_static_initialization",
                "AP02: static-only initialization (diagnostic)",
                [
                    *python_module,
                    "camera_rig_calibration.methods.ap02.initialize_stage",
                    "--out",
                    str(output),
                    "--ref-marker-id",
                    reference,
                    "--mode",
                    "static_only",
                    "--initialization-algorithm",
                    contract.initialization_algorithm,
                    "--edge-weight-policy",
                    contract.graph_edge_weight_policy,
                ],
                output / "05_graph_initialization/static_only",
                ("ap02_build_graph",),
            ),
            (
                "ap02_static_ba",
                "AP02: static-only bundle adjustment (diagnostic)",
                [
                    *python_module,
                    "camera_rig_calibration.methods.ap02.optimize_stage",
                    "--out",
                    str(output),
                    "--ref-marker-id",
                    reference,
                    "--mode",
                    "static_only",
                    "--max-nfev",
                    str(contract.static_maximum_function_evaluations),
                    "--robust-loss",
                    contract.robust_loss,
                    "--robust-loss-scale-px",
                    str(contract.robust_loss_scale_px),
                    "--reprojection-model",
                    contract.reprojection_model,
                    "--moving-frame-selection-policy",
                    contract.moving_frame_selection_policy,
                    *historical_arguments,
                    *selection_arguments,
                ],
                output / "07_graph_ba/static_only",
                ("ap02_static_initialization",),
            ),
            (
                "ap02_combined_initialization",
                "AP02: combined initialization (primary)",
                [
                    *python_module,
                    "camera_rig_calibration.methods.ap02.initialize_stage",
                    "--out",
                    str(output),
                    "--ref-marker-id",
                    reference,
                    "--mode",
                    "with_moving",
                    "--initialization-algorithm",
                    contract.initialization_algorithm,
                    "--edge-weight-policy",
                    contract.graph_edge_weight_policy,
                ],
                output / "05_graph_initialization/with_moving",
                ("ap02_build_graph",),
            ),
            (
                "ap02_combined_ba",
                "AP02: combined bundle adjustment (primary)",
                [
                    *python_module,
                    "camera_rig_calibration.methods.ap02.optimize_stage",
                    "--out",
                    str(output),
                    "--ref-marker-id",
                    reference,
                    "--mode",
                    "with_moving",
                    "--max-nfev",
                    str(contract.combined_maximum_function_evaluations),
                    "--robust-loss",
                    contract.robust_loss,
                    "--robust-loss-scale-px",
                    str(contract.robust_loss_scale_px),
                    "--reprojection-model",
                    contract.reprojection_model,
                    "--moving-frame-selection-policy",
                    contract.moving_frame_selection_policy,
                    *historical_arguments,
                    *selection_arguments,
                ],
                output / "07_graph_ba/with_moving",
                ("ap02_combined_initialization",),
            ),
            (
                "ap02_report",
                "AP02: write diagnostic and primary report",
                [
                    *python_module,
                    "camera_rig_calibration.methods.ap02.report",
                    "--out",
                    str(output),
                    "--cameras",
                    cameras,
                    "--ref-marker-id",
                    reference,
                ],
                output / "08_final_results",
                (
                    "ap02_static_ba",
                    "ap02_combined_ba",
                    "ap02_component_diagnostics",
                ),
            ),
        ]
        diagnostics = {
            "ap02_static_initialization",
            "ap02_static_ba",
            "ap02_component_diagnostics",
        }
        return tuple(
            CommandSpec(
                stage_id,
                display_name,
                tuple(argv),
                context.repository_root,
                directory,
                depends_on=dependencies,
                diagnostic=stage_id in diagnostics,
            )
            for stage_id, display_name, argv, directory, dependencies in stages
        )

    def collect(self, context: RunContext) -> dict[str, Any]:
        return read_method_status(context.run_directory / "03_AP02")
