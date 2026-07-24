from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from pydantic import BaseModel, ConfigDict

from .config.models import (
    AP01Settings,
    AP02Settings,
    AP03Settings,
    RigConfig,
)
from .contracts import CommandSpec, RequirementResult, RunContext
from .registry import (
    calibration_methods,
    evaluators,
    experiment_providers,
    input_adapters,
)


class EmptyOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _canonical_files(context: RunContext) -> dict[str, Path]:
    raw = context.dataset_root / "raw_images"
    return {
        "raw": raw,
        "static": raw / "static",
        "moving": raw / "moving",
        "camera_info": raw / "camera_info",
    }


def _base_requirements(context: RunContext) -> RequirementResult:
    files = _canonical_files(context)
    reasons = []
    if len(context.config.static_cameras) < 2:
        reasons.append("at least two static cameras are required")
    for camera in context.config.static_cameras:
        if not any(files["static"].glob(f"{camera.id}.*")):
            reasons.append(f"static image is missing for '{camera.id}'")
        if not (files["camera_info"] / f"{camera.id}.json").is_file():
            reasons.append(f"intrinsics are missing for '{camera.id}'")
    if not any(files["moving"].glob("frame_*.*")):
        reasons.append("moving-camera frames are missing")
    if not (
        files["camera_info"] / f"{context.config.moving_camera.id}.json"
    ).is_file():
        reasons.append(
            f"intrinsics are missing for '{context.config.moving_camera.id}'"
        )
    return RequirementResult.unavailable(*reasons) if reasons else RequirementResult.ok()


def _status_payload(directory: Path) -> dict[str, Any]:
    path = directory / "METHOD_STATUS.json"
    if not path.is_file():
        return {"status": "MISSING", "success": False, "directory": str(directory)}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {
            "status": "INVALID_STATUS",
            "success": False,
            "directory": str(directory),
            "error": str(exc),
        }
    payload["directory"] = str(directory)
    return payload


@dataclass(frozen=True)
class PreparedInputAdapter:
    id: str = "prepared_dataset"
    display_name: str = "Prepared dataset"

    def matches(self, config: RigConfig) -> bool:
        return config.dataset.prepared_root is not None

    def requirements(self, config: RigConfig) -> RequirementResult:
        root = config.dataset.prepared_root
        if root is None:
            return RequirementResult.unavailable("dataset.prepared_root is not set")
        if not root.exists():
            return RequirementResult.unavailable(
                f"prepared dataset does not exist: {root}"
            )
        dataset_root = root.parent if root.name == "raw_images" else root
        raw = dataset_root / "raw_images"
        reasons = []
        for camera in config.static_cameras:
            if not any((raw / "static").glob(f"{camera.id}.*")):
                reasons.append(f"prepared static image is missing for '{camera.id}'")
            if not (raw / "camera_info" / f"{camera.id}.json").is_file():
                reasons.append(f"prepared intrinsics are missing for '{camera.id}'")
        if not any((raw / "moving").glob("frame_*.*")):
            reasons.append("prepared moving frames are missing")
        if not (
            raw / "camera_info" / f"{config.moving_camera.id}.json"
        ).is_file():
            reasons.append(
                f"prepared intrinsics are missing for '{config.moving_camera.id}'"
            )
        return (
            RequirementResult.unavailable(*reasons)
            if reasons
            else RequirementResult.ok()
        )

    def commands(self, context: RunContext) -> Sequence[CommandSpec]:
        return ()


@dataclass(frozen=True)
class FilesystemInputAdapter:
    id: str = "filesystem"
    display_name: str = "Videos, frames and direct images"

    def matches(self, config: RigConfig) -> bool:
        return (
            config.dataset.prepared_root is None
            and config.mcap.path is None
            and not config.simulation.enabled
        )

    def requirements(self, config: RigConfig) -> RequirementResult:
        reasons = []
        moving = config.moving_camera
        for label, path, expected_kind in (
            ("moving video", moving.video, "file"),
            ("moving frames", moving.frames, "directory"),
            ("moving intrinsics", moving.intrinsics, "file"),
            (
                "intrinsic calibration video",
                moving.intrinsic_calibration_video,
                "file",
            ),
            (
                "intrinsic calibration images",
                moving.intrinsic_calibration_images,
                "directory",
            ),
        ):
            if path is None:
                continue
            exists = path.is_file() if expected_kind == "file" else path.is_dir()
            if not exists:
                reasons.append(f"{label} does not exist: {path}")
        for camera in config.static_cameras:
            for path in camera.images:
                if not path.is_file():
                    reasons.append(f"static image does not exist: {path}")
            if camera.video is not None and not camera.video.is_file():
                reasons.append(f"static video does not exist: {camera.video}")
            if camera.intrinsics is not None and not camera.intrinsics.is_file():
                reasons.append(f"static intrinsics do not exist: {camera.intrinsics}")
        return (
            RequirementResult.unavailable(*reasons)
            if reasons
            else RequirementResult.ok()
        )

    def commands(self, context: RunContext) -> Sequence[CommandSpec]:
        return ()


@dataclass(frozen=True)
class McapInputAdapter:
    id: str = "mcap"
    display_name: str = "MCAP or ROS bag"

    def matches(self, config: RigConfig) -> bool:
        return config.mcap.path is not None

    def requirements(self, config: RigConfig) -> RequirementResult:
        if config.mcap.path is None:
            return RequirementResult.unavailable("mcap.path is not set")
        missing = [camera.id for camera in config.static_cameras if not camera.image_topic]
        if missing:
            return RequirementResult.unavailable(
                f"image topics are missing for cameras: {', '.join(missing)}"
            )
        if config.mcap.path is None or not config.mcap.path.is_file():
            return RequirementResult.unavailable(
                f"MCAP does not exist: {config.mcap.path}"
            )
        moving = config.moving_camera
        if moving.video is None and moving.frames is None and not moving.image_topic:
            return RequirementResult.unavailable(
                "moving camera needs a local video/frame folder or an MCAP image topic"
            )
        filesystem = FilesystemInputAdapter().requirements(config)
        return filesystem

    def commands(self, context: RunContext) -> Sequence[CommandSpec]:
        return ()


@dataclass(frozen=True)
class SimulationInputAdapter:
    id: str = "simulation"
    display_name: str = "Gazebo simulation capture"

    def matches(self, config: RigConfig) -> bool:
        return config.simulation.enabled

    def requirements(self, config: RigConfig) -> RequirementResult:
        simulation = config.simulation
        reasons = []
        if simulation.world is None or not simulation.world.is_file():
            reasons.append(f"simulation world does not exist: {simulation.world}")
        if simulation.route is None or not simulation.route.is_file():
            reasons.append(f"simulation route does not exist: {simulation.route}")
        for camera in config.static_cameras:
            if not camera.image_topic:
                reasons.append(f"image topic is missing for '{camera.id}'")
            if not camera.camera_info_topic:
                reasons.append(f"CameraInfo topic is missing for '{camera.id}'")
        if not config.moving_camera.image_topic:
            reasons.append("moving-camera image topic is missing")
        if not config.moving_camera.camera_info_topic:
            reasons.append("moving-camera CameraInfo topic is missing")
        return (
            RequirementResult.unavailable(*reasons)
            if reasons
            else RequirementResult.ok()
        )

    def commands(self, context: RunContext) -> Sequence[CommandSpec]:
        return ()


@dataclass(frozen=True)
class AP01Method:
    id: str = "ap01"
    display_name: str = "AP01 Baseline"
    config_model: type[BaseModel] = AP01Settings

    def requirements(self, context: RunContext) -> RequirementResult:
        result = _base_requirements(context)
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
        common = [
            sys.executable,
            "-m",
        ]
        arguments = [
            "--dataset", str(context.dataset_root),
            "--observations-root", str(context.observations_root),
            "--out", str(output),
            "--cameras", ",".join(
                camera.id for camera in config.static_cameras
            ),
            "--root-camera", root,
            "--moving-camera-id", config.moving_camera.id,
        ]
        reconstruct = [
            *common,
            "camera_rig_calibration.methods.ap01.reconstruct_moving",
            *arguments,
            "--matcher", config.colmap.matcher,
            "--use-gpu", "1" if config.colmap.use_gpu else "0",
            "--max-image-size", str(config.colmap.maximum_image_size),
            "--colmap-executable", config.colmap.executable,
            "--max-features", str(config.colmap.maximum_features),
            "--sequential-overlap", str(config.colmap.sequential_overlap),
            "--loop-detection", "1" if config.colmap.loop_detection else "0",
            "--mapper-min-matches", str(
                config.colmap.mapper_minimum_matches
            ),
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
                    *common,
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
                    *common,
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
                    *common,
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
                    *common,
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
        return _status_payload(context.run_directory / "02_AP01")


@dataclass(frozen=True)
class AP02Method:
    id: str = "ap02"
    display_name: str = "AP02 Baseline"
    config_model: type[BaseModel] = AP02Settings

    def requirements(self, context: RunContext) -> RequirementResult:
        result = _base_requirements(context)
        if not result.compatible:
            return result
        if context.resolved_ap02_reference_marker_id is None:
            return RequirementResult.unavailable("AP02 reference marker is unresolved")
        return RequirementResult.ok()

    def commands(self, context: RunContext) -> Sequence[CommandSpec]:
        settings = context.config.methods.ap02
        output = context.run_directory / "03_AP02"
        reference = str(context.resolved_ap02_reference_marker_id)
        cameras = ",".join(
            camera.id for camera in context.config.static_cameras
        )
        module = [sys.executable, "-m"]
        stages = [
            (
                "ap02_build_graph",
                "AP02: build unweighted observation graph",
                [
                    *module,
                    "camera_rig_calibration.methods.ap02.build_graph",
                    "--observations-root", str(context.observations_root),
                    "--out", str(output),
                    "--cameras", cameras,
                    "--ref-marker-id", reference,
                ],
                output / "02_aruco_observations",
                (),
            ),
            (
                "ap02_static_initialization",
                "AP02: static-only initialization (diagnostic)",
                [
                    *module,
                    "camera_rig_calibration.methods.ap02.initialize_stage",
                    "--out", str(output),
                    "--ref-marker-id", reference,
                    "--mode", "static_only",
                ],
                output / "05_graph_initialization/static_only",
                ("ap02_build_graph",),
            ),
            (
                "ap02_static_ba",
                "AP02: static-only bundle adjustment (diagnostic)",
                [
                    *module,
                    "camera_rig_calibration.methods.ap02.optimize_stage",
                    "--out", str(output),
                    "--ref-marker-id", reference,
                    "--mode", "static_only",
                    "--max-nfev", str(
                        settings.static_only_ba_max_function_evaluations
                    ),
                    "--robust-loss", settings.ba_robust_loss,
                    "--robust-loss-scale-px", str(
                        settings.ba_robust_loss_scale_px
                    ),
                ],
                output / "07_graph_ba/static_only",
                ("ap02_static_initialization",),
            ),
            (
                "ap02_combined_initialization",
                "AP02: combined initialization (primary)",
                [
                    *module,
                    "camera_rig_calibration.methods.ap02.initialize_stage",
                    "--out", str(output),
                    "--ref-marker-id", reference,
                    "--mode", "with_moving",
                ],
                output / "05_graph_initialization/with_moving",
                ("ap02_build_graph",),
            ),
            (
                "ap02_combined_ba",
                "AP02: combined bundle adjustment (primary)",
                [
                    *module,
                    "camera_rig_calibration.methods.ap02.optimize_stage",
                    "--out", str(output),
                    "--ref-marker-id", reference,
                    "--mode", "with_moving",
                    "--max-nfev", str(
                        settings.combined_ba_max_function_evaluations
                    ),
                    "--robust-loss", settings.ba_robust_loss,
                    "--robust-loss-scale-px", str(
                        settings.ba_robust_loss_scale_px
                    ),
                ],
                output / "07_graph_ba/with_moving",
                ("ap02_combined_initialization",),
            ),
            (
                "ap02_report",
                "AP02: write diagnostic and primary report",
                [
                    *module,
                    "camera_rig_calibration.methods.ap02.report",
                    "--out", str(output),
                    "--cameras", cameras,
                    "--ref-marker-id", reference,
                ],
                output / "08_final_results",
                ("ap02_static_ba", "ap02_combined_ba"),
            ),
        ]
        return tuple(
            CommandSpec(
                stage_id,
                display_name,
                tuple(argv),
                context.repository_root,
                directory,
                depends_on=dependencies,
                diagnostic=stage_id
                in {"ap02_static_initialization", "ap02_static_ba"},
            )
            for stage_id, display_name, argv, directory, dependencies in stages
        )

    def collect(self, context: RunContext) -> dict[str, Any]:
        return _status_payload(context.run_directory / "03_AP02")


@dataclass(frozen=True)
class AP03Method:
    id: str = "ap03"
    display_name: str = "AP03 — shared COLMAP with single and multi scale"
    config_model: type[BaseModel] = AP03Settings

    def requirements(self, context: RunContext) -> RequirementResult:
        result = _base_requirements(context)
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
        module = [sys.executable, "-m"]
        prepare = [
            *module,
            "camera_rig_calibration.methods.ap03.prepare_colmap",
            "--dataset", str(context.dataset_root),
            "--out", str(output),
            "--cameras", cameras,
            "--moving-camera-id", config.moving_camera.id,
        ]
        reconstruct = [
            *module,
            "camera_rig_calibration.methods.ap03.reconstruct_stage",
            "--dataset", str(context.dataset_root),
            "--out", str(output),
            "--cameras", cameras,
            "--moving-camera-id", config.moving_camera.id,
            "--matcher", config.colmap.matcher,
            "--use-gpu", "1" if config.colmap.use_gpu else "0",
            "--max-image-size", str(
                config.colmap.ap03_maximum_image_size
                or config.colmap.maximum_image_size
            ),
            "--max-features", str(
                config.colmap.ap03_maximum_features
                or config.colmap.maximum_features
            ),
            "--sequential-overlap", str(config.colmap.sequential_overlap),
            "--loop-detection", (
                "1"
                if (
                    config.colmap.ap03_loop_detection
                    if config.colmap.ap03_loop_detection is not None
                    else config.colmap.loop_detection
                )
                else "0"
            ),
            "--mapper-min-matches", str(
                config.colmap.mapper_minimum_matches
            ),
            "--colmap-executable", config.colmap.executable,
        ]
        if config.colmap.reuse or context.reuse_colmap_artifact:
            reconstruct.append("--reuse-colmap")
        scale_common = [
            "--repository-root", str(context.repository_root),
            "--observations-root", str(context.observations_root),
            "--out", str(output),
            "--cameras", cameras,
            "--marker-length-m", str(config.markers.length_m),
            "--reprojection-threshold-px", str(
                scale.reprojection_threshold_px
            ),
            "--ransac-iterations", str(scale.ransac_iterations),
            "--minimum-inliers", str(scale.minimum_inliers),
            "--dictionary", config.markers.dictionary,
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
                    *module,
                    "camera_rig_calibration.methods.ap03.inspect_stage",
                    "--out", str(output),
                    "--cameras", cameras,
                ],
                output / "colmap/inspection",
                ("ap03_reconstruct",),
            ),
            (
                "ap03_single_scale",
                "AP03: single-marker scale (diagnostic)",
                [
                    *module,
                    "camera_rig_calibration.methods.ap03.estimate_scale",
                    *scale_common,
                    "--mode", "single",
                    "--marker-ids", str(single_marker),
                ],
                output / "scale_single",
                ("ap03_inspect",),
            ),
            (
                "ap03_multi_scale",
                "AP03: multi-marker scale (primary)",
                [
                    *module,
                    "camera_rig_calibration.methods.ap03.estimate_scale",
                    *scale_common,
                    "--mode", "multi",
                    "--marker-ids", ",".join(
                        str(value) for value in multi_markers
                    ),
                ],
                output / "scale_multi",
                ("ap03_inspect",),
            ),
            (
                "ap03_report",
                "AP03: write combined report",
                [
                    *module,
                    "camera_rig_calibration.methods.ap03.report",
                    "--out", str(output),
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
        return _status_payload(context.run_directory / "04_AP03")


@dataclass(frozen=True)
class MarkerConsistencyEvaluator:
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
            str(context.repository_root / "run/real_vehicle_data/12_evaluate_real_marker_consistency.py"),
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
        return (CommandSpec("evaluation", self.display_name, tuple(argv), context.repository_root, context.run_directory / "06_EVALUATION"),)


@dataclass(frozen=True)
class ColmapMatcherExperiments:
    id: str = "colmap_matcher"
    display_name: str = "COLMAP matcher comparison"
    description: str = "Create separate exhaustive and sequential matcher runs."

    def variants(self, config: RigConfig) -> Sequence[tuple[str, RigConfig]]:
        variants = []
        for matcher in ("exhaustive", "sequential"):
            variants.append(
                (
                    f"colmap_{matcher}",
                    config.model_copy(
                        update={
                            "project": config.project.model_copy(
                                update={"run_label": f"colmap_{matcher}"}
                            ),
                            "colmap": config.colmap.model_copy(update={"matcher": matcher}),
                        },
                        deep=True,
                    ),
                )
            )
        return variants


def register_builtin_components() -> None:
    components = (
        (input_adapters, PreparedInputAdapter()),
        (input_adapters, SimulationInputAdapter()),
        (input_adapters, FilesystemInputAdapter()),
        (input_adapters, McapInputAdapter()),
        (calibration_methods, AP01Method()),
        (calibration_methods, AP02Method()),
        (calibration_methods, AP03Method()),
        (evaluators, MarkerConsistencyEvaluator()),
        (experiment_providers, ColmapMatcherExperiments()),
    )
    for registry, component in components:
        if component.id not in registry:
            registry.register(component)
