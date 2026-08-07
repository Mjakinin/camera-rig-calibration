from __future__ import annotations

import re
from enum import Enum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


class StrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid", validate_assignment=True, populate_by_name=True
    )


class SceneType(str, Enum):
    INTERIOR = "interior"
    EXTERIOR = "exterior"
    SIMULATION = "simulation"
    OTHER = "other"


class DatasetCategory(str, Enum):
    REAL_VEHICLE = "real_vehicle"
    SIMULATION = "simulation"


class InputSourceKind(str, Enum):
    VIDEO = "video"
    FRAMES = "frames"
    ROSBAG = "rosbag"
    PREPARED = "prepared"


class DatasetSettings(StrictModel):
    id: str
    category: DatasetCategory = DatasetCategory.REAL_VEHICLE
    source_kind: InputSourceKind = InputSourceKind.PREPARED
    scene_type: SceneType = SceneType.OTHER
    description: str = ""
    prepared_root: Path | None = None
    input_root: Path | None = None

    @field_validator("id")
    @classmethod
    def validate_id(cls, value: str) -> str:
        value = value.strip()
        if not ID_PATTERN.fullmatch(value):
            raise ValueError(
                "must start with a letter or digit and contain only letters, "
                "digits, '_', '-' or '.'"
            )
        return value


class StaticCameraSettings(StrictModel):
    id: str
    label: str | None = None
    images: list[Path] = Field(default_factory=list)
    video: Path | None = None
    intrinsics: Path | None = None
    required: bool = True
    image_topic: str | None = None
    camera_info_topic: str | None = None

    @field_validator("id")
    @classmethod
    def validate_id(cls, value: str) -> str:
        if not ID_PATTERN.fullmatch(value.strip()):
            raise ValueError("invalid camera ID")
        return value.strip()

    @model_validator(mode="after")
    def validate_media_source(self) -> "StaticCameraSettings":
        if self.images and self.video is not None:
            raise ValueError(
                "static camera must use either image files or one video, not both"
            )
        return self


class IntrinsicScanSettings(StrictModel):
    mode: Literal["balanced", "full_frame"] = "balanced"
    target_hz: float = Field(default=3.0, gt=0)
    preview_max_dimension: int = Field(default=1920, ge=320)


class MovingCameraSettings(StrictModel):
    id: str = "moving_calib_camera"
    video: Path | None = None
    frames: Path | None = None
    intrinsics: Path | None = None
    intrinsics_profile: str | None = None
    intrinsic_calibration_video: Path | None = None
    intrinsic_calibration_images: Path | None = None
    intrinsic_scan: IntrinsicScanSettings = Field(
        default_factory=IntrinsicScanSettings
    )
    checkerboard_columns: int = Field(default=8, ge=2)
    checkerboard_rows: int = Field(default=6, ge=2)
    intrinsic_maximum_views: int = Field(default=80, ge=10)
    intrinsic_minimum_frame_gap: int = Field(default=5, ge=0)
    intrinsic_minimum_detections: int = Field(default=20, ge=8)
    image_topic: str | None = None
    camera_info_topic: str | None = None

    @field_validator("intrinsics_profile")
    @classmethod
    def validate_intrinsics_profile(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        if not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9_.-]*(?:@[a-fA-F0-9]{8,64})?",
            value,
        ):
            raise ValueError(
                "must be '<profile-id>' or '<profile-id>@<content-hash>'"
            )
        return value

    @model_validator(mode="after")
    def validate_sources(self) -> "MovingCameraSettings":
        if self.video is not None and self.frames is not None:
            raise ValueError("moving camera must use either video or frames, not both")
        calibration_sources = [
            source
            for source in (
                self.intrinsic_calibration_video,
                self.intrinsic_calibration_images,
            )
            if source is not None
        ]
        if len(calibration_sources) > 1:
            raise ValueError(
                "use either an intrinsic-calibration video or image folder, not both"
            )
        if self.intrinsics is not None and calibration_sources:
            raise ValueError(
                "use either existing intrinsics or an intrinsic-calibration source"
            )
        return self


class McapSettings(StrictModel):
    path: Path | None = None
    save_all_candidates: bool = False


class SimulationSettings(StrictModel):
    enabled: bool = False
    preset: str = "custom"
    world_id: str = "bus"
    world_baseline: dict[str, Any] = Field(default_factory=dict)
    capture_id: str | None = None
    world: Path | None = None
    route: Path | None = None
    resource_paths: list[Path] = Field(default_factory=list)
    moving_model_name: str = "moving_calib_camera"
    moving_sensor_name: str | None = None
    settle_seconds: float = Field(default=0.35, ge=0)
    post_pose_skip: int = Field(default=5, ge=0)
    frame_timeout_seconds: float = Field(default=3.0, gt=0)
    startup_timeout_seconds: float = Field(default=60.0, gt=0)
    route_name: str = "route2"
    moving_width: int = Field(default=1280, ge=64)
    moving_height: int = Field(default=720, ge=64)
    moving_hfov_deg: float = Field(default=69.1, gt=1.0, lt=179.0)
    lighting: Literal[
        "baseline", "dark_extreme", "low", "normal", "bright", "custom"
    ] = "baseline"
    lighting_scale: float = Field(default=1.0, gt=0.0, le=10.0)
    motion_blur_kernel: int = Field(default=0, ge=0, le=999)
    motion_blur_angle_deg: float = Field(default=0.0, ge=-180.0, le=180.0)
    target_route_frames: int | None = Field(default=None, ge=2)
    route_sampling_strategy: str = "original_route_poses"

    @field_validator(
        "preset",
        "world_id",
        "moving_model_name",
        "route_sampling_strategy",
    )
    @classmethod
    def validate_names(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be empty")
        return value

    @field_validator("world_id")
    @classmethod
    def validate_bus_world(cls, value: str) -> str:
        if value != "bus":
            raise ValueError(
                "only the built-in bus Gazebo world is supported"
            )
        return value

    @field_validator("moving_sensor_name")
    @classmethod
    def validate_optional_sensor_name(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        if not value:
            raise ValueError("must not be empty")
        return value

    @field_validator("capture_id")
    @classmethod
    def validate_capture_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        if not ID_PATTERN.fullmatch(value):
            raise ValueError(
                "must start with a letter or digit and contain only letters, "
                "digits, '_', '-' or '.'"
            )
        return value

    @field_validator("motion_blur_kernel")
    @classmethod
    def validate_blur_kernel(cls, value: int) -> int:
        if value != 0 and value % 2 == 0:
            raise ValueError("motion_blur_kernel must be zero or an odd integer")
        return value


class SamplingSettings(StrictModel):
    target_hz: float | None = Field(default=None, gt=0)
    image_format: Literal["png"] = "png"
    start_seconds: float = Field(default=0.0, ge=0)
    end_seconds: float | None = Field(default=None, gt=0)
    maximum_frames: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def validate_range(self) -> "SamplingSettings":
        if self.end_seconds is not None and self.end_seconds <= self.start_seconds:
            raise ValueError("end_seconds must be greater than start_seconds")
        return self


class MarkerSettings(StrictModel):
    dictionary: str = "DICT_4X4_50"
    length_m: float = Field(default=0.17, gt=0)
    accepted_ids: list[int] | Literal["all_detected"] = "all_detected"
    detection_mode: Literal[
        "baseline", "subpixel_refined", "high_sensitivity"
    ] = "baseline"

class SelectionSettings(StrictModel):
    mode: Literal["review_once", "auto", "explicit"] = "auto"


class ColmapSettings(StrictModel):
    executable: str = "auto"
    matcher: Literal["exhaustive", "sequential"] = "exhaustive"
    compute_mode: Literal["cpu_baseline", "gpu", "auto"] = "cpu_baseline"
    maximum_image_size: int = Field(default=1600, ge=256)
    maximum_features: int = Field(default=4096, ge=128)
    sequential_overlap: int = Field(default=20, ge=1)
    loop_detection: bool = True
    mapper_minimum_matches: int = Field(default=8, ge=1)
    reuse: bool = False
    ap03_maximum_image_size: int | None = Field(default=2400, ge=256)
    ap03_maximum_features: int | None = Field(default=8192, ge=128)
    ap03_loop_detection: bool | None = None

    @model_validator(mode="before")
    @classmethod
    def migrate_legacy_gpu_mode(cls, value: Any) -> Any:
        """Accept direct construction with the former schema-v5 GPU field."""
        if not isinstance(value, dict) or "gpu_mode" not in value:
            return value
        migrated = dict(value)
        legacy = str(migrated.pop("gpu_mode")).strip().lower()
        translated = {
            "false": "cpu_baseline",
            "true": "gpu",
            "auto": "auto",
        }.get(legacy)
        if translated is None:
            raise ValueError(
                "gpu_mode must be one of false, true or auto"
            )
        configured = migrated.get("compute_mode")
        if configured is not None and configured != translated:
            raise ValueError(
                "compute_mode conflicts with the deprecated gpu_mode value"
            )
        migrated["compute_mode"] = translated
        return migrated

    @property
    def use_gpu(self) -> bool:
        """Resolved boolean consumed by the command adapters."""
        return self.compute_mode == "gpu"


class ObservationQualitySettings(StrictModel):
    minimum_marker_area_ratio: float = Field(
        default=0.000008, ge=0.0, le=1.0
    )
    maximum_pnp_reprojection_error_px: float | Literal["disabled"] = 25.0
    require_positive_depth: bool = True
    maximum_marker_distance_m: float | Literal["disabled"] = "disabled"

    @field_validator(
        "maximum_pnp_reprojection_error_px",
        "maximum_marker_distance_m",
    )
    @classmethod
    def validate_optional_positive_limit(
        cls, value: float | Literal["disabled"]
    ) -> float | Literal["disabled"]:
        if value == "disabled":
            return value
        number = float(value)
        if number <= 0:
            raise ValueError("must be 'disabled' or greater than zero")
        return number


class ObservationQualityOverrides(StrictModel):
    """Per-method overrides; ``None`` means inherit the global baseline."""

    minimum_marker_area_ratio: float | None = Field(
        default=None, ge=0.0, le=1.0
    )
    maximum_pnp_reprojection_error_px: (
        float | Literal["disabled"] | None
    ) = None
    require_positive_depth: bool | None = None
    maximum_marker_distance_m: float | Literal["disabled"] | None = None

    @field_validator(
        "maximum_pnp_reprojection_error_px",
        "maximum_marker_distance_m",
    )
    @classmethod
    def validate_optional_positive_override(
        cls, value: float | Literal["disabled"] | None
    ) -> float | Literal["disabled"] | None:
        if value is None or value == "disabled":
            return value
        number = float(value)
        if number <= 0:
            raise ValueError(
                "must be null, 'disabled', or greater than zero"
            )
        return number


class AP01PathQualityGate(StrictModel):
    minimum_inlier_ratio: float = Field(default=0.70, ge=0.0, le=1.0)
    maximum_translation_dispersion_m: float = Field(default=0.12, gt=0)
    maximum_rotation_dispersion_deg: float = Field(default=4.0, gt=0)


class AP01DirectQualityGate(AP01PathQualityGate):
    minimum_independent_markers: int = Field(default=3, ge=1)


class AP01RelayQualityGate(AP01PathQualityGate):
    maximum_translation_dispersion_m: float = Field(default=0.30, gt=0)
    maximum_rotation_dispersion_deg: float = Field(default=7.0, gt=0)


class AP01PathConsistency(StrictModel):
    maximum_translation_disagreement_m: float = Field(default=0.12, gt=0)
    maximum_rotation_disagreement_deg: float = Field(default=4.0, gt=0)


class AP01Settings(StrictModel):
    method_contract: Literal[
        "baseline_v1", "main_route2_parity_v1", "recommended_wizard_v1"
    ] = "baseline_v1"
    historical_reproduction: bool = False
    advanced_strategy: Literal[
        "legacy_main_v1", "wizard_robustness_v1"
    ] = "legacy_main_v1"
    direct_target_camera: str = "cam_edge_1"
    root_camera: str = "auto"
    top_moving_per_marker: int | None = Field(default=8, ge=1)
    scale_top_per_marker: int | None = Field(default=30, ge=1)
    direct_quality_gate: AP01DirectQualityGate = Field(
        default_factory=AP01DirectQualityGate
    )
    relay_quality_gate: AP01RelayQualityGate = Field(
        default_factory=AP01RelayQualityGate
    )
    direct_relay_consistency: AP01PathConsistency = Field(
        default_factory=AP01PathConsistency
    )
    observation_quality: ObservationQualityOverrides = Field(
        default_factory=ObservationQualityOverrides
    )

    @field_validator("direct_target_camera")
    @classmethod
    def validate_direct_target_camera(cls, value: str) -> str:
        value = value.strip()
        if not ID_PATTERN.fullmatch(value):
            raise ValueError("invalid AP01 direct target camera ID")
        return value


class AP02Settings(StrictModel):
    method_contract: Literal["baseline_v1"] = "baseline_v1"
    historical_reproduction: bool = False
    reference_marker_selection_mode: Literal[
        "baseline", "auto", "manual", "explicit"
    ] = "baseline"
    reference_marker_id: int | Literal["auto"] = 14
    frame_selection_strategy: Literal[
        "legacy_smart_v1", "wizard_graph_preserving_v1"
    ] = "legacy_smart_v1"
    initialization_strategy: Literal[
        "legacy_maximum_bottleneck_v1",
        "wizard_maximum_bottleneck_v2",
        "unweighted_bfs_diagnostic",
    ] = "legacy_maximum_bottleneck_v1"
    graph_edge_weight_strategy: Literal[
        "legacy_observation_quality_v1", "wizard_selection_score_v2"
    ] = "legacy_observation_quality_v1"
    reprojection_model: Literal[
        "legacy_pinhole_v1", "distortion_aware_v1"
    ] = "legacy_pinhole_v1"
    reference_marker_maximum_frames: int | None = Field(
        default=None, ge=1
    )
    top_per_marker: int | None = Field(default=8, ge=1)
    top_per_marker_pair: int | None = Field(default=4, ge=1)
    maximum_total_frames: int | None = Field(default=None, ge=1)
    static_only_ba_max_function_evaluations: int = Field(default=80, ge=1)
    combined_ba_max_function_evaluations: int = Field(default=80, ge=1)
    ba_robust_loss: Literal["soft_l1", "huber", "linear"] = "soft_l1"
    ba_robust_loss_scale_px: float = Field(default=3.0, gt=0)
    observation_quality: ObservationQualityOverrides = Field(
        default_factory=ObservationQualityOverrides
    )

    @model_validator(mode="before")
    @classmethod
    def infer_legacy_reference_mode(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        migrated = dict(value)
        if "reference_marker_selection_mode" not in migrated:
            if "reference_marker_id" not in migrated:
                return migrated
            configured = migrated.get("reference_marker_id", "auto")
            migrated["reference_marker_selection_mode"] = (
                "auto" if configured == "auto" else "explicit"
            )
        if (
            migrated.get("reference_marker_selection_mode") == "manual"
            and "reference_marker_id" not in migrated
        ):
            migrated["reference_marker_id"] = "auto"
        return migrated

    @model_validator(mode="after")
    def validate_reference_selection(self) -> "AP02Settings":
        if (
            self.reference_marker_selection_mode == "baseline"
            and self.reference_marker_id != 14
        ):
            raise ValueError(
                "baseline reference-marker selection requires marker 14"
            )
        if (
            self.reference_marker_selection_mode == "explicit"
            and self.reference_marker_id == "auto"
        ):
            raise ValueError(
                "explicit reference-marker selection requires a marker ID"
            )
        return self

    @property
    def max_nfev_static(self) -> int:
        return self.static_only_ba_max_function_evaluations

    @property
    def max_nfev_moving(self) -> int:
        return self.combined_ba_max_function_evaluations


class AP03ScaleSettings(StrictModel):
    reprojection_threshold_px: float = Field(default=5.0, gt=0)
    ransac_iterations: int = Field(default=1000, ge=1)
    minimum_inliers: int = Field(default=4, ge=2)
    maximum_observations_per_marker: int | None = Field(
        default=None, ge=1
    )


class AP03SingleSettings(StrictModel):
    scale_marker_id: int | Literal["auto"] = 14


class AP03MultiSettings(StrictModel):
    marker_ids: list[int] | Literal["auto"] = Field(
        default_factory=lambda: list(range(15))
    )


class AP03Settings(StrictModel):
    method_contract: Literal["baseline_v1"] = "baseline_v1"
    feature_limit_policy: Literal[
        "legacy_colmap_defaults_v1", "wizard_explicit_limits_v1"
    ] = "legacy_colmap_defaults_v1"
    scale_input_policy: Literal[
        "legacy_registered_image_redetection_v1",
        "wizard_filtered_observations_v1",
    ] = "legacy_registered_image_redetection_v1"
    minimum_marker_area_px2: float = Field(default=100.0, ge=0.0)
    single: AP03SingleSettings = Field(default_factory=AP03SingleSettings)
    multi: AP03MultiSettings = Field(default_factory=AP03MultiSettings)
    scale: AP03ScaleSettings = Field(default_factory=AP03ScaleSettings)
    observation_quality: ObservationQualityOverrides = Field(
        default_factory=ObservationQualityOverrides
    )


class MethodSettings(StrictModel):
    enabled: list[str] = Field(default_factory=lambda: ["ap01", "ap02", "ap03"])
    ap01: AP01Settings = Field(default_factory=AP01Settings)
    ap02: AP02Settings = Field(default_factory=AP02Settings)
    ap03: AP03Settings = Field(default_factory=AP03Settings)
    extensions: dict[str, dict[str, Any]] = Field(default_factory=dict)

    @property
    def ap03_single(self) -> AP03SingleSettings:
        return self.ap03.single

    @property
    def ap03_multi(self) -> AP03MultiSettings:
        return self.ap03.multi

    @field_validator("enabled")
    @classmethod
    def validate_methods(cls, values: list[str]) -> list[str]:
        result: list[str] = []
        for value in values:
            normalized = value.strip().lower().replace("-", "_")
            if not ID_PATTERN.fullmatch(normalized):
                raise ValueError(f"invalid method ID: {value}")
            if normalized not in result:
                result.append(normalized)
        if not result:
            raise ValueError("at least one calibration method must be enabled")
        return result


class EvaluationSettings(StrictModel):
    enabled: bool = True
    anchor_marker_id: int | Literal["auto"] = "auto"
    anchor_selection_mode: Literal["auto", "review_once", "explicit"] = "auto"
    reprojection_threshold_px: float = Field(default=5.0, gt=0)
    minimum_inliers: int = Field(default=4, ge=2)
    ransac_iterations: int = Field(default=800, ge=1)
    minimum_triangulation_angle_deg: float = Field(default=0.5, gt=0)
    maximum_moving_observations_per_marker: int = Field(default=80, ge=2)


class ReportingSettings(StrictModel):
    terminal_verbosity: Literal["quiet", "normal", "verbose"] = "normal"
    csv: bool = True
    json_enabled: bool = Field(default=True, alias="json", serialization_alias="json")
    text: bool = True


class ProjectSettings(StrictModel):
    workspace_root: Path = Path("workspace")
    dataset_cache_root: Path = Path("workspace/preparation_cache")
    output_root: Path = Path("results")
    experiment_id: str | None = None
    run_label: str = "baseline"
    execution_mode: Literal["complete", "prepare_only"] = "complete"
    duplicate_policy: Literal["skip", "force", "error"] = "skip"

    @field_validator("run_label", "experiment_id")
    @classmethod
    def validate_run_label(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        if not ID_PATTERN.fullmatch(value):
            raise ValueError("must be safe for use in a directory name")
        return value


class RigConfig(StrictModel):
    schema_version: Literal[5] = 5
    project: ProjectSettings = Field(default_factory=ProjectSettings)
    dataset: DatasetSettings
    static_cameras: list[StaticCameraSettings] = Field(default_factory=list)
    moving_camera: MovingCameraSettings = Field(default_factory=MovingCameraSettings)
    mcap: McapSettings = Field(default_factory=McapSettings)
    simulation: SimulationSettings = Field(default_factory=SimulationSettings)
    sampling: SamplingSettings = Field(default_factory=SamplingSettings)
    markers: MarkerSettings = Field(default_factory=MarkerSettings)
    selection: SelectionSettings = Field(default_factory=SelectionSettings)
    colmap: ColmapSettings = Field(default_factory=ColmapSettings)
    methods: MethodSettings = Field(default_factory=MethodSettings)
    observation_quality: ObservationQualitySettings = Field(
        default_factory=ObservationQualitySettings
    )
    evaluation: EvaluationSettings = Field(default_factory=EvaluationSettings)
    reporting: ReportingSettings = Field(default_factory=ReportingSettings)

    @model_validator(mode="before")
    @classmethod
    def require_schema_v5(cls, value: Any) -> Any:
        """Reject legacy configuration contracts instead of guessing a migration."""
        if not isinstance(value, dict):
            return value
        version = value.get("schema_version")
        if version is None:
            # Direct Python construction uses the model default. File loading
            # validates the explicit version before model construction.
            return value
        if version != 5:
            raise ValueError(
                "Only schema_version 5 is supported. Recreate the configuration "
                "with the current rigcal wizard."
            )
        return value

    @model_validator(mode="after")
    def validate_dataset_contract(self) -> "RigConfig":
        if not self.static_cameras:
            raise ValueError("at least one static camera must be declared")
        ids = [camera.id for camera in self.static_cameras]
        duplicates = sorted({camera_id for camera_id in ids if ids.count(camera_id) > 1})
        if duplicates:
            raise ValueError(f"duplicate static camera IDs: {duplicates}")
        if self.moving_camera.id in ids:
            raise ValueError("moving camera ID must differ from every static camera ID")
        if self.simulation.enabled:
            if self.dataset.category != DatasetCategory.SIMULATION:
                raise ValueError(
                    "simulation.enabled requires dataset.category=simulation"
                )
            if self.dataset.prepared_root is not None or self.mcap.path is not None:
                raise ValueError(
                    "simulation capture cannot be combined with prepared_root or MCAP"
                )
            if self.simulation.world is None or self.simulation.route is None:
                raise ValueError("simulation capture requires world and route files")
            if any(
                not camera.image_topic or not camera.camera_info_topic
                for camera in self.static_cameras
            ):
                raise ValueError(
                    "simulation static cameras require image_topic and camera_info_topic"
                )
            if (
                not self.moving_camera.image_topic
                or not self.moving_camera.camera_info_topic
            ):
                raise ValueError(
                    "simulation moving camera requires image_topic and camera_info_topic"
                )
        elif self.dataset.prepared_root is None:
            has_moving = (
                self.moving_camera.video is not None
                or self.moving_camera.frames is not None
                or (
                    self.mcap.path is not None
                    and self.moving_camera.image_topic is not None
                )
            )
            if not has_moving:
                raise ValueError(
                    "provide dataset.prepared_root or a moving video/frames source"
                )
            if not self.static_cameras and self.mcap.path is None:
                raise ValueError("provide static cameras or an MCAP source")
        if self.moving_camera.video is not None and self.sampling.target_hz is None:
            raise ValueError("sampling.target_hz is required for moving-video input")
        return self


def effective_observation_quality(
    config: RigConfig,
    method_id: str,
) -> tuple[ObservationQualitySettings, dict[str, Literal["global", "method_override"]]]:
    """Resolve one method's quality settings and record each value's origin."""

    normalized = method_id.strip().lower().replace("-", "_")
    baseline = config.observation_quality.model_dump(mode="python")
    sources: dict[str, Literal["global", "method_override"]] = {
        field_name: "global" for field_name in baseline
    }
    method_settings = getattr(config.methods, normalized, None)
    overrides = getattr(method_settings, "observation_quality", None)
    if overrides is not None:
        for field_name, value in overrides.model_dump(mode="python").items():
            if value is None:
                continue
            baseline[field_name] = value
            sources[field_name] = "method_override"
    return ObservationQualitySettings.model_validate(baseline), sources
