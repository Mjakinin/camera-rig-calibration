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

    @model_validator(mode="before")
    @classmethod
    def infer_category_from_legacy_scene_type(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        payload = dict(value)
        if (
            "category" not in payload
            and payload.get("scene_type") in {SceneType.SIMULATION, "simulation"}
        ):
            payload["category"] = DatasetCategory.SIMULATION
        return payload

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
    mode: Literal["balanced", "exhaustive_compatibility"] = "balanced"
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

    @property
    def allowed_ids(self) -> list[int] | Literal["auto"]:
        """Read-only compatibility alias for schema-v1/v2 callers."""
        return "auto" if self.accepted_ids == "all_detected" else self.accepted_ids


class ReferenceSettings(StrictModel):
    """Legacy schema-v1 reference bundle.

    New configurations keep each scientific choice with the component that
    consumes it.  This model remains importable so old Python callers and YAML
    files can be migrated without guessing.
    """

    root_camera: str = "auto"
    ap02_pose_reference_marker_id: int | Literal["auto", "sweep"] = "auto"
    evaluation_scale_anchor_marker_id: int | Literal["auto", "sweep"] = "auto"


class SelectionSettings(StrictModel):
    mode: Literal["review_once", "auto", "explicit"] = "review_once"


class ColmapSettings(StrictModel):
    executable: str = "auto"
    matcher: Literal["exhaustive", "sequential"] = "exhaustive"
    gpu_mode: Literal["auto", "true", "false"] = "auto"
    maximum_image_size: int = Field(default=2400, ge=256)
    maximum_features: int = Field(default=8192, ge=128)
    sequential_overlap: int = Field(default=20, ge=1)
    loop_detection: bool = True
    mapper_minimum_matches: int = Field(default=8, ge=1)
    reuse: bool = False
    ap03_maximum_image_size: int | None = Field(default=None, ge=256)
    ap03_maximum_features: int | None = Field(default=None, ge=128)
    ap03_loop_detection: bool | None = None

    @model_validator(mode="before")
    @classmethod
    def migrate_gpu_flag(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        payload = dict(value)
        old = payload.pop("use_gpu", None)
        if "gpu_mode" not in payload and old is not None:
            payload["gpu_mode"] = "true" if bool(old) else "false"
        return payload

    @property
    def use_gpu(self) -> bool:
        """Compatibility value for legacy command adapters.

        Runtime preflight resolves ``auto`` before a command is executed.
        """
        return self.gpu_mode == "true"


class AP01Settings(StrictModel):
    root_camera: str = "auto"


class AP02Settings(StrictModel):
    reference_marker_id: int | Literal["auto"] = "auto"
    static_only_ba_max_function_evaluations: int = Field(default=100, ge=1)
    combined_ba_max_function_evaluations: int = Field(default=120, ge=1)
    ba_robust_loss: Literal["soft_l1", "huber", "linear"] = "soft_l1"
    ba_robust_loss_scale_px: float = Field(default=3.0, gt=0)

    @model_validator(mode="before")
    @classmethod
    def migrate_evaluation_limits(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        payload = dict(value)
        old_static = payload.pop("max_nfev_static", None)
        old_combined = payload.pop("max_nfev_moving", None)
        if (
            "static_only_ba_max_function_evaluations" not in payload
            and old_static is not None
        ):
            payload["static_only_ba_max_function_evaluations"] = old_static
        if (
            "combined_ba_max_function_evaluations" not in payload
            and old_combined is not None
        ):
            payload["combined_ba_max_function_evaluations"] = old_combined
        return payload

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


class AP03SingleSettings(StrictModel):
    scale_marker_id: int | Literal["auto"] = "auto"


class AP03MultiSettings(StrictModel):
    marker_ids: list[int] | Literal["auto"] = "auto"


class AP03Settings(StrictModel):
    single: AP03SingleSettings = Field(default_factory=AP03SingleSettings)
    multi: AP03MultiSettings = Field(default_factory=AP03MultiSettings)
    scale: AP03ScaleSettings = Field(default_factory=AP03ScaleSettings)


class MethodSettings(StrictModel):
    enabled: list[str] = Field(default_factory=lambda: ["ap01", "ap02", "ap03"])
    ap01: AP01Settings = Field(default_factory=AP01Settings)
    ap02: AP02Settings = Field(default_factory=AP02Settings)
    ap03: AP03Settings = Field(default_factory=AP03Settings)
    extensions: dict[str, dict[str, Any]] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def migrate_split_ap03(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        payload = dict(value)
        enabled = [
            str(item).strip().lower().replace("-", "_")
            for item in payload.get("enabled", ["ap01", "ap02", "ap03"])
        ]
        if "ap03_single" in enabled or "ap03_multi" in enabled:
            enabled = [
                item
                for item in enabled
                if item not in {"ap03_single", "ap03_multi"}
            ]
            enabled.append("ap03")
        payload["enabled"] = list(dict.fromkeys(enabled))

        ap03_value = payload.get("ap03", {})
        ap03 = (
            ap03_value.model_dump(mode="python")
            if isinstance(ap03_value, BaseModel)
            else dict(ap03_value or {})
        )
        single = payload.pop("ap03_single", None)
        multi = payload.pop("ap03_multi", None)
        if single is not None and "single" not in ap03:
            ap03["single"] = (
                single.model_dump(mode="python")
                if isinstance(single, BaseModel)
                else single
            )
        if multi is not None and "multi" not in ap03:
            ap03["multi"] = (
                multi.model_dump(mode="python")
                if isinstance(multi, BaseModel)
                else multi
            )
        payload["ap03"] = ap03
        return payload

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


class ObservationQualitySettings(StrictModel):
    maximum_pnp_reprojection_error_px: float | Literal["disabled"] = 25.0
    minimum_marker_area_px2: float = Field(default=0.0, ge=0)
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


class EvaluationSettings(StrictModel):
    enabled: bool = True
    anchor_marker_id: int | Literal["auto_common"] = "auto_common"
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
    dataset_cache_root: Path = Path("datasets")
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
    def migrate_legacy_schemas(cls, value: Any) -> Any:
        """Read schema v1-v4 while emitting only the schema-v5 contract."""
        if not isinstance(value, dict):
            return value
        payload = dict(value)
        source_version = int(payload.get("schema_version", 1))
        if source_version not in {1, 2, 3, 4, 5}:
            return payload
        if source_version in {4, 5}:
            # Schema v4 is deliberately strict.  Removed experiment switches
            # must not be silently accepted by one of the legacy migration
            # aliases below.
            methods = payload.get("methods", {})
            ap02 = methods.get("ap02", {}) if isinstance(methods, dict) else {}
            removed_v4_fields = {
                "frame_selection": payload.get("frame_selection"),
                "methods.ap03_single": (
                    methods.get("ap03_single")
                    if isinstance(methods, dict)
                    else None
                ),
                "methods.ap03_multi": (
                    methods.get("ap03_multi")
                    if isinstance(methods, dict)
                    else None
                ),
                "methods.ap02.max_nfev_static": (
                    ap02.get("max_nfev_static")
                    if isinstance(ap02, dict)
                    else None
                ),
                "methods.ap02.max_nfev_moving": (
                    ap02.get("max_nfev_moving")
                    if isinstance(ap02, dict)
                    else None
                ),
            }
            present = [
                name
                for name, item in removed_v4_fields.items()
                if item is not None
            ]
            if present:
                raise ValueError(
                    "schema v4/v5 does not support removed fields: "
                    + ", ".join(present)
                )
            dataset_value = payload.get("dataset", {})
            dataset = (
                dataset_value.model_dump(mode="python")
                if isinstance(dataset_value, BaseModel)
                else dict(dataset_value or {})
            )
            simulation_value = payload.get("simulation", {})
            simulation = (
                simulation_value.model_dump(mode="python")
                if isinstance(simulation_value, BaseModel)
                else dict(simulation_value or {})
            )
            moving_value = payload.get("moving_camera", {})
            moving = (
                moving_value.model_dump(mode="python")
                if isinstance(moving_value, BaseModel)
                else dict(moving_value or {})
            )
            mcap_value = payload.get("mcap", {})
            mcap = (
                mcap_value.model_dump(mode="python")
                if isinstance(mcap_value, BaseModel)
                else dict(mcap_value or {})
            )
            is_simulation = bool(
                simulation.get("enabled")
                or dataset.get("scene_type") == "simulation"
            )
            dataset.setdefault(
                "category",
                "simulation" if is_simulation else "real_vehicle",
            )
            if "source_kind" not in dataset:
                if moving.get("video"):
                    dataset["source_kind"] = "video"
                elif moving.get("frames"):
                    dataset["source_kind"] = "frames"
                elif mcap.get("path") and moving.get("image_topic"):
                    dataset["source_kind"] = "rosbag"
                else:
                    dataset["source_kind"] = "prepared"
            simulation.setdefault("world_id", "bus")
            simulation.setdefault("world_baseline", {})
            payload["dataset"] = dataset
            payload["simulation"] = simulation
            payload["schema_version"] = 5
            return payload

        payload["schema_version"] = 5
        payload.pop("frame_selection", None)

        markers_value = payload.get("markers", {})
        markers = (
            markers_value.model_dump(mode="python")
            if isinstance(markers_value, BaseModel)
            else dict(markers_value or {})
        )
        old_allowed = markers.pop("allowed_ids", None)
        if "accepted_ids" not in markers and old_allowed is not None:
            markers["accepted_ids"] = (
                "all_detected" if old_allowed == "auto" else old_allowed
            )
        old_minimum_area = markers.pop("minimum_area_px2", None)
        payload["markers"] = markers

        quality_value = payload.get("observation_quality", {})
        quality = (
            quality_value.model_dump(mode="python")
            if isinstance(quality_value, BaseModel)
            else dict(quality_value or {})
        )
        if old_minimum_area is not None:
            quality.setdefault("minimum_marker_area_px2", old_minimum_area)
        payload["observation_quality"] = quality

        methods_value = payload.get("methods", {})
        if isinstance(methods_value, BaseModel):
            methods = methods_value.model_dump(mode="python")
        else:
            methods = dict(methods_value or {})
        evaluation_value = payload.get("evaluation", {})
        if isinstance(evaluation_value, BaseModel):
            evaluation = evaluation_value.model_dump(mode="python")
        else:
            evaluation = dict(evaluation_value or {})

        legacy = payload.pop("references", None)
        if isinstance(legacy, BaseModel):
            legacy = legacy.model_dump(mode="python")
        if isinstance(legacy, dict):
            ap01_value = methods.get("ap01", {})
            ap01 = (
                ap01_value.model_dump(mode="python")
                if isinstance(ap01_value, BaseModel)
                else dict(ap01_value or {})
            )
            ap01.setdefault("root_camera", legacy.get("root_camera", "auto"))
            methods["ap01"] = ap01

            ap02_value = methods.get("ap02", {})
            ap02 = (
                ap02_value.model_dump(mode="python")
                if isinstance(ap02_value, BaseModel)
                else dict(ap02_value or {})
            )
            ap02.setdefault(
                "reference_marker_id",
                legacy.get("ap02_pose_reference_marker_id", "auto"),
            )
            methods["ap02"] = ap02
            legacy_anchor = legacy.get(
                "evaluation_scale_anchor_marker_id", "auto"
            )
            evaluation.setdefault(
                "anchor_marker_id",
                "auto_common" if legacy_anchor == "auto" else legacy_anchor,
            )

        ap01_value = methods.get("ap01", {})
        ap01 = (
            ap01_value.model_dump(mode="python")
            if isinstance(ap01_value, BaseModel)
            else dict(ap01_value or {})
        )
        ap01.pop("top_moving_per_marker", None)
        ap01.pop("scale_top_per_marker", None)
        methods["ap01"] = ap01

        ap02_value = methods.get("ap02", {})
        ap02 = (
            ap02_value.model_dump(mode="python")
            if isinstance(ap02_value, BaseModel)
            else dict(ap02_value or {})
        )
        for removed in (
            "moving_selection",
            "top_per_marker",
            "top_per_pair",
            "max_moving_frames",
        ):
            ap02.pop(removed, None)
        methods["ap02"] = ap02

        ap03_value = methods.get("ap03", {})
        ap03 = (
            ap03_value.model_dump(mode="python")
            if isinstance(ap03_value, BaseModel)
            else dict(ap03_value or {})
        )
        single_value = methods.pop("ap03_single", ap03.get("single", {}))
        single = (
            single_value.model_dump(mode="python")
            if isinstance(single_value, BaseModel)
            else dict(single_value or {})
        )
        if "marker_id" in single and "scale_marker_id" not in single:
            single["scale_marker_id"] = single.pop("marker_id")
        single_scale = {
            key: single.pop(key)
            for key in (
                "reprojection_threshold_px",
                "ransac_iterations",
                "minimum_inliers",
            )
            if key in single
        }
        single.pop("minimum_area_px2", None)
        ap03["single"] = single

        multi_value = methods.pop("ap03_multi", ap03.get("multi", {}))
        multi = (
            multi_value.model_dump(mode="python")
            if isinstance(multi_value, BaseModel)
            else dict(multi_value or {})
        )
        legacy_multi_marker = multi.pop("marker_id", None)
        if "marker_ids" not in multi and legacy_multi_marker not in {
            None,
            "auto",
        }:
            multi["marker_ids"] = [int(legacy_multi_marker)]
        multi_scale = {
            key: multi.pop(key)
            for key in (
                "reprojection_threshold_px",
                "ransac_iterations",
                "minimum_inliers",
            )
            if key in multi
        }
        multi.pop("minimum_area_px2", None)
        ap03["multi"] = multi
        scale_value = ap03.get("scale", {})
        scale = (
            scale_value.model_dump(mode="python")
            if isinstance(scale_value, BaseModel)
            else dict(scale_value or {})
        )
        # AP03 Multi is the primary result, so its legacy values take
        # precedence when old Single/Multi configs disagree.
        for key, item in {**single_scale, **multi_scale}.items():
            scale.setdefault(key, item)
        ap03["scale"] = scale
        methods["ap03"] = ap03
        payload["methods"] = methods
        payload["evaluation"] = evaluation
        dataset_value = payload.get("dataset", {})
        dataset = (
            dataset_value.model_dump(mode="python")
            if isinstance(dataset_value, BaseModel)
            else dict(dataset_value or {})
        )
        simulation_value = payload.get("simulation", {})
        simulation = (
            simulation_value.model_dump(mode="python")
            if isinstance(simulation_value, BaseModel)
            else dict(simulation_value or {})
        )
        moving_value = payload.get("moving_camera", {})
        moving = (
            moving_value.model_dump(mode="python")
            if isinstance(moving_value, BaseModel)
            else dict(moving_value or {})
        )
        mcap_value = payload.get("mcap", {})
        mcap = (
            mcap_value.model_dump(mode="python")
            if isinstance(mcap_value, BaseModel)
            else dict(mcap_value or {})
        )
        is_simulation = bool(
            simulation.get("enabled")
            or dataset.get("scene_type") == "simulation"
        )
        dataset.setdefault(
            "category",
            "simulation" if is_simulation else "real_vehicle",
        )
        if "source_kind" not in dataset:
            if moving.get("video"):
                dataset["source_kind"] = "video"
            elif moving.get("frames"):
                dataset["source_kind"] = "frames"
            elif mcap.get("path") and moving.get("image_topic"):
                dataset["source_kind"] = "rosbag"
            else:
                dataset["source_kind"] = "prepared"
        simulation.setdefault("world_id", "bus")
        simulation.setdefault("world_baseline", {})
        payload["dataset"] = dataset
        payload["simulation"] = simulation
        return payload

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
