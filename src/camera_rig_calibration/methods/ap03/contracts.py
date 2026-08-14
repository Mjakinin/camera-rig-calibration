"""Immutable scientific contract for the canonical AP03 baseline."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Literal, Sequence


AP03ContractName = Literal["baseline_v1"]
AP03_CONTRACT_NAMES: tuple[AP03ContractName, ...] = ("baseline_v1",)
DEFAULT_AP03_CONTRACT: AP03ContractName = "baseline_v1"


@dataclass(frozen=True)
class AP03MethodContract:
    """Every resolved AP03 choice that can change scientific output."""

    name: AP03ContractName
    contract_schema_version: int
    static_image_policy: str
    moving_image_policy: str
    image_ordering_policy: str
    image_registration_policy: str
    physical_camera_group_policy: str
    camera_model_policy: str
    intrinsic_serialization_policy: str
    refine_focal_length: bool
    refine_principal_point: bool
    refine_extra_parameters: bool
    feature_limit_policy: str
    sift_maximum_image_size: int | None
    sift_maximum_features: int | None
    sift_use_gpu: bool
    matching_strategy: str
    matching_use_gpu: bool
    sequential_overlap: int
    sequential_loop_detection: bool | None
    mapper_minimum_matches: int
    sparse_model_selection_policy: str
    registered_image_acceptance_policy: str
    scale_input_policy: str
    scale_marker_ids: tuple[int, ...]
    scale_marker_length_m: float
    scale_dictionary: str
    scale_detection_mode: str
    scale_minimum_marker_area_px2: float | None
    scale_reprojection_threshold_px: float
    scale_ransac_iterations: int
    scale_minimum_inliers: int
    scale_maximum_observations_per_marker: int | None
    scale_observation_geometry: str
    scale_outlier_policy: str
    scale_final_statistic: str
    pose_extraction_policy: str
    reference_frame_convention: str
    final_static_camera_policy: str
    multi_single_semantics: str
    final_export_ordering: str
    ground_truth_policy: str

    def fingerprint_payload(self) -> dict[str, object]:
        return asdict(self)

    def scientific_fingerprint(self) -> str:
        encoded = json.dumps(
            self.fingerprint_payload(), sort_keys=True, separators=(",", ":")
        )
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def resolve_ap03_method_contract(
    name: AP03ContractName | str = DEFAULT_AP03_CONTRACT,
    *,
    feature_limit_policy: str = "colmap_defaults_v1",
    scale_input_policy: str = "registered_image_redetection_v1",
    scale_marker_ids: Sequence[int] = tuple(range(15)),
    marker_length_m: float = 0.17,
    marker_dictionary: str = "DICT_4X4_50",
    marker_detection_mode: str = "baseline",
    minimum_marker_area_px2: float = 100.0,
    reprojection_threshold_px: float = 5.0,
    ransac_iterations: int = 1000,
    minimum_inliers: int = 4,
    maximum_observations_per_marker: int | None = None,
    colmap_matcher: str = "exhaustive",
    colmap_use_gpu: bool = False,
    colmap_maximum_image_size: int = 2400,
    colmap_maximum_features: int = 8192,
    colmap_sequential_overlap: int = 20,
    colmap_loop_detection: bool | None = None,
    colmap_mapper_minimum_matches: int = 8,
) -> AP03MethodContract:
    if name != "baseline_v1":
        raise ValueError(
            f"Unknown AP03 method contract '{name}'; choose baseline_v1"
        )
    if feature_limit_policy not in {
        "colmap_defaults_v1",
        "wizard_explicit_limits_v1",
    }:
        raise ValueError(f"Unknown AP03 feature-limit policy: {feature_limit_policy}")
    if scale_input_policy not in {
        "registered_image_redetection_v1",
        "wizard_filtered_observations_v1",
    }:
        raise ValueError(f"Unknown AP03 scale-input policy: {scale_input_policy}")
    marker_ids = tuple(sorted(set(int(value) for value in scale_marker_ids)))
    if not marker_ids:
        raise ValueError("AP03 scale marker set must not be empty")
    explicit_limits = feature_limit_policy == "wizard_explicit_limits_v1"
    redetection_scale = scale_input_policy == "registered_image_redetection_v1"
    return AP03MethodContract(
        name="baseline_v1",
        contract_schema_version=1,
        static_image_policy="one_canonical_image_per_configured_static_camera_v1",
        moving_image_policy="all_sorted_prepared_moving_frames_v1",
        image_ordering_policy="configured_static_order_then_moving_filename_v1",
        image_registration_policy="joint_static_and_moving_sparse_reconstruction_v1",
        physical_camera_group_policy="one_camera_per_static_shared_camera_for_moving_v1",
        camera_model_policy="calibration_distortion_mapping_v1",
        intrinsic_serialization_policy="camera_info_decimal_17g_v1",
        refine_focal_length=False,
        refine_principal_point=False,
        refine_extra_parameters=False,
        feature_limit_policy=feature_limit_policy,
        sift_maximum_image_size=(colmap_maximum_image_size if explicit_limits else None),
        sift_maximum_features=(colmap_maximum_features if explicit_limits else None),
        sift_use_gpu=colmap_use_gpu,
        matching_strategy=colmap_matcher,
        matching_use_gpu=colmap_use_gpu,
        sequential_overlap=colmap_sequential_overlap,
        sequential_loop_detection=(
            colmap_loop_detection
            if colmap_matcher == "sequential" and explicit_limits
            else None
        ),
        mapper_minimum_matches=colmap_mapper_minimum_matches,
        sparse_model_selection_policy=(
            "registered_static_then_registered_total_then_points_descending_v1"
        ),
        registered_image_acceptance_policy="all_images_in_selected_colmap_model_v1",
        scale_input_policy=scale_input_policy,
        scale_marker_ids=marker_ids,
        scale_marker_length_m=marker_length_m,
        scale_dictionary=marker_dictionary,
        scale_detection_mode=marker_detection_mode,
        scale_minimum_marker_area_px2=minimum_marker_area_px2,
        scale_reprojection_threshold_px=reprojection_threshold_px,
        scale_ransac_iterations=ransac_iterations,
        scale_minimum_inliers=minimum_inliers,
        scale_maximum_observations_per_marker=(
            maximum_observations_per_marker if not redetection_scale else None
        ),
        scale_observation_geometry="four_sides_plus_two_diagonals_per_complete_marker_v1",
        scale_outlier_policy="median_max_3mad_or_10_percent_then_min4_fallback_all_v1",
        scale_final_statistic="median_inlier_metres_per_colmap_unit_v1",
        pose_extraction_policy="registered_static_T_colmap_camera_v1",
        reference_frame_convention="native_colmap_gauge_translation_scaled_only_v1",
        final_static_camera_policy="export_every_registered_static_camera_v1",
        multi_single_semantics="multi_primary_single_diagnostic_shared_reconstruction_v1",
        final_export_ordering="static_camera_id_ascending_then_unordered_pairs_v1",
        ground_truth_policy="forbidden_during_reconstruction_and_calibration",
    )
