"""Explicit, immutable scientific contracts for AP01.

The preset name is resolved once at the AP01 boundary.  Candidate construction
and selection consume the resulting fields and never infer a mode from a
dataset name or filesystem path.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Literal, Sequence


AP01ContractName = Literal[
    "baseline_v1",
    "recommended_wizard_v1",
]
AP01_CONTRACT_NAMES: tuple[AP01ContractName, ...] = (
    "baseline_v1",
    "recommended_wizard_v1",
)
DEFAULT_AP01_CONTRACT: AP01ContractName = "baseline_v1"


def ap01_execution_contract_name(
    name: AP01ContractName | str,
) -> AP01ContractName:
    """Validate and return the selected AP01 strategy."""

    if name not in AP01_CONTRACT_NAMES:
        choices = ", ".join(AP01_CONTRACT_NAMES)
        raise ValueError(f"Unknown AP01 method contract '{name}'; choose {choices}")
    return name


@dataclass(frozen=True)
class AP01MethodContract:
    """Complete AP01 behavior that can affect scientific output."""

    name: AP01ContractName
    contract_schema_version: int
    sfm_execution_policy: str
    colmap_camera_model_policy: str
    colmap_single_shared_camera: bool
    colmap_intrinsics_serialization: str
    colmap_intrinsics_precision: int
    colmap_sift_max_features: int
    colmap_sift_maximum_image_size: int
    colmap_sift_extraction_threads: int | None
    colmap_matching_mode: str
    colmap_matcher_use_gpu: bool
    colmap_sequential_overlap: int
    colmap_loop_detection: bool
    colmap_mapper_minimum_matches: int
    colmap_refine_focal_length: bool
    colmap_refine_principal_point: bool
    colmap_refine_extra_parameters: bool
    colmap_sparse_model_selection_policy: str
    scale_execution_policy: str
    scale_observation_construction_policy: str
    scale_registered_frames_only: bool
    scale_pnp_success_only: bool
    scale_pnp_quantity_policy: str
    scale_minimum_marker_area_px2: float | None
    scale_maximum_marker_distance_m: float | None
    scale_maximum_center_norm: float | None
    scale_frame_gap_minimum: int
    scale_frame_gap_maximum: int
    scale_metric_translation_minimum_m: float
    scale_metric_translation_maximum_m: float
    scale_colmap_translation_minimum_units: float
    scale_colmap_translation_rejection_policy: str
    scale_observation_limit_per_marker: int | None
    scale_sample_multiplicity_policy: str
    scale_pair_quality_policy: str
    scale_aggregation_policy: str
    scale_mad_sigma_factor: float
    scale_mad_multiplier: float
    scale_relative_deviation_floor_fraction: float | None
    scale_fallback_minimum_count: int
    scale_fallback_fraction: float
    scale_final_statistic: str
    scale_minimum_pair_count: int
    quality_model: str
    quality_image_width_px: int | None
    quality_image_height_px: int | None
    static_support_policy: str
    moving_support_policy: str
    direct_target_policy: str
    direct_target_camera: str | None
    relay_target_policy: str
    candidate_construction_order: str
    relay_input_limit: int | None
    direct_aggregation_policy: str
    relay_aggregation_policy: str
    candidate_priority_policy: str
    eligibility_policy: str
    consensus_policy: str
    missing_direct_policy: str
    omission_policy: str
    tie_break_policy: tuple[str, ...]
    preferred_direct_marker_id: int | None
    direct_minimum_area_px2: float | None
    direct_maximum_distance_m: float | None
    direct_minimum_combined_quality: float | None
    direct_quality_fallback_count: int | None
    direct_translation_mad_floor_m: float
    direct_rotation_mad_floor_deg: float
    relay_translation_mad_floor_m: float
    relay_rotation_mad_floor_deg: float
    relay_fallback_minimum_count: int | None
    relay_fallback_fraction: float | None
    final_pose_serialization_policy: str
    final_pose_serialization_decimal_places: int | None

    def fingerprint_payload(self) -> dict[str, object]:
        """Return every immutable field; callers must fingerprint this whole value."""

        return asdict(self)

    def scientific_fingerprint(self) -> str:
        canonical = json.dumps(
            self.fingerprint_payload(),
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def direct_targets(
        self, camera_ids: Sequence[str], root_camera: str
    ) -> tuple[str, ...]:
        non_root = tuple(camera for camera in camera_ids if camera != root_camera)
        if self.direct_target_policy == "all_non_root":
            return non_root
        if self.direct_target_policy == "single_configured_target":
            if self.direct_target_camera in non_root:
                return (str(self.direct_target_camera),)
            return ()
        raise ValueError(f"Unknown AP01 direct-target policy: {self.direct_target_policy}")

    def relay_targets(
        self, camera_ids: Sequence[str], root_camera: str
    ) -> tuple[str, ...]:
        if self.relay_target_policy == "all_non_root":
            return tuple(camera for camera in camera_ids if camera != root_camera)
        raise ValueError(f"Unknown AP01 relay-target policy: {self.relay_target_policy}")


def resolve_ap01_method_contract(
    name: AP01ContractName | str = DEFAULT_AP01_CONTRACT,
    *,
    direct_target_camera: str = "cam_edge_1",
    top_moving_per_marker: int | None = 8,
    scale_top_per_marker: int | None = 30,
    colmap_matcher: str = "exhaustive",
    colmap_use_gpu: bool = False,
    colmap_maximum_image_size: int = 1600,
    colmap_maximum_features: int = 4096,
    colmap_sequential_overlap: int = 20,
    colmap_loop_detection: bool = True,
    colmap_mapper_minimum_matches: int = 8,
) -> AP01MethodContract:
    """Resolve one of the two supported AP01 scientific strategies."""

    if name == "baseline_v1":
        return AP01MethodContract(
            name="baseline_v1",
            contract_schema_version=3,
            sfm_execution_policy="fresh_colmap",
            colmap_camera_model_policy="baseline_shared_pinhole_v1",
            colmap_single_shared_camera=True,
            colmap_intrinsics_serialization="fixed_decimal_places",
            colmap_intrinsics_precision=8,
            colmap_sift_max_features=4096,
            colmap_sift_maximum_image_size=1600,
            colmap_sift_extraction_threads=1,
            colmap_matching_mode="exhaustive",
            colmap_matcher_use_gpu=False,
            colmap_sequential_overlap=20,
            colmap_loop_detection=True,
            colmap_mapper_minimum_matches=15,
            colmap_refine_focal_length=False,
            colmap_refine_principal_point=False,
            colmap_refine_extra_parameters=False,
            colmap_sparse_model_selection_policy=(
                "maximum_registered_images_first_lexicographic_tie"
            ),
            scale_execution_policy="fresh_metric_scale_estimation",
            scale_observation_construction_policy=(
                "baseline_registered_quality_filters_then_all_pairs_v1"
            ),
            scale_registered_frames_only=True,
            scale_pnp_success_only=True,
            scale_pnp_quantity_policy=(
                "relative_camera_translation_norm_from_T_cam_marker_v1"
            ),
            scale_minimum_marker_area_px2=1200.0,
            scale_maximum_marker_distance_m=4.0,
            scale_maximum_center_norm=0.95,
            scale_frame_gap_minimum=3,
            scale_frame_gap_maximum=45,
            scale_metric_translation_minimum_m=0.12,
            scale_metric_translation_maximum_m=5.0,
            scale_colmap_translation_minimum_units=1e-9,
            scale_colmap_translation_rejection_policy="less_than",
            scale_observation_limit_per_marker=None,
            scale_sample_multiplicity_policy=(
                "all_within_marker_unordered_frame_pairs"
            ),
            scale_pair_quality_policy="sqrt_marker_area_product",
            scale_aggregation_policy="baseline_median_three_sigma_mad_v1",
            scale_mad_sigma_factor=1.4826,
            scale_mad_multiplier=3.0,
            scale_relative_deviation_floor_fraction=None,
            scale_fallback_minimum_count=10,
            scale_fallback_fraction=0.30,
            scale_final_statistic="median",
            scale_minimum_pair_count=10,
            quality_model="baseline_area_over_distance_squared_center_v1",
            quality_image_width_px=1280,
            quality_image_height_px=720,
            static_support_policy="best_quality_per_camera_marker_first_tie",
            moving_support_policy=(
                "best_quality_per_frame_marker_first_tie_registered_only_frame_ascending"
            ),
            direct_target_policy="single_configured_target",
            direct_target_camera=direct_target_camera,
            relay_target_policy="all_non_root",
            candidate_construction_order="all_direct_then_all_relay",
            relay_input_limit=None,
            direct_aggregation_policy=(
                "baseline_quality_filter_preferred_marker_then_medoid_v1"
            ),
            relay_aggregation_policy="baseline_flat_weighted_mad_v1",
            candidate_priority_policy="configured_direct_target_else_relay",
            eligibility_policy="available_aggregate_is_eligible",
            consensus_policy="none",
            missing_direct_policy="omit_without_relay_fallback",
            omission_policy="omit_missing_fixed_aggregate_and_continue",
            tie_break_policy=(
                "first_observation_on_equal_quality",
                "marker_id_ascending",
                "moving_frame_ascending",
                "first_medoid_on_equal_score",
                "stable_quality_sort",
                "first_preferred_aggregate_row",
            ),
            preferred_direct_marker_id=14,
            direct_minimum_area_px2=500.0,
            direct_maximum_distance_m=5.5,
            direct_minimum_combined_quality=20.0,
            direct_quality_fallback_count=2,
            direct_translation_mad_floor_m=0.08,
            direct_rotation_mad_floor_deg=2.0,
            relay_translation_mad_floor_m=0.30,
            relay_rotation_mad_floor_deg=7.0,
            relay_fallback_minimum_count=3,
            relay_fallback_fraction=0.5,
            final_pose_serialization_policy=(
                "baseline_aggregate_csv_rpy_roundtrip_v1"
            ),
            final_pose_serialization_decimal_places=9,
        )
    if name == "recommended_wizard_v1":
        return AP01MethodContract(
            name="recommended_wizard_v1",
            contract_schema_version=3,
            sfm_execution_policy="fresh_colmap",
            colmap_camera_model_policy="camera_info_distortion_model_v1",
            colmap_single_shared_camera=True,
            colmap_intrinsics_serialization="significant_digits",
            colmap_intrinsics_precision=17,
            colmap_sift_max_features=colmap_maximum_features,
            colmap_sift_maximum_image_size=colmap_maximum_image_size,
            colmap_sift_extraction_threads=None,
            colmap_matching_mode=colmap_matcher,
            colmap_matcher_use_gpu=colmap_use_gpu,
            colmap_sequential_overlap=colmap_sequential_overlap,
            colmap_loop_detection=colmap_loop_detection,
            colmap_mapper_minimum_matches=colmap_mapper_minimum_matches,
            colmap_refine_focal_length=False,
            colmap_refine_principal_point=False,
            colmap_refine_extra_parameters=False,
            colmap_sparse_model_selection_policy=(
                "maximum_registered_images_first_lexicographic_tie"
            ),
            scale_execution_policy="fresh_metric_scale_estimation",
            scale_observation_construction_policy=(
                "quality_ranked_per_marker_before_pairing_v1"
            ),
            scale_registered_frames_only=True,
            scale_pnp_success_only=True,
            scale_pnp_quantity_policy=(
                "relative_camera_translation_norm_from_T_cam_marker_v1"
            ),
            scale_minimum_marker_area_px2=None,
            scale_maximum_marker_distance_m=None,
            scale_maximum_center_norm=None,
            scale_frame_gap_minimum=2,
            scale_frame_gap_maximum=80,
            scale_metric_translation_minimum_m=0.05,
            scale_metric_translation_maximum_m=6.0,
            scale_colmap_translation_minimum_units=1e-10,
            scale_colmap_translation_rejection_policy="less_than_or_equal",
            scale_observation_limit_per_marker=scale_top_per_marker,
            scale_sample_multiplicity_policy=(
                "all_within_marker_unordered_frame_pairs"
            ),
            scale_pair_quality_policy="sqrt_observation_quality_product",
            scale_aggregation_policy="wizard_median_mad_relative_floor_v1",
            scale_mad_sigma_factor=1.4826,
            scale_mad_multiplier=3.0,
            scale_relative_deviation_floor_fraction=0.10,
            scale_fallback_minimum_count=10,
            scale_fallback_fraction=0.25,
            scale_final_statistic="median",
            scale_minimum_pair_count=10,
            quality_model="observation_quality_v2_selection_score",
            quality_image_width_px=None,
            quality_image_height_px=None,
            static_support_policy="best_quality_per_camera_marker_first_tie",
            moving_support_policy="quality_ranked_registered_then_frame_ascending",
            direct_target_policy="all_non_root",
            direct_target_camera=None,
            relay_target_policy="all_non_root",
            candidate_construction_order="per_target_direct_then_relay",
            relay_input_limit=top_moving_per_marker,
            direct_aggregation_policy="wizard_direct_robust_mad_v1",
            relay_aggregation_policy="wizard_hierarchical_marker_chain_mad_v1",
            candidate_priority_policy="stable_direct_then_stable_relay",
            eligibility_policy="configured_consensus_gates",
            consensus_policy="direct_relay_stability_and_consistency_v1",
            missing_direct_policy="allow_stable_relay_fallback",
            omission_policy="retain_diagnostic_pose_but_reject_deployment",
            tie_break_policy=(
                "first_observation_on_equal_quality",
                "quality_descending_then_frame_ascending",
                "camera_order_direct_before_relay",
                "marker_chain_ascending",
            ),
            preferred_direct_marker_id=None,
            direct_minimum_area_px2=None,
            direct_maximum_distance_m=None,
            direct_minimum_combined_quality=None,
            direct_quality_fallback_count=None,
            direct_translation_mad_floor_m=0.12,
            direct_rotation_mad_floor_deg=4.0,
            relay_translation_mad_floor_m=0.30,
            relay_rotation_mad_floor_deg=7.0,
            relay_fallback_minimum_count=3,
            relay_fallback_fraction=None,
            final_pose_serialization_policy="native_full_precision_v1",
            final_pose_serialization_decimal_places=None,
        )
    choices = ", ".join(AP01_CONTRACT_NAMES)
    raise ValueError(f"Unknown AP01 method contract '{name}'; choose {choices}")
