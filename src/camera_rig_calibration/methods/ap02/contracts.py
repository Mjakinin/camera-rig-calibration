"""Immutable scientific contract for the canonical AP02 baseline."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Literal


AP02ContractName = Literal["baseline_v1"]
AP02_CONTRACT_NAMES: tuple[AP02ContractName, ...] = ("baseline_v1",)
DEFAULT_AP02_CONTRACT: AP02ContractName = "baseline_v1"


@dataclass(frozen=True)
class AP02MethodContract:
    """Every AP02 option that can change scientific output."""

    name: AP02ContractName
    contract_schema_version: int
    reference_marker_policy: str
    reference_marker_id: int | str
    graph_observation_policy: str
    moving_frame_selection_policy: str
    reference_marker_maximum_frames: int | None
    top_per_marker: int | None
    top_per_marker_pair: int | None
    maximum_total_frames: int | None
    frame_scoring_policy: str
    graph_node_policy: str
    graph_edge_policy: str
    graph_edge_weight_policy: str
    initialization_observation_policy: str
    initialization_algorithm: str
    root_pose_policy: str
    pose_propagation_policy: str
    disconnected_frame_policy: str
    ba_observation_policy: str
    fixed_pose_policy: str
    variable_pose_policy: str
    parameter_block_policy: str
    parameter_ordering_policy: str
    residual_construction_policy: str
    residual_ordering_policy: str
    reprojection_model: str
    robust_loss: str
    robust_loss_scale_px: float
    static_maximum_function_evaluations: int
    combined_maximum_function_evaluations: int
    solver_method: str
    solver_bounds_policy: str
    deterministic_tie_policy: tuple[str, ...]
    ground_truth_policy: str

    def fingerprint_payload(self) -> dict[str, object]:
        return asdict(self)

    def scientific_fingerprint(self) -> str:
        encoded = json.dumps(
            self.fingerprint_payload(), sort_keys=True, separators=(",", ":")
        )
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def resolve_ap02_method_contract(
    name: AP02ContractName | str = DEFAULT_AP02_CONTRACT,
    *,
    reference_marker_selection_mode: str = "baseline",
    reference_marker_id: int | str = 14,
    frame_selection_strategy: str = "legacy_smart_v1",
    initialization_strategy: str = "legacy_maximum_bottleneck_v1",
    graph_edge_weight_strategy: str = "legacy_observation_quality_v1",
    reprojection_model: str = "legacy_pinhole_v1",
    reference_marker_maximum_frames: int | None = None,
    top_per_marker: int | None = 8,
    top_per_marker_pair: int | None = 4,
    maximum_total_frames: int | None = None,
    static_maximum_function_evaluations: int = 80,
    combined_maximum_function_evaluations: int = 80,
    robust_loss: str = "soft_l1",
    robust_loss_scale_px: float = 3.0,
) -> AP02MethodContract:
    if name != "baseline_v1":
        raise ValueError(
            f"Unknown AP02 method contract '{name}'; choose baseline_v1"
        )

    legacy_selection = frame_selection_strategy == "legacy_smart_v1"
    if not legacy_selection and frame_selection_strategy != (
        "wizard_graph_preserving_v1"
    ):
        raise ValueError(
            f"Unknown AP02 frame-selection strategy: {frame_selection_strategy}"
        )
    if initialization_strategy not in {
        "legacy_maximum_bottleneck_v1",
        "wizard_maximum_bottleneck_v2",
        "unweighted_bfs_diagnostic",
    }:
        raise ValueError(
            f"Unknown AP02 initialization strategy: {initialization_strategy}"
        )
    if graph_edge_weight_strategy not in {
        "legacy_observation_quality_v1",
        "wizard_selection_score_v2",
    }:
        raise ValueError(
            "Unknown AP02 graph-edge weight strategy: "
            f"{graph_edge_weight_strategy}"
        )
    if reprojection_model not in {
        "legacy_pinhole_v1",
        "distortion_aware_v1",
    }:
        raise ValueError(f"Unknown AP02 reprojection model: {reprojection_model}")

    return AP02MethodContract(
        name="baseline_v1",
        contract_schema_version=1,
        reference_marker_policy=reference_marker_selection_mode,
        reference_marker_id=reference_marker_id,
        graph_observation_policy=(
            "legacy_quality_valid_all_observations_v1"
            if legacy_selection
            else "wizard_graph_preserving_preselection_v1"
        ),
        moving_frame_selection_policy=(
            "legacy_smart_at_ba_boundary_v1"
            if legacy_selection
            else "all_graph_preselected_frames_v1"
        ),
        reference_marker_maximum_frames=reference_marker_maximum_frames,
        top_per_marker=top_per_marker,
        top_per_marker_pair=top_per_marker_pair,
        maximum_total_frames=maximum_total_frames,
        frame_scoring_policy=(
            "legacy_observation_quality_v1"
            if legacy_selection
            else "wizard_selection_score_v2"
        ),
        graph_node_policy="bipartite_observer_marker_v1",
        graph_edge_policy="best_observation_per_observer_marker_v1",
        graph_edge_weight_policy=graph_edge_weight_strategy,
        initialization_observation_policy="all_quality_valid_graph_edges_v1",
        initialization_algorithm=initialization_strategy,
        root_pose_policy="reference_marker_identity_v1",
        pose_propagation_policy="legacy_T_ref_entity_chain_v1",
        disconnected_frame_policy="omit_unreachable_v1",
        ba_observation_policy=(
            "all_static_plus_legacy_smart_moving_v1"
            if legacy_selection
            else "all_graph_preselected_observations_v1"
        ),
        fixed_pose_policy="reference_marker_only_v1",
        variable_pose_policy="used_observers_then_used_nonreference_markers_v1",
        parameter_block_policy="rotation_vector_then_translation_v1",
        parameter_ordering_policy="observer_id_then_marker_id_ascending_v1",
        residual_construction_policy="four_marker_corners_xy_v1",
        residual_ordering_policy="observation_input_then_corner_0_to_3_xy_v1",
        reprojection_model=reprojection_model,
        robust_loss=robust_loss,
        robust_loss_scale_px=robust_loss_scale_px,
        static_maximum_function_evaluations=(
            static_maximum_function_evaluations
        ),
        combined_maximum_function_evaluations=(
            combined_maximum_function_evaluations
        ),
        solver_method="scipy_least_squares_default_trf_v1",
        solver_bounds_policy="unbounded_v1",
        deterministic_tie_policy=(
            "first_input_observation_on_equal_score",
            "moving_frame_number_ascending_on_equal_score",
            "legacy_frontier_insertion_order_on_equal_edge_weight",
            "observer_id_then_marker_id_parameter_order",
        ),
        ground_truth_policy="forbidden_during_calibration",
    )
