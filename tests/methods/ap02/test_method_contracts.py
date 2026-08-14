from __future__ import annotations

from dataclasses import FrozenInstanceError, fields, replace

import pytest

from camera_rig_calibration.config.models import AP01Settings, AP02Settings
from camera_rig_calibration.experiments import method_fingerprint
from camera_rig_calibration.methods.ap01.contracts import (
    ap01_execution_contract_name,
    resolve_ap01_method_contract,
)
from camera_rig_calibration.methods.ap02.contracts import (
    AP02MethodContract,
    resolve_ap02_method_contract,
)
from camera_rig_calibration.observations import ResolvedSelections


def _resolved() -> ResolvedSelections:
    return ResolvedSelections(
        root_camera="front-left",
        ap02_reference_marker_id=14,
        ap03_single_scale_marker_id=0,
        ap03_multi_marker_ids=(0,),
        evaluation_anchor_marker_id=None,
        marker_ids=(0, 14),
        payload={},
    )


def test_ap01_baseline_v1_uses_fresh_inputs_by_default() -> None:
    settings = AP01Settings()
    assert settings.method_contract == "baseline_v1"
    contract = resolve_ap01_method_contract(
        ap01_execution_contract_name(settings.method_contract)
    )
    assert contract.name == "baseline_v1"
    assert contract.sfm_execution_policy == "fresh_colmap"
    assert contract.scale_execution_policy == "fresh_metric_scale_estimation"
    assert contract.quality_model == (
        "baseline_area_over_distance_squared_center_v1"
    )
    assert contract.direct_target_policy == "single_configured_target"
    assert contract.relay_input_limit is None


def test_ap02_baseline_v1_resolves_complete_contract() -> None:
    settings = AP02Settings()
    contract = resolve_ap02_method_contract(
        settings.method_contract,
        reference_marker_selection_mode=(
            settings.reference_marker_selection_mode
        ),
        reference_marker_id=settings.reference_marker_id,
        frame_selection_strategy=settings.frame_selection_strategy,
        initialization_strategy=settings.initialization_strategy,
        graph_edge_weight_strategy=settings.graph_edge_weight_strategy,
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
    assert contract.name == "baseline_v1"
    assert contract.reference_marker_id == 14
    assert contract.graph_observation_policy == (
        "quality_valid_all_observations_v1"
    )
    assert contract.moving_frame_selection_policy == (
        "smart_at_ba_boundary_v1"
    )
    assert contract.top_per_marker == 8
    assert contract.top_per_marker_pair == 4
    assert contract.initialization_algorithm == (
        "maximum_frontier_v1"
    )
    assert contract.graph_edge_weight_policy == (
        "geometric_observation_quality_v1"
    )
    assert contract.reprojection_model == "pinhole_v1"
    assert contract.robust_loss == "soft_l1"
    assert contract.robust_loss_scale_px == 3.0
    assert contract.static_maximum_function_evaluations == 80
    assert contract.combined_maximum_function_evaluations == 80
    assert contract.ground_truth_policy == "forbidden_during_calibration"


def test_ap02_contract_is_immutable_complete_and_fingerprint_sensitive() -> None:
    contract = resolve_ap02_method_contract()
    assert set(contract.fingerprint_payload()) == {
        field.name for field in fields(AP02MethodContract)
    }
    with pytest.raises(FrozenInstanceError):
        contract.robust_loss = "linear"  # type: ignore[misc]
    original = contract.scientific_fingerprint()
    for field in fields(AP02MethodContract):
        value = getattr(contract, field.name)
        if isinstance(value, bool):
            changed = not value
        elif isinstance(value, int):
            changed = value + 1
        elif isinstance(value, float):
            changed = value + 0.25
        elif isinstance(value, str):
            changed = value + "_probe"
        elif isinstance(value, tuple):
            changed = (*value, "probe")
        elif value is None:
            changed = "probe"
        else:  # pragma: no cover
            raise AssertionError(field.name)
        assert replace(
            contract, **{field.name: changed}
        ).scientific_fingerprint() != original


def test_ap02_wizard_robustness_features_remain_explicitly_configurable() -> None:
    contract = resolve_ap02_method_contract(
        frame_selection_strategy="wizard_graph_preserving_v1",
        initialization_strategy="wizard_maximum_bottleneck_v2",
        graph_edge_weight_strategy="wizard_selection_score_v2",
        reprojection_model="distortion_aware_v1",
        top_per_marker=5,
        top_per_marker_pair=2,
        maximum_total_frames=40,
        robust_loss="huber",
        robust_loss_scale_px=1.5,
    )
    assert contract.graph_observation_policy == (
        "wizard_graph_preserving_preselection_v1"
    )
    assert contract.moving_frame_selection_policy == (
        "all_graph_preselected_frames_v1"
    )
    assert contract.initialization_algorithm == (
        "wizard_maximum_bottleneck_v2"
    )
    assert contract.graph_edge_weight_policy == "wizard_selection_score_v2"
    assert contract.reprojection_model == "distortion_aware_v1"
    assert contract.robust_loss == "huber"
    assert contract.scientific_fingerprint() != (
        resolve_ap02_method_contract().scientific_fingerprint()
    )


def test_ap01_and_ap02_advanced_state_change_method_fingerprints(
    prepared_config,
) -> None:
    ap01 = prepared_config.model_copy(deep=True)
    ap01.methods.enabled = ["ap01"]
    advanced_ap01 = ap01.model_copy(deep=True)
    advanced_ap01.methods.ap01.method_contract = "recommended_wizard_v1"
    assert method_fingerprint(ap01, "ap01", _resolved()) != method_fingerprint(
        advanced_ap01, "ap01", _resolved()
    )

    ap02 = prepared_config.model_copy(deep=True)
    ap02.methods.enabled = ["ap02"]
    advanced = ap02.model_copy(deep=True)
    advanced.methods.ap02.frame_selection_strategy = (
        "wizard_graph_preserving_v1"
    )
    assert method_fingerprint(ap02, "ap02", _resolved()) != method_fingerprint(
        advanced, "ap02", _resolved()
    )
