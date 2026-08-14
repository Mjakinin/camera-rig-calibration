from __future__ import annotations

from dataclasses import FrozenInstanceError, fields, replace
import json
from pathlib import Path

import numpy as np
import pytest

from camera_rig_calibration.config.models import MethodSettings
from camera_rig_calibration.experiments import method_fingerprint
from camera_rig_calibration.methods.ap01 import core, solve_extrinsics
from camera_rig_calibration.methods.ap01.build_candidates import (
    construct_candidates,
)
from camera_rig_calibration.methods.ap01.contracts import (
    AP01MethodContract,
    resolve_ap01_method_contract,
)
from camera_rig_calibration.observations import ResolvedSelections


ROOT = "root"
TARGET = "target"
def _transform(x: float = 0.0) -> np.ndarray:
    result = np.eye(4, dtype=np.float64)
    result[0, 3] = x
    return result


def _static(camera: str, marker: int, *, quality: float = 100.0) -> dict:
    return {
        "_camera": camera,
        "_marker": marker,
        "_quality": quality,
        "_area_px2": 900.0,
        "_distance_m": 1.0,
        "_T_cam_marker": _transform(float(marker) / 100.0),
        "observation_key": f"s:{camera}:{marker}",
    }


def _moving(marker: int, frame: int, quality: float) -> dict:
    return {
        "_marker": marker,
        "_frame": frame,
        "_quality": quality,
        "_T_cam_marker": _transform(float(marker + frame) / 1000.0),
        "observation_key": f"m:{marker}:{frame}",
    }


def _scale_row(
    frame: int,
    *,
    area: float = 2000.0,
    distance: float = 2.0,
    center_u: float = 640.0,
    center_v: float = 360.0,
) -> dict:
    transform = _transform(frame * 0.1)
    return {
        "_marker": 14,
        "_frame": frame,
        "_quality": 1000.0 - frame,
        "_area_px2": area,
        "_distance_m": distance,
        "_T_cam_marker": transform,
        "pnp_success": "True",
        "center_u": str(center_u),
        "center_v": str(center_v),
    }


def _scale_poses(frames: list[int]) -> dict[int, np.ndarray]:
    return {frame: _transform(frame * 0.2) for frame in frames}


def _option(command: list[str], name: str) -> str:
    return command[command.index(name) + 1]


def _candidate(marker: int = 14, *, x: float = 0.0) -> dict:
    return {
        "mode": "direct",
        "root_camera": ROOT,
        "target_camera": TARGET,
        "root_marker": marker,
        "target_marker": marker,
        "root_frame": "",
        "target_frame": "",
        "quality": 30.0,
        "root_area_px2": 900.0,
        "target_area_px2": 900.0,
        "root_distance_m": 1.0,
        "target_distance_m": 1.0,
        "T": _transform(x),
    }


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


def test_contracts_are_immutable_complete_and_fingerprint_distinct() -> None:
    baseline = resolve_ap01_method_contract("baseline_v1")
    recommended = resolve_ap01_method_contract("recommended_wizard_v1")
    assert set(baseline.fingerprint_payload()) == {
        field.name for field in fields(AP01MethodContract)
    }
    assert baseline.scientific_fingerprint() != recommended.scientific_fingerprint()
    with pytest.raises(FrozenInstanceError):
        baseline.quality_model = "changed"  # type: ignore[misc]


def test_baseline_contract_resolves_complete_baseline_colmap_behavior() -> None:
    contract = resolve_ap01_method_contract(
        "baseline_v1",
        colmap_matcher="sequential",
        colmap_use_gpu=True,
        colmap_maximum_image_size=999,
        colmap_maximum_features=999,
        colmap_mapper_minimum_matches=3,
    )

    assert contract.colmap_camera_model_policy == "baseline_shared_pinhole_v1"
    assert contract.colmap_single_shared_camera is True
    assert contract.colmap_intrinsics_serialization == "fixed_decimal_places"
    assert contract.colmap_intrinsics_precision == 8
    assert contract.colmap_sift_max_features == 4096
    assert contract.colmap_sift_maximum_image_size == 1600
    assert contract.colmap_sift_extraction_threads == 1
    assert contract.colmap_matching_mode == "exhaustive"
    assert contract.colmap_matcher_use_gpu is False
    assert contract.colmap_mapper_minimum_matches == 15
    assert contract.colmap_refine_focal_length is False
    assert contract.colmap_refine_principal_point is False
    assert contract.colmap_refine_extra_parameters is False


def test_recommended_contract_preserves_configurable_colmap_behavior() -> None:
    contract = resolve_ap01_method_contract(
        "recommended_wizard_v1",
        colmap_matcher="sequential",
        colmap_use_gpu=True,
        colmap_maximum_image_size=2048,
        colmap_maximum_features=6000,
        colmap_sequential_overlap=13,
        colmap_loop_detection=False,
        colmap_mapper_minimum_matches=9,
    )

    assert contract.colmap_camera_model_policy == (
        "camera_info_distortion_model_v1"
    )
    assert contract.colmap_intrinsics_serialization == "significant_digits"
    assert contract.colmap_intrinsics_precision == 17
    assert contract.colmap_sift_extraction_threads is None
    assert contract.colmap_matching_mode == "sequential"
    assert contract.colmap_matcher_use_gpu is True
    assert contract.colmap_sift_maximum_image_size == 2048
    assert contract.colmap_sift_max_features == 6000
    assert contract.colmap_sequential_overlap == 13
    assert contract.colmap_loop_detection is False
    assert contract.colmap_mapper_minimum_matches == 9
    assert contract.sfm_execution_policy == "fresh_colmap"
    assert contract.scale_execution_policy == "fresh_metric_scale_estimation"


def test_baseline_intrinsics_and_colmap_commands_are_fixed() -> None:
    contract = resolve_ap01_method_contract("baseline_v1")
    info = {
        "K": np.asarray(
            [[929.4671630859375, 0.0, 640.0], [0.0, 929.467134475708, 360.0], [0.0, 0.0, 1.0]]
        ),
        "D": np.zeros(8),
        "distortion_model": "plumb_bob",
    }
    model, parameters = core.colmap_camera_model(info, contract)
    feature = core.colmap_feature_extractor_command(
        executable="colmap",
        database=Path("database.db"),
        image_dir=Path("moving"),
        camera_model=model,
        camera_parameters=parameters,
        contract=contract,
    )
    mapper = core.colmap_mapper_command(
        executable="colmap",
        database=Path("database.db"),
        image_dir=Path("moving"),
        sparse=Path("sparse"),
        contract=contract,
    )

    assert model == "PINHOLE"
    assert parameters == (
        "929.46716309,929.46713448,640.00000000,360.00000000"
    )
    assert _option(feature, "--ImageReader.single_camera") == "1"
    assert _option(feature, "--SiftExtraction.use_gpu") == "0"
    assert _option(feature, "--SiftExtraction.num_threads") == "1"
    assert _option(feature, "--SiftExtraction.max_num_features") == "4096"
    assert _option(feature, "--SiftExtraction.max_image_size") == "1600"
    assert _option(mapper, "--Mapper.min_num_matches") == "15"
    assert _option(mapper, "--Mapper.ba_refine_focal_length") == "0"
    assert _option(mapper, "--Mapper.ba_refine_principal_point") == "0"
    assert _option(mapper, "--Mapper.ba_refine_extra_params") == "0"


def test_baseline_scale_pair_construction_bounds_filters_and_no_cap() -> None:
    contract = resolve_ap01_method_contract("baseline_v1")
    good_frames = [0, 3, 6, 9, 12]
    all_frames = [*good_frames, 15, 18, 21]
    rows = [
        *[_scale_row(frame) for frame in good_frames],
        _scale_row(15, area=1199.0),
        _scale_row(18, distance=4.01),
        _scale_row(21, center_u=1280.0, center_v=720.0),
    ]

    scale, statistics, pairs = core.robust_scale(
        rows,
        _scale_poses(all_frames),
        maximum_observations_per_marker=1,
        contract=contract,
    )

    assert contract.scale_observation_limit_per_marker is None
    assert statistics["selected_observations_per_marker"] == {14: 5}
    assert statistics["rejected_observations_by_reason"] == {
        "marker_area_below_minimum": 1,
        "marker_center_norm_above_maximum": 1,
        "marker_distance_above_maximum": 1,
    }
    assert statistics["raw_pairs"] == 10
    assert statistics["used_pairs"] == 10
    assert {(row["frame_i"], row["frame_j"]) for row in pairs} == {
        (first, second)
        for index, first in enumerate(good_frames)
        for second in good_frames[index + 1 :]
    }
    assert all(3 <= row["frame_gap"] <= 45 for row in pairs)
    assert scale == pytest.approx(0.5)
    assert all(row["quality"] == 2000.0 for row in pairs)


def test_recommended_scale_contract_retains_current_behavior() -> None:
    contract = resolve_ap01_method_contract(
        "recommended_wizard_v1", scale_top_per_marker=30
    )

    assert contract.scale_observation_construction_policy == (
        "quality_ranked_per_marker_before_pairing_v1"
    )
    assert contract.scale_observation_limit_per_marker == 30
    assert contract.scale_frame_gap_minimum == 2
    assert contract.scale_frame_gap_maximum == 80
    assert contract.scale_metric_translation_minimum_m == 0.05
    assert contract.scale_metric_translation_maximum_m == 6.0
    assert contract.scale_minimum_marker_area_px2 is None
    assert contract.scale_maximum_marker_distance_m is None
    assert contract.scale_maximum_center_norm is None
    assert contract.scale_relative_deviation_floor_fraction == 0.10
    assert contract.scale_fallback_fraction == 0.25
    assert contract.scale_pair_quality_policy == (
        "sqrt_observation_quality_product"
    )
    assert contract.final_pose_serialization_policy == (
        "native_full_precision_v1"
    )


def test_baseline_final_pose_uses_baseline_nine_decimal_rpy_roundtrip() -> None:
    contract = resolve_ap01_method_contract("baseline_v1")
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = core.rpy_deg_to_R(
        10.5240656136, -51.6364639324, -29.9764764846
    )
    transform[:3, 3] = [
        2.0445664746,
        -1.1795410854,
        3.0265273374,
    ]

    serialized = core.serialize_final_pose(transform, contract)

    assert serialized[:3, 3].tolist() == [
        2.044566475,
        -1.179541085,
        3.026527337,
    ]
    assert core.R_to_rpy_deg(serialized[:3, :3]) == pytest.approx(
        (10.524065614, -51.636463932, -29.976476485), abs=1e-12
    )


def test_every_method_contract_field_changes_scientific_fingerprint() -> None:
    contract = resolve_ap01_method_contract("baseline_v1")
    original = contract.scientific_fingerprint()

    for field in fields(AP01MethodContract):
        value = getattr(contract, field.name)
        if isinstance(value, bool):
            changed = not value
        elif isinstance(value, int):
            changed = value + 1
        elif isinstance(value, float):
            changed = value + 0.123456789
        elif isinstance(value, str):
            changed = value + "_fingerprint_probe"
        elif isinstance(value, tuple):
            changed = (*value, "fingerprint_probe")
        elif value is None:
            changed = "fingerprint_probe"
        else:  # pragma: no cover - protects additions of unsupported field types
            raise AssertionError(f"No fingerprint probe for {field.name}")
        mutated = replace(contract, **{field.name: changed})
        assert mutated.scientific_fingerprint() != original, field.name


def test_baseline_quality_formula_and_wizard_score_remain_separate() -> None:
    row = {
        "distance_m": "2",
        "center_u": "640",
        "center_v": "360",
        "selection_score": "0.125",
        **{
            f"corner{index}_{axis}": str(value)
            for index, (u, v) in enumerate(
                ((0.0, 0.0), (20.0, 0.0), (20.0, 20.0), (0.0, 20.0))
            )
            for axis, value in (("u", u), ("v", v))
        },
    }
    baseline_score, components = core.baseline_detection_quality(row)
    assert baseline_score == pytest.approx(400.0 / 4.0)
    assert components["area_px2_from_corners"] == 400.0
    assert float(row["selection_score"]) == 0.125
    assert baseline_score != float(row["selection_score"])


def test_baseline_direct_scope_order_and_uncapped_relay_multiplicity() -> None:
    camera_ids = ("target_b", "target_a", ROOT)
    contract = resolve_ap01_method_contract(
        "baseline_v1", direct_target_camera="target_a"
    )
    static = [
        _static(ROOT, 1),
        _static("target_a", 1),
        _static("target_a", 2),
        _static("target_b", 2),
    ]
    moving = [
        *[_moving(1, frame, 100.0 - frame) for frame in range(9)],
        *[_moving(2, frame + 20, 100.0 - frame) for frame in range(9)],
    ]
    poses = {int(row["_frame"]): np.eye(4) for row in moving}
    records, _ = construct_candidates(
        static_rows=static,
        moving_rows=moving,
        poses=poses,
        scale=1.0,
        camera_ids=camera_ids,
        root_camera=ROOT,
        contract=contract,
    )
    assert contract.direct_targets(camera_ids, ROOT) == ("target_a",)
    assert records[0]["mode"] == "direct"
    assert records[0]["target_camera"] == "target_a"
    assert all(record["mode"] == "relay" for record in records[1:])
    by_target = {
        target: sum(record["target_camera"] == target for record in records)
        for target in ("target_a", "target_b")
    }
    assert by_target == {"target_a": 154, "target_b": 81}


def test_recommended_contract_preserves_all_direct_scope_and_relay_cap() -> None:
    contract = resolve_ap01_method_contract(
        "recommended_wizard_v1", top_moving_per_marker=8
    )
    camera_ids = (ROOT, TARGET)
    static = [_static(ROOT, 1), _static(TARGET, 1)]
    moving = [_moving(1, frame, float(frame)) for frame in range(9)]
    poses = {frame: np.eye(4) for frame in range(9)}
    records, selection = construct_candidates(
        static_rows=static,
        moving_rows=moving,
        poses=poses,
        scale=1.0,
        camera_ids=camera_ids,
        root_camera=ROOT,
        contract=contract,
    )
    assert contract.direct_targets(camera_ids, ROOT) == (TARGET,)
    assert sum(record["mode"] == "direct" for record in records) == 1
    assert sum(record["mode"] == "relay" for record in records) == 56
    assert sum(bool(row["selected"]) for row in selection) == 8
    assert [
        row["frame_id"] for row in selection if not bool(row["selected"])
    ] == [0]


def test_baseline_direct_priority_and_missing_direct_omission() -> None:
    contract = resolve_ap01_method_contract(
        "baseline_v1", direct_target_camera=TARGET
    )
    candidates = [_candidate(2, x=2.0), _candidate(14, x=14.0)]
    result = solve_extrinsics.select_candidate_aggregates(
        candidates,
        camera_ids=(ROOT, TARGET),
        root_camera=ROOT,
        contract=contract,
        include_flattened=False,
    )
    assert result["poses"][TARGET][0, 3] == 14.0
    assert result["camera_statuses"][TARGET]["deployment_eligible"] is True
    missing = solve_extrinsics.select_candidate_aggregates(
        [],
        camera_ids=(ROOT, TARGET),
        root_camera=ROOT,
        contract=contract,
        include_flattened=False,
    )
    assert TARGET not in missing["poses"]
    assert missing["camera_statuses"][TARGET]["quality_status"] == (
        "unavailable_missing_direct_aggregate"
    )


def test_baseline_selection_diagnostics_are_json_serializable() -> None:
    contract = resolve_ap01_method_contract(
        "baseline_v1", direct_target_camera=TARGET
    )
    result = solve_extrinsics.select_candidate_aggregates(
        [_candidate(2, x=2.0), _candidate(14, x=14.0)],
        camera_ids=(ROOT, TARGET),
        root_camera=ROOT,
        contract=contract,
        include_flattened=False,
    )

    serialized = json.dumps(result["diagnostics"], allow_nan=False)

    assert "quality_filtered_weighted_mean_diagnostic" in serialized


def test_same_candidate_is_baseline_eligible_but_recommended_rejected() -> None:
    baseline = solve_extrinsics.select_candidate_aggregates(
        [_candidate()],
        camera_ids=(ROOT, TARGET),
        root_camera=ROOT,
        contract=resolve_ap01_method_contract(
            "baseline_v1", direct_target_camera=TARGET
        ),
        include_flattened=False,
    )
    recommended = solve_extrinsics.select_candidate_aggregates(
        [_candidate()],
        camera_ids=(ROOT, TARGET),
        root_camera=ROOT,
        contract=resolve_ap01_method_contract("recommended_wizard_v1"),
        include_flattened=False,
    )
    assert baseline["camera_statuses"][TARGET]["deployment_eligible"] is True
    assert recommended["camera_statuses"][TARGET]["deployment_eligible"] is False
    assert recommended["camera_statuses"][TARGET]["quality_status"] == (
        "rejected_unstable_consensus"
    )
    assert recommended["diagnostics"][TARGET]["direct"]["gate_checks"][
        "minimum_independent_markers"
    ] is False


def test_ap01_method_fingerprint_changes_with_resolved_contract(
    prepared_config,
) -> None:
    recommended = prepared_config.model_copy(
        update={"methods": MethodSettings(enabled=["ap01"])}, deep=True
    )
    recommended.methods.ap01.method_contract = "recommended_wizard_v1"
    baseline = recommended.model_copy(deep=True)
    baseline.methods.ap01.method_contract = "baseline_v1"
    assert method_fingerprint(
        recommended, "ap01", _resolved()
    ) != method_fingerprint(baseline, "ap01", _resolved())


def test_pure_contract_boundaries_invoke_no_stage_runner_or_colmap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*args, **kwargs):
        raise AssertionError("forbidden stage invoked")

    monkeypatch.setattr(solve_extrinsics, "run", forbidden)
    monkeypatch.setattr(core, "run_colmap", forbidden)
    result = solve_extrinsics.select_candidate_aggregates(
        [_candidate()],
        camera_ids=(ROOT, TARGET),
        root_camera=ROOT,
        contract=resolve_ap01_method_contract(
            "baseline_v1", direct_target_camera=TARGET
        ),
        include_flattened=False,
    )
    assert result["camera_statuses"][TARGET]["deployment_eligible"] is True
