from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path

import pytest

from parity.main_route2_v1.ap01_final_pose_parity import (
    PARITY_CONTRACT,
    RECOMMENDED_CONTRACT,
    assert_final_pose_input_path,
    compare_final_pose_views,
    compose_direct_candidate,
    compose_relay_candidate,
    identity_transform,
    invert_transform,
    legacy_aggregate_csv_roundtrip,
    legacy_final_pose_view,
    make_transform,
    multiply_transforms,
    quaternion_sign_equivalent,
    so3_metrics,
    wizard_final_pose_view,
)


def _selected_payload() -> dict:
    angle = 0.31
    rotation = [
        [math.cos(angle), -math.sin(angle), 0.0],
        [math.sin(angle), math.cos(angle), 0.0],
        [0.0, 0.0, 1.0],
    ]
    camera_order = ["cam_edge_3", "cam_edge_0", "cam_edge_1", "cam_edge_5"]
    candidate_types = {
        "cam_edge_3": "root",
        "cam_edge_0": "relay",
        "cam_edge_1": "direct",
        "cam_edge_5": "relay",
    }
    records = []
    for index, camera in enumerate(camera_order):
        transform = (
            identity_transform()
            if camera == "cam_edge_3"
            else make_transform(
                rotation,
                [index + 0.1234567894, -index - 0.9876543216, index / 3.0],
            )
        )
        records.append(
            {
                "camera_id": camera,
                "role": "anchor" if camera == "cam_edge_3" else "target",
                "selected_candidate_type": candidate_types[camera],
                "selected_method": "gauge_identity"
                if camera == "cam_edge_3"
                else "frozen_selected_aggregate",
                "selected_candidate_identity": {
                    "semantic_candidate_key": f"sha256:{index:064x}"
                },
                "homogeneous_transform_4x4": transform,
                "transform_chain": {
                    "column_vector_equation": (
                        f"p_cam_edge_3 = T_cam_edge_3_{camera} @ p_{camera}"
                    ),
                    "inversion_history": []
                    if camera == "cam_edge_3"
                    else ["inverse(source_support)"],
                    "composition_history": ["I_4x4"]
                    if camera == "cam_edge_3"
                    else ["selected_aggregate_transform"],
                },
                "deployment_eligible": True,
                "omitted": False,
                "omission_reason": None,
            }
        )
    return {
        "root_camera": "cam_edge_3",
        "camera_order": camera_order,
        "target_export_order": ["cam_edge_0", "cam_edge_1", "cam_edge_5"],
        "selected_candidates": records,
    }


def _max_delta(first: list[list[float]], second: list[list[float]]) -> float:
    return max(
        abs(first[i][j] - second[i][j])
        for i in range(4)
        for j in range(4)
    )


def test_root_camera_is_identity_in_both_final_pose_views() -> None:
    selected = _selected_payload()
    legacy = legacy_final_pose_view(selected)
    wizard = wizard_final_pose_view(
        selected, method_contract=PARITY_CONTRACT, parity_adapter=True
    )
    for view in (legacy, wizard):
        root = next(
            row for row in view["camera_records"] if row["camera_id"] == "cam_edge_3"
        )
        assert root["role"] == "anchor"
        assert root["homogeneous_transform_4x4"] == identity_transform()


@pytest.mark.parametrize("camera", ["cam_edge_1", "cam_edge_0"])
def test_direct_and_relay_selected_transforms_use_legacy_roundtrip(camera: str) -> None:
    selected = _selected_payload()
    source = next(
        row for row in selected["selected_candidates"] if row["camera_id"] == camera
    )
    expected, details = legacy_aggregate_csv_roundtrip(
        source["homogeneous_transform_4x4"]
    )
    actual = next(
        row
        for row in legacy_final_pose_view(selected)["camera_records"]
        if row["camera_id"] == camera
    )
    assert actual["homogeneous_transform_4x4"] == expected
    assert actual["serialization"]["decimal_places"] == 9
    assert actual["serialization"]["serialized_translation_m"] == details[
        "serialized_translation_m"
    ]


def test_transform_inverse_is_a_two_sided_identity() -> None:
    transform = make_transform(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        [1.0, 2.0, 3.0],
    )
    inverse = invert_transform(transform)
    assert _max_delta(multiply_transforms(transform, inverse), identity_transform()) < 1e-15
    assert _max_delta(multiply_transforms(inverse, transform), identity_transform()) < 1e-15


def test_direct_transform_multiplication_order_maps_target_to_root() -> None:
    root_marker = make_transform(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        [10.0, 0.0, 0.0],
    )
    target_marker = make_transform(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        [4.0, 0.0, 0.0],
    )
    direct = compose_direct_candidate(root_marker, target_marker)
    assert direct[0][3] == 6.0
    assert compose_direct_candidate(target_marker, root_marker)[0][3] == -6.0


def test_relay_transform_multiplication_order() -> None:
    def tx(value: float) -> list[list[float]]:
        return make_transform(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            [value, 0.0, 0.0],
        )

    relay = compose_relay_candidate(tx(10.0), tx(2.0), tx(3.0), tx(8.0), tx(1.0))
    # 10 - 2 + 3 - (8 - 1) = 4
    assert relay[0][3] == 4.0


def test_frames_and_homogeneous_output_are_explicit() -> None:
    record = next(
        row
        for row in legacy_final_pose_view(_selected_payload())["camera_records"]
        if row["camera_id"] == "cam_edge_1"
    )
    assert record["frame_from"] == "cam_edge_1_optical_frame"
    assert record["frame_to"] == "cam_edge_3_optical_frame"
    assert record["transform_direction"] == "target_to_root"
    assert record["homogeneous_transform_4x4"][3] == [0.0, 0.0, 0.0, 1.0]


def test_final_rotations_are_valid_so3() -> None:
    for record in legacy_final_pose_view(_selected_payload())["camera_records"]:
        metrics = so3_metrics(record["rotation_matrix"])
        assert metrics["valid"] is True
        assert abs(metrics["determinant"] - 1.0) < 1e-12


def test_quaternion_global_sign_is_representation_equivalent() -> None:
    quaternion = [0.5, 0.5, -0.5, 0.5]
    assert quaternion_sign_equivalent(quaternion, [-value for value in quaternion])
    assert not quaternion_sign_equivalent(quaternion, [0.5, 0.5, 0.5, 0.5])


def test_parity_adapter_has_legacy_order_and_anchor_omission() -> None:
    selected = _selected_payload()
    legacy = legacy_final_pose_view(selected)
    wizard = wizard_final_pose_view(
        selected, method_contract=PARITY_CONTRACT, parity_adapter=True
    )
    assert wizard["camera_order"] == [
        "cam_edge_3",
        "cam_edge_0",
        "cam_edge_1",
        "cam_edge_5",
    ]
    assert wizard["export_semantics"] == legacy["export_semantics"]
    assert "cam_edge_3" not in wizard["export_semantics"][
        "explicit_pose_record_inventory"
    ]


def test_contract_isolation_keeps_recommended_wizard_behavior() -> None:
    selected = _selected_payload()
    recommended = wizard_final_pose_view(
        selected,
        method_contract=RECOMMENDED_CONTRACT,
        parity_adapter=False,
    )
    direct_source = next(
        row
        for row in selected["selected_candidates"]
        if row["camera_id"] == "cam_edge_1"
    )
    direct_output = next(
        row
        for row in recommended["camera_records"]
        if row["camera_id"] == "cam_edge_1"
    )
    assert recommended["camera_order"] == sorted(selected["camera_order"])
    assert direct_output["homogeneous_transform_4x4"] == direct_source[
        "homogeneous_transform_4x4"
    ]
    assert recommended["export_semantics"]["anchor_representation"] == "explicit_pose_row"
    with pytest.raises(ValueError, match="parity-contract only"):
        wizard_final_pose_view(
            selected,
            method_contract=RECOMMENDED_CONTRACT,
            parity_adapter=True,
        )


def test_current_behavior_mismatch_and_parity_fix_are_detected() -> None:
    selected = _selected_payload()
    legacy = legacy_final_pose_view(selected)
    current = wizard_final_pose_view(
        selected, method_contract=PARITY_CONTRACT, parity_adapter=False
    )
    fixed = wizard_final_pose_view(
        selected, method_contract=PARITY_CONTRACT, parity_adapter=True
    )
    pre_report, _ = compare_final_pose_views(legacy, current)
    post_report, rows = compare_final_pose_views(legacy, fixed)
    assert pre_report["classification"] == "DIFFERENT_EXPORT_SEMANTICS"
    assert pre_report["first_causal_divergence"]["camera_id"] == "cam_edge_0"
    assert post_report["classification"] == "EXACT"
    assert all(row["status"] == "EXACT" for row in rows)


def test_final_pose_input_guard_rejects_ground_truth_before_read(tmp_path: Path) -> None:
    forbidden = tmp_path / "ground_truth" / "poses.json"
    with pytest.raises(ValueError, match="forbidden"):
        assert_final_pose_input_path(forbidden)


def test_pure_views_do_not_invoke_external_pipeline_stages() -> None:
    selected = copy.deepcopy(_selected_payload())
    for view in (
        legacy_final_pose_view(selected),
        wizard_final_pose_view(
            selected, method_contract=PARITY_CONTRACT, parity_adapter=True
        ),
    ):
        assert view["ground_truth_used"] is False
        assert view["candidate_generation_invoked"] is False
        assert view["aggregate_selection_invoked"] is False
        assert view["solver_invoked"] is False
        assert view["colmap_invoked"] is False
        assert view["publication_invoked"] is False
        assert view["reconciliation_invoked"] is False


def test_repository_final_pose_evidence_is_exact_and_hash_locked() -> None:
    repository = Path(__file__).resolve().parents[1]
    evidence = repository / "parity/main_route2_v1"

    def load(relative: str) -> dict:
        return json.loads((evidence / relative).read_text(encoding="utf-8"))

    def digest(relative: str) -> str:
        sha = hashlib.sha256()
        with (evidence / relative).open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                sha.update(block)
        return sha.hexdigest()

    report = load("ap01/final_pose/AP01_FINAL_POSE_PARITY.json")
    manifest = load("frozen/AP01_FINAL_POSE_INPUT_MANIFEST.json")
    selected = load("frozen/AP01_SELECTED_CANDIDATES.json")
    lock = load("PARITY_LOCK.json")
    pre_fix = load("ap01/final_pose/pre_fix/PRE_FIX_MANIFEST.json")

    assert report["classification"] == "EXACT"
    assert report["first_causal_divergence"] is None
    assert report["maximum_rotation_matrix_element_abs_delta"] == 0.0
    assert report["maximum_relative_rotation_angle_deg"] == 0.0
    assert report["maximum_translation_vector_norm_delta_m"] == 0.0
    assert all(row["status"] == "EXACT" for row in report["per_camera"])
    assert manifest["selected_input"]["sha256"] == digest(
        "frozen/AP01_SELECTED_CANDIDATES.json"
    )
    assert manifest["selected_input"]["sha256"] == report[
        "selected_input_sha256"
    ]
    assert selected["camera_order"] == report["legacy_camera_inventory"]
    assert selected["camera_order"] == report["wizard_camera_inventory"]
    assert selected["ground_truth_payload_absent"] is True
    assert selected["candidate_generation_invoked"] is False
    assert selected["aggregate_selection_invoked"] is False
    assert pre_fix["classification"] == "DIFFERENT_EXPORT_SEMANTICS"
    for artifact in pre_fix["artifacts"]:
        assert artifact["sha256"] == digest(artifact["path"])
    assert lock["locks"]["ap01_final_pose_selected_candidates_sha256"] == digest(
        "frozen/AP01_SELECTED_CANDIDATES.json"
    )
    assert lock["locks"]["ap01_final_pose_parity_sha256"] == digest(
        "ap01/final_pose/AP01_FINAL_POSE_PARITY.json"
    )
