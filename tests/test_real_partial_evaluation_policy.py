from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from camera_rig_calibration.observations import ResolvedSelections
from camera_rig_calibration.preflight import PreflightJobResult
from camera_rig_calibration.real_partial_evaluation_policy import (
    _calibration_readiness,
    _component_pose_detail,
    _component_summary_text,
    _evaluation_only_error,
    _selection_with_real_anchor,
    install_real_partial_evaluation_policy,
)


def _config(method_id: str):
    return SimpleNamespace(
        methods=SimpleNamespace(enabled=[method_id]),
    )


def test_evaluation_only_marker_zero_error_is_classified_nonblocking() -> None:
    assert _evaluation_only_error(
        "Real Vehicle canonical marker 0 was observed, but it is not "
        "export-compatible for every enabled method after filtering."
    )
    assert _evaluation_only_error(
        "Evaluation is enabled, but shared detection found no marker that can "
        "be selected as the common anchor."
    )
    assert not _evaluation_only_error(
        "The selected AP02 reference component is not calibratable as Combined BA"
    )


def test_ap02_incomplete_calibratable_graph_restores_ready_partial() -> None:
    report = SimpleNamespace(
        ap02_graph_diagnosis=SimpleNamespace(complete=False),
        warnings=(),
    )
    assert _calibration_readiness(_config("ap02"), report, ()) == "READY_PARTIAL"
    assert (
        _calibration_readiness(
            _config("ap02"), report, ("real calibration error",)
        )
        == "FAILED_PREFLIGHT"
    )


def test_marker_zero_can_remain_evaluation_anchor_without_changing_ap02_reference(
    tmp_path: Path,
) -> None:
    selections = ResolvedSelections(
        root_camera="camera_a",
        ap02_reference_marker_id=8,
        ap03_single_scale_marker_id=0,
        ap03_multi_marker_ids=(0, 8),
        evaluation_anchor_marker_id=8,
        marker_ids=(0, 8),
        payload={
            "evaluation_anchor": {
                "configured": "auto",
                "selected": 8,
                "observation_candidates": [8],
                "automatic_observation_candidates": [8],
            },
            "automatic_recommendations": {
                "evaluation_anchor_marker_id": 8,
            },
        },
    )
    report = PreflightJobResult(
        job_id="ap02_partial",
        status="READY_PARTIAL",
        errors=(),
        warnings=(),
        details=(),
        filter_result=None,
        selections=selections,
        output_directory=tmp_path,
    )
    updated = _selection_with_real_anchor(report, 0)
    assert updated.selections is not None
    assert updated.selections.ap02_reference_marker_id == 8
    assert updated.selections.evaluation_anchor_marker_id == 0
    assert updated.selections.payload["evaluation_anchor"]["selected"] == 0
    # Marker 0 remains the requested canonical anchor, but the scientific
    # compatibility evidence must not be forged to claim that it is evaluable.
    assert updated.selections.payload["evaluation_anchor"]["observation_candidates"] == [8]
    assert updated.selections.payload["evaluation_anchor"]["automatic_observation_candidates"] == [8]
    assert (
        updated.selections.payload["real_vehicle_marker_zero_policy"]
        ["compatibility_evidence_overridden"]
        is False
    )
    assert (
        updated.selections.payload["real_vehicle_marker_zero_policy"]
        ["calibration_gating"]
        is False
    )


def test_component_summary_is_visible_in_result_text_contract() -> None:
    payload = {
        "method": "ap02",
        "metrics": {
            "ap02_component_results": {
                "status": "partial_coverage",
                "primary_component_id": "component_01",
                "cross_component_extrinsics": "not_observable",
                "camera_pair_observability": [
                    {"status": "within_component"},
                    {"status": "not_observable"},
                ],
                "components": [
                    {
                        "component_id": "component_01",
                        "execution_status": "primary_component",
                        "anchor_marker_id": 8,
                        "static_cameras": ["camera_a", "camera_b"],
                        "marker_ids": [2, 8],
                        "moving_frame_count": 91,
                    },
                    {
                        "component_id": "component_02",
                        "execution_status": "available",
                        "local_reference_marker_id": 7,
                        "static_cameras": ["camera_c", "camera_d"],
                        "marker_ids": [7, 9],
                        "moving_frame_count": 65,
                        "quality_status": "converged",
                        "result_path": "component_02",
                    },
                ],
            }
        },
    }
    text = _component_summary_text(payload)
    assert "AP02 DISCONNECTED / PARTIAL COMPONENT RESULTS" in text
    assert "component_01" in text
    assert "component_02" in text
    assert "not_observable" in text
    assert "marker" not in text.lower() or "8" in text


def test_component_pose_detail_includes_local_numeric_extrinsics(
    tmp_path: Path,
) -> None:
    root = tmp_path / "result"
    component_root = (
        root
        / "diagnostics"
        / "method"
        / "component_diagnostics"
    )
    component_root.mkdir(parents=True)
    summary = {
        "primary_component_id": "component_01",
        "components": [
            {
                "component_id": "component_01",
                "execution_status": "primary_component",
                "anchor_marker_id": 8,
                "static_cameras": ["camera_a", "camera_b"],
            },
            {
                "component_id": "component_02",
                "execution_status": "available",
                "local_reference_marker_id": 7,
                "static_cameras": ["camera_c", "camera_d"],
            },
        ],
    }
    (component_root / "AP02_COMPONENT_RESULTS.json").write_text(
        json.dumps(summary), encoding="utf-8"
    )
    root.mkdir(parents=True, exist_ok=True)
    (root / "camera_extrinsics.csv").write_text(
        "entity_id,x_m,y_m,z_m,roll_deg,pitch_deg,yaw_deg\n"
        "camera_a,0,0,0,0,0,0\n"
        "camera_b,1,0,0,0,0,5\n",
        encoding="utf-8",
    )
    diagnostic = component_root / "component_02"
    diagnostic.mkdir()
    (diagnostic / "camera_extrinsics.csv").write_text(
        "entity_id,x_m,y_m,z_m,roll_deg,pitch_deg,yaw_deg\n"
        "camera_c,0,0,0,0,0,0\n"
        "camera_d,0,2,0,0,0,10\n",
        encoding="utf-8",
    )
    text = _component_pose_detail(root)
    assert "AP02 LOCAL COMPONENT CAMERA POSES" in text
    assert "component_01" in text
    assert "component_02" in text
    assert "camera_b: x=1.000000m" in text
    assert "camera_d: x=0.000000m, y=2.000000m" in text
    assert "NOT observable" in text


def test_real_vehicle_wizard_hides_evaluation_disable_switch() -> None:
    from camera_rig_calibration import wizard
    from camera_rig_calibration.product_policy import _DATASET_CONTEXT

    install_real_partial_evaluation_policy()
    token = _DATASET_CONTEXT.set("real_vehicle")
    try:
        job = wizard._new_method_job(
            "ap02", prompt_for_single_marker=False
        )
        assert job.evaluation.enabled is True
        rows = wizard._setting_rows(job, {"COMMON EVALUATION"})
        assert all(row[0] != "evaluation_enabled" for row in rows)
    finally:
        _DATASET_CONTEXT.reset(token)
