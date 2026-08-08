from __future__ import annotations

import json
from pathlib import Path

from camera_rig_calibration.components import register_builtin_components
from camera_rig_calibration.product_policy import (
    _DATASET_CONTEXT,
    _refresh_derived_tree,
    install_product_policy,
)


install_product_policy()

from camera_rig_calibration import wizard  # noqa: E402
from camera_rig_calibration.evaluation import reporting  # noqa: E402


def _job(method_id: str):
    register_builtin_components()
    return wizard._new_method_job(
        method_id,
        prompt_for_single_marker=False,
    )


def test_simulation_baseline_defaults_are_frozen_and_visible() -> None:
    token = _DATASET_CONTEXT.set("simulation")
    try:
        ap01 = _job("ap01")
        ap02 = _job("ap02")
        ap03 = _job("ap03")
        assert ap01.methods.ap01.method_contract == "baseline_v1"
        assert ap01.methods.ap01.advanced_strategy == "legacy_main_v1"
        assert ap01.methods.ap01.root_camera == "cam_edge_3"
        assert ap02.methods.ap02.reference_marker_selection_mode == "baseline"
        assert ap02.methods.ap02.reference_marker_id == 14
        assert ap02.methods.ap02.static_only_ba_max_function_evaluations == 80
        assert ap02.methods.ap02.combined_ba_max_function_evaluations == 80
        assert ap03.methods.ap03.method_contract == "baseline_v1"
        assert ap03.methods.ap03.single.scale_marker_id == 14
        assert ap03.methods.ap03.multi.marker_ids == list(range(15))
        for job in (ap01, ap02, ap03):
            assert job.evaluation.anchor_marker_id == 14
            assert job.evaluation.anchor_selection_mode == "explicit"
    finally:
        _DATASET_CONTEXT.reset(token)


def test_real_vehicle_uses_same_method_cores_with_data_driven_anchors() -> None:
    ap01 = _job("ap01")
    ap02 = _job("ap02")
    ap03 = _job("ap03")
    assert ap01.methods.ap01.method_contract == "baseline_v1"
    assert ap01.methods.ap01.advanced_strategy == "legacy_main_v1"
    assert ap01.methods.ap01.root_camera == "auto"
    assert ap02.methods.ap02.method_contract == "baseline_v1"
    assert ap02.methods.ap02.reference_marker_selection_mode == "auto"
    assert ap02.methods.ap02.reference_marker_id == "auto"
    assert ap03.methods.ap03.method_contract == "baseline_v1"
    for job in (ap01, ap02, ap03):
        assert job.evaluation.anchor_marker_id == "auto"
        assert job.evaluation.anchor_selection_mode == "auto"


def test_ap02_ui_describes_explicit_limits_not_smart_selection() -> None:
    token = _DATASET_CONTEXT.set("simulation")
    try:
        job = _job("ap02")
        rows = wizard._setting_rows(job)
    finally:
        _DATASET_CONTEXT.reset(token)
    text = "\n".join(
        f"{label} {current} {baseline} {description}"
        for _, _, label, current, baseline, description in rows
    ).lower()
    assert "smart" not in text
    assert "top frames per marker" in text
    assert "top frames per marker pair" in text
    assert "marker 14" in text


def test_reporting_contract_uses_actual_ap02_80_80_baseline() -> None:
    contract = reporting._baseline_contract(
        category="simulation",
        method_payloads=[
            {
                "method": "ap02",
                "label": "baseline",
                "config_summary": {
                    "reference_marker_id": 14,
                    "resolved_reference_marker_id": 14,
                    "static_max_nfev": 80,
                    "combined_max_nfev": 80,
                },
            }
        ],
        evaluation_anchor={"selected": 14},
    )
    assert contract["contract"] == "route2_cpu_ref14_80x80_v1"
    checks = contract["variants"][0]["checks"]
    assert "static_nfev_50" not in checks
    assert "combined_nfev_50" not in checks
    assert checks["static_nfev_80"] is True
    assert checks["combined_nfev_80"] is True


def test_derived_evaluations_refresh_without_touching_unrelated_files(
    tmp_path: Path,
) -> None:
    source = tmp_path / "transaction" / "evaluations"
    destination = tmp_path / "experiment" / "evaluations"
    source.mkdir(parents=True)
    destination.mkdir(parents=True)
    (source / "SELECTED_COMMON_EVALUATION.json").write_text(
        json.dumps({"anchor_marker_id": 14}), encoding="utf-8"
    )
    (destination / "SELECTED_COMMON_EVALUATION.json").write_text(
        json.dumps({"anchor_marker_id": 2}), encoding="utf-8"
    )
    unrelated = tmp_path / "experiment" / "methods" / "native.txt"
    unrelated.parent.mkdir(parents=True)
    unrelated.write_text("immutable", encoding="utf-8")

    _refresh_derived_tree(source, destination)

    assert json.loads(
        (destination / "SELECTED_COMMON_EVALUATION.json").read_text(
            encoding="utf-8"
        )
    )["anchor_marker_id"] == 14
    assert unrelated.read_text(encoding="utf-8") == "immutable"
