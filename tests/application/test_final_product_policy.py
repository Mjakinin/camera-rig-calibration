from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from camera_rig_calibration.components import register_builtin_components
from camera_rig_calibration.policies.marker_preference_policy import install_marker_preference_policy
from camera_rig_calibration.policies.product_policy import (
    _DATASET_CONTEXT,
    _install_reporting_policy,
    _refresh_derived_tree,
    install_product_policy,
)
from camera_rig_calibration.policies.real_vehicle_marker_zero_policy import (
    install_real_vehicle_marker_zero_policy,
)
from camera_rig_calibration.policies.reporting_authority_policy import (
    install_reporting_authority_policy,
)
from camera_rig_calibration.policies.submission_bindings import install_submission_bindings
from camera_rig_calibration.policies.submission_policy import install_submission_policy
from camera_rig_calibration.policies.ui_display_policy import install_ui_display_policy


install_product_policy()
install_reporting_authority_policy()
install_submission_policy()
install_marker_preference_policy()
install_real_vehicle_marker_zero_policy()
install_submission_bindings()
install_ui_display_policy()

from camera_rig_calibration import wizard  # noqa: E402
from camera_rig_calibration.evaluation import ap03_derived, reporting  # noqa: E402


def _job(method_id: str):
    register_builtin_components()
    return wizard._new_method_job(method_id, prompt_for_single_marker=False)


def test_simulation_baseline_defaults_are_frozen_and_visible() -> None:
    token = _DATASET_CONTEXT.set("simulation")
    try:
        ap01 = _job("ap01")
        ap02 = _job("ap02")
        ap03 = _job("ap03")
        assert ap01.methods.ap01.method_contract == "baseline_v1"
        assert ap01.methods.ap01.root_camera == "cam_edge_3"
        assert ap01.methods.ap01.direct_target_camera == "auto"
        assert ap02.methods.ap02.reference_marker_selection_mode == "auto"
        assert ap02.methods.ap02.reference_marker_id == 14
        assert ap02.methods.ap02.static_only_ba_max_function_evaluations == 80
        assert ap02.methods.ap02.combined_ba_max_function_evaluations == 80
        assert ap03.methods.ap03.method_contract == "baseline_v1"
        assert ap03.methods.ap03.single.scale_marker_id == 14
        assert ap03.methods.ap03.multi.marker_ids == list(range(15))
        for job in (ap01, ap02, ap03):
            assert job.evaluation.anchor_marker_id == 14
            assert job.evaluation.anchor_selection_mode == "auto"
            assert job.selection.mode == "auto"
    finally:
        _DATASET_CONTEXT.reset(token)


def test_real_vehicle_defaults_use_canonical_marker_zero() -> None:
    ap01 = _job("ap01")
    ap02 = _job("ap02")
    ap03 = _job("ap03")
    assert ap01.methods.ap01.method_contract == "baseline_v1"
    assert ap01.methods.ap01.root_camera == "auto"
    assert ap01.methods.ap01.direct_target_camera == "auto"
    assert ap02.methods.ap02.method_contract == "baseline_v1"
    assert ap02.methods.ap02.reference_marker_selection_mode == "auto"
    assert ap02.methods.ap02.reference_marker_id == 0
    assert ap03.methods.ap03.method_contract == "baseline_v1"
    assert ap03.methods.ap03.single.scale_marker_id == 0
    assert ap03.methods.ap03.multi.marker_ids == "auto"
    for job in (ap01, ap02, ap03):
        assert job.evaluation.anchor_marker_id == 0
        assert job.evaluation.anchor_selection_mode == "auto"
        assert job.selection.mode == "auto"


def test_real_vehicle_ui_explains_strict_zero_and_absence_only_fallback() -> None:
    real_ap02 = _job("ap02")
    real_text = "\n".join(
        f"{label} {current} {baseline} {description}"
        for _, _, label, current, baseline, description in wizard._setting_rows(real_ap02)
    ).lower()
    assert "marker 0 required if observed" in real_text
    assert "auto fallback only if absent" in real_text
    assert "not observable instead of failing the method" in real_text

    token = _DATASET_CONTEXT.set("simulation")
    try:
        sim_ap02 = _job("ap02")
        sim_text = "\n".join(
            f"{label} {current} {baseline} {description}"
            for _, _, label, current, baseline, description in wizard._setting_rows(sim_ap02)
        ).lower()
    finally:
        _DATASET_CONTEXT.reset(token)
    assert "preferred marker 14" in sim_text
    assert "fallback" in sim_text


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
    assert "smart_v1" not in text
    assert "top frames per marker" in text
    assert "top frames per marker pair" in text
    assert "marker 14" in text
    assert "algorithm variant" in text


def test_reporting_contract_accepts_preferred_14_fallback_baseline() -> None:
    # Policy tests are collected alongside modules that install additional
    # reporting wrappers. Re-assert the product layer at execution time so this
    # contract does not depend on pytest's module collection order.
    _install_reporting_policy()

    contract = reporting._baseline_contract(
        category="simulation",
        method_payloads=[
            {
                "method": "ap02",
                "label": "baseline",
                "config_summary": {
                    "reference_marker_id": 14,
                    "resolved_reference_marker_id": 14,
                    "reference_marker_selection_mode": "auto",
                    "static_max_nfev": 80,
                    "combined_max_nfev": 80,
                    "initialization_algorithm": "maximum_bottleneck",
                },
            },
            {
                "method": "ap01",
                "label": "old_diagnostic_variant",
                "config_summary": {"root_camera": "cam_edge_1"},
            },
        ],
        evaluation_anchor={"selected": 14},
    )
    assert contract["contract"] == "route2_cpu_ref14_80x80_v1"
    assert [(item["method"], item["label"]) for item in contract["variants"]] == [
        ("ap02", "baseline")
    ]
    checks = contract["variants"][0]["checks"]
    assert "static_nfev_50" not in checks
    assert "combined_nfev_50" not in checks
    assert checks["static_nfev_80"] is True
    assert checks["combined_nfev_80"] is True
    assert checks["reference_preference_14_with_auto_fallback"] is True
    assert checks["reference_marker_14"] is True


def test_published_common_evaluation_is_reporting_authority(tmp_path: Path) -> None:
    observations = tmp_path / "observations"
    evaluations = tmp_path / "evaluations"
    observations.mkdir(parents=True)
    evaluations.mkdir(parents=True)
    selection_path = observations / "SELECTION_CANDIDATES.json"
    selection_path.write_text(
        json.dumps(
            {
                "evaluation_anchor": {
                    "selected": 2,
                    "configured": "auto",
                    "reason": "old preflight recommendation",
                }
            }
        ),
        encoding="utf-8",
    )
    (evaluations / "SELECTED_COMMON_EVALUATION.json").write_text(
        json.dumps({"anchor_marker_id": 14}), encoding="utf-8"
    )

    effective = reporting._read_json(selection_path)["evaluation_anchor"]
    assert effective["selected"] == 14
    assert effective["configured"] == 14
    assert effective["selection_mode"] == "published_common_evaluation"


def test_direct_anchor_gt_rejects_mismatched_marker_frame() -> None:
    identity = np.eye(4, dtype=float)
    rows = reporting._anchor_camera_gt_rows(
        "ap02",
        "old_marker_2_variant",
        {"anchor_marker_id": 2, "cameras": [{"camera_id": "cam", "matrix": identity.tolist()}]},
        anchor_marker_id=14,
        gt_cameras={"cam": identity},
        gt_markers={14: identity},
    )
    assert rows == []


def test_common_evaluation_anchor_overrides_stale_dataset_anchor_for_ap03(
    tmp_path: Path,
) -> None:
    (tmp_path / "evaluations").mkdir(parents=True)
    (tmp_path / "observations").mkdir(parents=True)
    (tmp_path / "evaluations" / "SELECTED_COMMON_EVALUATION.json").write_text(
        json.dumps({"anchor_marker_id": 14}), encoding="utf-8"
    )
    (tmp_path / "observations" / "SELECTION_CANDIDATES.json").write_text(
        json.dumps({"evaluation_anchor": {"selected": 2}}), encoding="utf-8"
    )
    assert ap03_derived._selection_anchor(tmp_path) == 14


def test_derived_evaluations_refresh_without_touching_unrelated_files(tmp_path: Path) -> None:
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
        (destination / "SELECTED_COMMON_EVALUATION.json").read_text(encoding="utf-8")
    )["anchor_marker_id"] == 14
    assert unrelated.read_text(encoding="utf-8") == "immutable"
