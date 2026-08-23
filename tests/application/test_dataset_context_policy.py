from __future__ import annotations

import typer

from camera_rig_calibration.components import register_builtin_components
from camera_rig_calibration.policies.dataset_context_policy import (
    install_dataset_context_policy,
)
from camera_rig_calibration.policies.marker_preference_policy import (
    install_marker_preference_policy,
)
from camera_rig_calibration.policies.product_policy import (
    _DATASET_CONTEXT,
    install_product_policy,
)
from camera_rig_calibration.policies.submission_policy import (
    install_submission_policy,
)


from camera_rig_calibration.application.bootstrap import _resolve_wizard_policy_target

# Keep this policy test isolated from unrelated product layers. Installing the
# complete bootstrap stack here would also install AP03 sensitivity/UI wrappers
# globally during pytest collection and make later tests order-dependent.
_resolve_wizard_policy_target()
install_product_policy()
install_submission_policy()
install_marker_preference_policy()
install_dataset_context_policy()

from camera_rig_calibration import wizard  # noqa: E402


def _choose_input_type(monkeypatch, value: str) -> None:
    monkeypatch.setattr(
        typer,
        "prompt",
        lambda *args, **kwargs: value,
    )
    selected = wizard._choice(
        "Input type",
        {
            "1": "real data",
            "2": "Gazebo simulation",
            "0": "back",
        },
        "1",
    )
    assert selected == value


def _job(method_id: str):
    register_builtin_components()
    return wizard._new_method_job(
        method_id,
        prompt_for_single_marker=False,
    )


def test_simulation_input_choice_applies_editable_simulation_defaults(
    monkeypatch,
) -> None:
    token = _DATASET_CONTEXT.set("real_vehicle")
    try:
        _choose_input_type(monkeypatch, "2")
        ap01 = _job("ap01")
        ap02 = _job("ap02")

        assert ap01.methods.ap01.root_camera == "cam_edge_3"
        assert ap02.methods.ap02.reference_marker_selection_mode == "baseline"
        assert ap02.methods.ap02.reference_marker_id == 14
        assert ap02.evaluation.anchor_marker_id == 14

        editable = {row[0] for row in wizard._setting_rows(ap02)}
        assert "ap02_reference_mode" in editable
        assert "evaluation_anchor" in editable
    finally:
        _DATASET_CONTEXT.reset(token)


def test_real_input_choice_applies_editable_real_vehicle_defaults(
    monkeypatch,
) -> None:
    token = _DATASET_CONTEXT.set("simulation")
    try:
        _choose_input_type(monkeypatch, "1")
        ap01 = _job("ap01")
        ap02 = _job("ap02")

        assert ap01.methods.ap01.root_camera == "auto"
        assert ap02.methods.ap02.reference_marker_id == 0
        assert ap02.evaluation.anchor_marker_id == 0

        editable = {row[0] for row in wizard._setting_rows(ap02)}
        assert "ap02_reference_mode" in editable
        assert "evaluation_anchor" in editable
    finally:
        _DATASET_CONTEXT.reset(token)
