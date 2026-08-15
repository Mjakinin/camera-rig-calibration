from __future__ import annotations

import subprocess
import sys
import textwrap

from camera_rig_calibration.application import bootstrap


INSTALLERS = (
    "install_product_policy",
    "install_real_ap02_budget_policy",
    "install_reporting_authority_policy",
    "install_submission_policy",
    "install_marker_preference_policy",
    "install_common_anchor_authority_policy",
    "install_real_vehicle_marker_zero_policy",
    "install_queue_anchor_preference_policy",
    "install_ap01_common_anchor_policy",
    "install_ap03_camera_model_sensitivity_policy",
    "install_ap02_convergence_reporting_policy",
    "install_result_output_policy",
    "install_submission_quality_policy",
    "install_real_partial_evaluation_policy",
    "install_ap02_partial_reference_reporting_policy",
    "install_real_marker_reporting_policy",
    "install_final_reporting_frontdoor_policy",
    "install_ap02_convergence_frontdoor_policy",
    "install_rviz_manifest_policy",
    "install_rviz_method_selection_policy",
    "install_submission_bindings",
    "install_ui_display_policy",
    "install_dataset_context_policy",
    "install_simulation_ui_consistency_policy",
    "install_result_view_policy",
)


def test_product_stack_preserves_install_order_and_is_idempotent(monkeypatch) -> None:
    calls: list[str] = []

    monkeypatch.setattr(
        bootstrap,
        "_resolve_wizard_policy_target",
        lambda: calls.append("_resolve_wizard_policy_target"),
    )
    for name in INSTALLERS:
        monkeypatch.setattr(
            bootstrap,
            name,
            lambda name=name: calls.append(name),
        )

    monkeypatch.setattr(bootstrap, "_INSTALLED", False)

    bootstrap.install_product_stack()
    bootstrap.install_product_stack()

    assert calls == ["_resolve_wizard_policy_target", *INSTALLERS]
    assert bootstrap._INSTALLED is True


def test_product_stack_defaults_reach_canonical_wizard_bindings() -> None:
    script = textwrap.dedent(
        """
        from camera_rig_calibration.application.bootstrap import install_product_stack
        from camera_rig_calibration.components import register_builtin_components
        from camera_rig_calibration.policies.product_policy import _DATASET_CONTEXT
        from camera_rig_calibration.ui.wizard_bindings import current_wizard_bindings

        register_builtin_components()
        install_product_stack()
        hooks = current_wizard_bindings()

        token = _DATASET_CONTEXT.set("simulation")
        try:
            ap01 = hooks.new_method_job("ap01", prompt_for_single_marker=False)
            ap02 = hooks.new_method_job("ap02", prompt_for_single_marker=False)
            assert ap01.methods.ap01.root_camera == "cam_edge_3"
            assert ap02.methods.ap02.reference_marker_id == 14
            assert ap02.evaluation.anchor_marker_id == 14
        finally:
            _DATASET_CONTEXT.reset(token)

        token = _DATASET_CONTEXT.set("real_vehicle")
        try:
            ap02 = hooks.new_method_job("ap02", prompt_for_single_marker=False)
            assert ap02.methods.ap02.reference_marker_id == 0
            assert ap02.evaluation.anchor_marker_id == 0
        finally:
            _DATASET_CONTEXT.reset(token)
        """
    )
    subprocess.run([sys.executable, "-c", script], check=True)
