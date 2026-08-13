from __future__ import annotations

import camera_rig_calibration.bootstrap as bootstrap


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
    "install_real_partial_evaluation_policy",
    "install_ap02_partial_reference_reporting_policy",
    "install_real_marker_reporting_policy",
    "install_final_reporting_frontdoor_policy",
    "install_ap02_convergence_frontdoor_policy",
    "install_rviz_manifest_policy",
    "install_rviz_method_selection_policy",
    "install_submission_bindings",
    "install_ui_display_policy",
    "install_result_view_policy",
)


def test_product_stack_preserves_install_order_and_is_idempotent(monkeypatch) -> None:
    calls: list[str] = []

    for name in INSTALLERS:
        monkeypatch.setattr(
            bootstrap,
            name,
            lambda name=name: calls.append(name),
        )

    monkeypatch.setattr(bootstrap, "_INSTALLED", False)

    bootstrap.install_product_stack()
    bootstrap.install_product_stack()

    assert calls == list(INSTALLERS)
    assert bootstrap._INSTALLED is True
