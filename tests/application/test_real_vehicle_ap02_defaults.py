from __future__ import annotations

from camera_rig_calibration import wizard
from camera_rig_calibration.components import register_builtin_components
from camera_rig_calibration.config.models import AP02Settings
from camera_rig_calibration.policies.product_policy import (
    _DATASET_CONTEXT,
    install_product_policy,
)
from camera_rig_calibration.policies.real_ap02_budget_policy import (
    install_real_ap02_budget_policy,
)


def test_real_vehicle_ap02_starts_at_editable_80_80_defaults() -> None:
    # Canonical config contract.
    settings = AP02Settings(
        reference_marker_selection_mode="auto",
        reference_marker_id="auto",
    )
    assert settings.max_nfev_static == 80
    assert settings.max_nfev_moving == 80

    edited = settings.model_copy(
        update={
            "static_only_ba_max_function_evaluations": 12,
            "combined_ba_max_function_evaluations": 34,
        }
    )
    assert edited.max_nfev_static == 12
    assert edited.max_nfev_moving == 34

    # Product-facing rigcal Wizard contract. This is intentionally separate
    # from the raw model assertion because product policies wrap new method
    # jobs before the Method Settings table is rendered.
    install_product_policy()
    install_real_ap02_budget_policy()
    register_builtin_components()
    token = _DATASET_CONTEXT.set("real_vehicle")
    try:
        job = wizard._new_method_job(
            "ap02",
            prompt_for_single_marker=False,
        )
    finally:
        _DATASET_CONTEXT.reset(token)

    assert job.methods.ap02.max_nfev_static == 80
    assert job.methods.ap02.max_nfev_moving == 80

    # Values remain ordinary editable method settings rather than a fixed
    # scientific constant imposed after the user edits the job.
    job.methods = job.methods.model_copy(
        update={
            "ap02": job.methods.ap02.model_copy(
                update={
                    "static_only_ba_max_function_evaluations": 12,
                    "combined_ba_max_function_evaluations": 34,
                }
            )
        },
        deep=True,
    )
    assert job.methods.ap02.max_nfev_static == 12
    assert job.methods.ap02.max_nfev_moving == 34
