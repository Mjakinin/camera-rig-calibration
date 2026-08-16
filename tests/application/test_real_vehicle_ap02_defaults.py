from __future__ import annotations

from camera_rig_calibration.config.models import AP02Settings


def test_real_vehicle_ap02_starts_at_editable_80_80_defaults() -> None:
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
