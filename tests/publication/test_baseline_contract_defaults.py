from __future__ import annotations

from camera_rig_calibration.config.models import AP02Settings
from camera_rig_calibration.evaluation.reporting_semantics import (
    baseline_contract,
)


def _ap02_payload(
    static_nfev: int,
    combined_nfev: int,
    initialization_strategy: str | None = None,
) -> dict:
    defaults = AP02Settings()
    return {
        "method": "ap02",
        "label": "baseline",
        "config_summary": {
            "reference_marker_selection_mode": "baseline",
            "reference_marker_id": 14,
            "resolved_reference_marker_id": 14,
            "static_max_nfev": static_nfev,
            "combined_max_nfev": combined_nfev,
            "initialization_strategy": (
                initialization_strategy or defaults.initialization_strategy
            ),
            # Kept only to prove the legacy field no longer controls the
            # effective baseline check.
            "initialization_algorithm": "maximum_bottleneck",
        },
    }


def test_route2_ap02_baseline_tracks_current_model_defaults() -> None:
    defaults = AP02Settings()

    contract = baseline_contract(
        category="simulation",
        method_payloads=[
            _ap02_payload(
                defaults.static_only_ba_max_function_evaluations,
                defaults.combined_ba_max_function_evaluations,
            )
        ],
        evaluation_anchor={"selected": 14},
    )

    assert defaults.static_only_ba_max_function_evaluations == 80
    assert defaults.combined_ba_max_function_evaluations == 80
    assert defaults.initialization_strategy == "maximum_frontier_v1"
    assert contract["contract"] == "route2_cpu_ref14_ap02_defaults_v3"
    assert contract["passes"] is True
    checks = contract["variants"][0]["checks"]
    assert checks["static_nfev_default"] is True
    assert checks["combined_nfev_default"] is True
    assert checks["initialization_strategy_default"] is True
    assert "maximum_bottleneck_initialization" not in checks


def test_route2_ap02_nondefault_nfev_is_not_baseline() -> None:
    contract = baseline_contract(
        category="simulation",
        method_payloads=[_ap02_payload(2, 2)],
        evaluation_anchor={"selected": 14},
    )

    assert contract["passes"] is False
    checks = contract["variants"][0]["checks"]
    assert checks["static_nfev_default"] is False
    assert checks["combined_nfev_default"] is False


def test_route2_ap02_nondefault_initializer_is_not_baseline() -> None:
    defaults = AP02Settings()
    contract = baseline_contract(
        category="simulation",
        method_payloads=[
            _ap02_payload(
                defaults.static_only_ba_max_function_evaluations,
                defaults.combined_ba_max_function_evaluations,
                initialization_strategy="wizard_maximum_bottleneck_v2",
            )
        ],
        evaluation_anchor={"selected": 14},
    )

    assert contract["passes"] is False
    assert (
        contract["variants"][0]["checks"]["initialization_strategy_default"]
        is False
    )
