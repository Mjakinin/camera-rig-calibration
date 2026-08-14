from __future__ import annotations

from camera_rig_calibration.policies.ap02_convergence_frontdoor_policy import _append_if_needed


def test_ap02_convergence_is_added_to_experiment_frontdoor() -> None:
    method_payloads = [
        {
            "method": "ap02",
            "label": "ref_marker_0__ref_mode_auto",
            "metrics": {
                "ap02_convergence_stages": {
                    "static_only": {
                        "stage": "static_only",
                        "termination": "converged",
                        "nfev": 20,
                        "maximum_function_evaluations": 80,
                        "tail_windows": {},
                    },
                    "with_moving": {
                        "stage": "with_moving",
                        "termination": "function_evaluation_limit_reached",
                        "nfev": 80,
                        "maximum_function_evaluations": 80,
                        "tail_windows": {
                            "10": {
                                "start_nfev": 70,
                                "end_nfev": 80,
                                "start_best_observed_cost": 200.0,
                                "end_best_observed_cost": 150.0,
                                "relative_cost_improvement_from_window_start": 0.25,
                                "fraction_of_total_observed_cost_improvement_in_window": 0.05,
                                "start_best_observed_reprojection_rmse_px": 15.0,
                                "end_best_observed_reprojection_rmse_px": 13.0,
                            }
                        },
                    },
                }
            },
        }
    ]
    text, payload = _append_if_needed("BASE RESULTS\n", {}, method_payloads)
    assert "AP02 OPTIMIZATION CONVERGENCE" in text
    assert "last ~10 nfev (70->80)" in text
    assert payload["ap02_optimization_convergence"]["ground_truth_used"] is False


def test_ap02_convergence_is_not_duplicated() -> None:
    text = "AP02 OPTIMIZATION CONVERGENCE\nexisting\n"
    method_payloads = [
        {
            "method": "ap02",
            "metrics": {"ap02_convergence_stages": {"with_moving": {"stage": "with_moving"}}},
        }
    ]
    updated, _ = _append_if_needed(text, {}, method_payloads)
    assert updated == text
