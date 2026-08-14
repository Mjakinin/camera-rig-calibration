from __future__ import annotations

import csv
import json
from pathlib import Path

from camera_rig_calibration.policies.ap02_convergence_reporting_policy import (
    _representative_points,
    _tail_window,
    analyze_ap02_convergence,
)


def test_tail_window_reports_nfev_70_to_80_improvement() -> None:
    rows = [
        {
            "residual_evaluation_index": str(index),
            "source": "residual",
            "solver_iteration": str(index),
            "solver_nfev": str(nfev),
            "elapsed_seconds": str(nfev),
            "robust_cost": str(cost),
            "reprojection_rmse_px": str(rmse),
            "mean_reprojection_error_px": "1.0",
            "maximum_reprojection_error_px": "2.0",
            "parameter_step_norm": "0.1",
        }
        for index, (nfev, cost, rmse) in enumerate(
            [
                (1, 1000.0, 30.0),
                (60, 300.0, 18.0),
                (70, 200.0, 15.0),
                # Two residual evaluations share nfev=80. The lower robust cost
                # is the conservative best-observed representative.
                (80, 190.0, 14.5),
                (80, 150.0, 13.0),
            ]
        )
    ]
    points = _representative_points(rows)
    window = _tail_window(points, 80, 10)
    assert window is not None
    assert window["start_nfev"] == 70
    assert window["end_nfev"] == 80
    assert window["start_best_observed_cost"] == 200.0
    assert window["end_best_observed_cost"] == 150.0
    assert window["absolute_cost_improvement"] == 50.0
    assert window["relative_cost_improvement_from_window_start"] == 0.25
    assert window["start_best_observed_reprojection_rmse_px"] == 15.0
    assert window["end_best_observed_reprojection_rmse_px"] == 13.0


def _write_history(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "residual_evaluation_index",
        "source",
        "solver_iteration",
        "solver_nfev",
        "elapsed_seconds",
        "robust_cost",
        "reprojection_rmse_px",
        "mean_reprojection_error_px",
        "maximum_reprojection_error_px",
        "parameter_step_norm",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_analyze_ap02_convergence_distinguishes_converged_and_budget_limit(
    tmp_path: Path,
) -> None:
    root = tmp_path / "ap02"
    graph = root / "diagnostics" / "method" / "graph_ba"
    for stage, nfev, maximum, success, message in (
        ("static_only", 20, 80, True, "`ftol` termination condition is satisfied."),
        (
            "with_moving",
            80,
            80,
            False,
            "The maximum number of function evaluations is exceeded.",
        ),
    ):
        stage_root = graph / stage
        stage_root.mkdir(parents=True, exist_ok=True)
        (stage_root / "ap02_optimization_summary.json").write_text(
            json.dumps(
                {
                    "solver_success": success,
                    "solver_message": message,
                    "nfev": nfev,
                    "maximum_function_evaluations": maximum,
                    "initial_cost": 1000.0,
                    "final_cost": 100.0,
                    "initial_reprojection_rmse_px": 30.0,
                    "final_reprojection_rmse_px": 10.0,
                    "optimality": 1.0,
                }
            ),
            encoding="utf-8",
        )
        _write_history(
            stage_root / "ap02_optimization_history.csv",
            [
                {
                    "residual_evaluation_index": 0,
                    "source": "residual",
                    "solver_iteration": 0,
                    "solver_nfev": 1,
                    "elapsed_seconds": 0.0,
                    "robust_cost": 1000.0,
                    "reprojection_rmse_px": 30.0,
                    "mean_reprojection_error_px": 20.0,
                    "maximum_reprojection_error_px": 50.0,
                    "parameter_step_norm": 0.0,
                },
                {
                    "residual_evaluation_index": 1,
                    "source": "residual",
                    "solver_iteration": 1,
                    "solver_nfev": max(1, nfev - 10),
                    "elapsed_seconds": 1.0,
                    "robust_cost": 150.0,
                    "reprojection_rmse_px": 12.0,
                    "mean_reprojection_error_px": 8.0,
                    "maximum_reprojection_error_px": 20.0,
                    "parameter_step_norm": 0.1,
                },
                {
                    "residual_evaluation_index": 2,
                    "source": "residual",
                    "solver_iteration": 2,
                    "solver_nfev": nfev,
                    "elapsed_seconds": 2.0,
                    "robust_cost": 100.0,
                    "reprojection_rmse_px": 10.0,
                    "mean_reprojection_error_px": 7.0,
                    "maximum_reprojection_error_px": 18.0,
                    "parameter_step_norm": 0.05,
                },
            ],
        )

    analysis = analyze_ap02_convergence(root)
    static = analysis["stages"]["static_only"]
    combined = analysis["stages"]["with_moving"]
    assert static["termination"] == "converged"
    assert static["nfev"] == 20
    assert combined["termination"] == "function_evaluation_limit_reached"
    assert combined["nfev"] == 80
    assert combined["tail_windows"]["10"]["start_nfev"] == 70
    assert combined["tail_windows"]["10"]["end_nfev"] == 80
