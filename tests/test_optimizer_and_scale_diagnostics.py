from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from camera_rig_calibration.methods.ap03.scale_core import (
    select_observations_per_marker,
)


def test_ap02_records_residual_calls_without_claiming_solver_iterations(
    monkeypatch, tmp_path: Path
) -> None:
    try:
        __import__("scipy.optimize")
    except ImportError:
        pytest.skip(
            "the AP02 optimizer contract requires a working SciPy build"
        )
    from camera_rig_calibration.methods.ap02 import optimize

    output = tmp_path / "ap02"

    def fake_least_squares(function, initial, **_kwargs):
        function(np.asarray(initial, dtype=np.float64))
        final_parameters = np.asarray(initial, dtype=np.float64) + 0.25
        final_residuals = function(final_parameters)
        return SimpleNamespace(
            success=True,
            status=1,
            message="synthetic convergence",
            nfev=2,
            njev=1,
            cost=0.5 * float(np.sum(final_residuals**2)),
            fun=final_residuals,
            optimality=0.01,
        )

    monkeypatch.setattr(optimize.core, "least_squares", fake_least_squares)

    def fake_run_ba(
        mode,
        reference_marker_id,
        maximum_function_evaluations,
        *,
        ap02_root,
        observations_csv,
        initialization_root,
    ):
        del (
            reference_marker_id,
            maximum_function_evaluations,
            observations_csv,
            initialization_root,
        )
        (ap02_root / "07_graph_ba" / mode).mkdir(parents=True)
        optimize.core.least_squares(
            lambda values: np.asarray(
                [
                    values[0] - 1.0,
                    values[1] - 2.0,
                    values[0] + 0.5,
                    values[1] + 0.25,
                ],
                dtype=np.float64,
            ),
            np.zeros(2, dtype=np.float64),
            max_nfev=5,
        )

    monkeypatch.setattr(optimize.core, "run_ba", fake_run_ba)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "ap02-optimize",
            "--mode",
            "with_moving",
            "--ref-marker-id",
            "7",
            "--max-nfev",
            "5",
            "--ap02-root",
            str(output),
            "--observations",
            str(tmp_path / "observations.csv"),
            "--initialization-root",
            str(tmp_path / "initialization"),
            "--robust-loss",
            "linear",
            "--robust-loss-scale-px",
            "3",
        ],
    )

    optimize.main()

    result_root = output / "07_graph_ba" / "with_moving"
    summary = json.loads(
        (result_root / "ap02_optimization_summary.json").read_text(
            encoding="utf-8"
        )
    )
    assert summary["nfev"] == 2
    assert summary["njev"] == 1
    assert summary["residual_evaluation_calls_recorded"] == 3
    assert summary["initial_cost"] > summary["final_cost"]
    assert summary["scalar_residual_count"] == 4
    with (result_root / "ap02_optimization_history.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        history = list(csv.DictReader(handle))
    assert [int(row["residual_evaluation_index"]) for row in history] == [
        1,
        2,
        3,
    ]
    assert all(row["solver_iteration"] == "" for row in history)
    assert all(row["solver_nfev"] == "" for row in history)


def test_ap03_observation_cap_is_quality_ranked_per_marker() -> None:
    observations: list[dict[str, object]] = []
    for marker_id, scores in ((1, (0.2, 0.9, 0.7)), (2, (0.8, 0.1))):
        for image_index, score in enumerate(scores):
            for corner_index in range(4):
                observations.append(
                    {
                        "marker_id": marker_id,
                        "image_name": (
                            f"marker_{marker_id}_image_{image_index}.png"
                        ),
                        "corner_idx": corner_index,
                        "selection_score": score,
                        "area_px2": 100.0 + score,
                    }
                )

    selected, diagnostics = select_observations_per_marker(
        observations, maximum_observations_per_marker=2
    )

    selected_images = {
        (int(row["marker_id"]), str(row["image_name"]))
        for row in selected
    }
    assert selected_images == {
        (1, "marker_1_image_1.png"),
        (1, "marker_1_image_2.png"),
        (2, "marker_2_image_0.png"),
        (2, "marker_2_image_1.png"),
    }
    marker_one = [
        row for row in diagnostics if row["marker_id"] == 1
    ]
    assert [row["quality_rank"] for row in marker_one] == [1, 2, 3]
    assert [row["selected"] for row in marker_one] == [True, True, False]
