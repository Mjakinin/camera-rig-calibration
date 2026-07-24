from __future__ import annotations

import json
from pathlib import Path

from camera_rig_calibration.results import (
    create_simulation_factor_views,
    index_results,
)


def test_baseline_factor_views_are_relative_symlinks_without_copies(
    tmp_path: Path,
) -> None:
    experiment = tmp_path / "results/simulation/route2"
    experiment.mkdir(parents=True)
    payload = experiment / "frames.bin"
    payload.write_bytes(b"shared")

    created = create_simulation_factor_views(
        tmp_path / "results",
        experiment,
        experiment_id="route2",
        parameters={
            "route": "route2",
            "moving_width": 1280,
            "moving_height": 720,
            "moving_hfov_deg": 69.1,
            "lighting": "baseline",
            "motion_blur_kernel": 0,
        },
    )

    assert created
    assert all(path.is_symlink() for path in created)
    assert all(not path.readlink().is_absolute() for path in created)
    assert (
        tmp_path / "results/simulation/fov/69.1deg"
    ).resolve() == experiment.resolve()
    assert list((tmp_path / "results/simulation/fov").rglob("frames.bin")) == []


def test_capture_timing_change_is_mixed_and_not_baseline(tmp_path: Path) -> None:
    experiment = tmp_path / "results/simulation/settle_050"
    experiment.mkdir(parents=True)

    created = create_simulation_factor_views(
        tmp_path / "results",
        experiment,
        experiment_id="settle_050",
        parameters={"settle_seconds": 0.5},
    )

    assert created == []
    assert not (tmp_path / "results/simulation/fov/69.1deg").exists()


def test_result_index_aggregates_method_children_once_and_ignores_views(
    tmp_path: Path,
) -> None:
    experiment = tmp_path / "results/simulation/density_stride_8_offset4"
    for method in ("ap01", "ap02", "ap03"):
        run = experiment / "methods" / method / "variant/executions/input/current"
        run.mkdir(parents=True)
        (run / "run_manifest.json").write_text(
            json.dumps(
                {
                    "dataset_id": "density_stride_8_offset4",
                    "experiment_id": "density_stride_8_offset4",
                    "result_category": "simulation",
                    "run_id": method,
                    "status": "completed",
                    "enabled_methods": [method],
                }
            ),
            encoding="utf-8",
        )
    view = tmp_path / "results/simulation/_views/mixed/density"
    view.parent.mkdir(parents=True)
    view.symlink_to(
        Path("../../../density_stride_8_offset4"),
        target_is_directory=True,
    )

    entries = index_results(tmp_path / "results")

    matches = [
        entry
        for entry in entries
        if entry.experiment_id == "density_stride_8_offset4"
    ]
    assert len(matches) == 1
    assert set(matches[0].methods) == {"ap01", "ap02", "ap03"}
    assert matches[0].run_id == "3 executions"


def test_result_index_collapses_legacy_ap03_branches_into_combined_ap03(
    tmp_path: Path,
) -> None:
    experiment = tmp_path / "results/simulation/legacy_ap03"
    for method in ("ap01", "ap03_single", "ap03_multi"):
        run = experiment / "methods" / method / "variant/executions/input/current"
        run.mkdir(parents=True)
        (run / "run_manifest.json").write_text(
            json.dumps(
                {
                    "dataset_id": "legacy_ap03",
                    "experiment_id": "legacy_ap03",
                    "result_category": "simulation",
                    "run_id": method,
                    "status": "completed",
                    "enabled_methods": [method],
                }
            ),
            encoding="utf-8",
        )

    entries = index_results(tmp_path / "results")

    match = next(
        entry for entry in entries if entry.experiment_id == "legacy_ap03"
    )
    assert match.methods == ("ap01", "ap03")
