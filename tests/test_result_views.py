from __future__ import annotations

import json
from io import StringIO
from pathlib import Path

import typer
from rich.console import Console

from camera_rig_calibration.results import index_results
from camera_rig_calibration.wizard import show_results


def _experiment(root: Path, *, dataset_local: bool = True) -> Path:
    experiment = root / "results/real_vehicle/1Hz/0.5x_1Hz"
    experiment.mkdir(parents=True)
    dataset = experiment
    if dataset_local:
        dataset.mkdir(parents=True, exist_ok=True)
        (dataset / "dataset.json").write_text(
            json.dumps({"layout_version": 2}), encoding="utf-8"
        )
    rows = [
        {
            "method": "ap01",
            "label": "baseline",
            "status": "available",
            "runtime_seconds": 12.5,
            "static_camera_count": 4,
            "primary_result": "baseline",
            "result_path": "methods/ap01/baseline",
            "warning": "",
        },
        {
            "method": "ap02",
            "label": "variant2",
            "status": "failed",
            "runtime_seconds": None,
            "static_camera_count": None,
            "primary_result": None,
            "result_path": "attempts/ap02/variant2/attempt_1",
            "warning": "optimizer failed",
        },
    ]
    summary = {
        "schema_version": 5,
        "layout_version": 2,
        "experiment": "0.5x_1Hz",
        "category": "real_vehicle",
        "sampling_rate": "1Hz",
        "status": "partial",
        "queue_id": "queue",
        "dataset_path": str(dataset.resolve()),
        "methods": rows,
    }
    comparison = {
        "schema_version": 5,
        "layout_version": 2,
        "methods": rows,
    }
    (experiment / "SUMMARY.json").write_text(
        json.dumps(summary), encoding="utf-8"
    )
    (experiment / "COMPARISON.json").write_text(
        json.dumps(comparison), encoding="utf-8"
    )
    (experiment / "RESULTS.txt").write_text(
        "human results", encoding="utf-8"
    )
    return experiment


def test_result_index_reads_only_layout_v2_front_door(tmp_path: Path) -> None:
    experiment = _experiment(tmp_path)
    stale = (
        tmp_path
        / "results/real_vehicle/1Hz/stale/methods/ap02/baseline"
    )
    stale.mkdir(parents=True)
    (stale / "run_manifest.json").write_text("{}", encoding="utf-8")

    entries = index_results(tmp_path / "results")

    assert len(entries) == 1
    assert entries[0].path == experiment
    assert entries[0].status == "partial"
    assert entries[0].dataset_state == "available"
    assert entries[0].methods == ("ap01", "ap02")


def test_missing_dataset_is_reported_not_local(tmp_path: Path) -> None:
    _experiment(tmp_path, dataset_local=False)

    entry = index_results(tmp_path / "results")[0]

    assert entry.dataset_state == "not local"


def test_result_catalogue_renders_variants_paths_and_scope_legend(
    tmp_path: Path, monkeypatch
) -> None:
    _experiment(tmp_path)
    answers = iter((1, 0))
    monkeypatch.setattr(
        typer, "prompt", lambda *args, **kwargs: next(answers)
    )
    output = StringIO()

    show_results(
        tmp_path,
        Console(file=output, force_terminal=False, width=180),
    )

    rendered = output.getvalue()
    assert "Dataset / input" in rendered
    assert "Result status" in rendered
    assert "ap01/baseline" in rendered
    assert "ap02/variant2" in rendered
    assert "RESULTS.txt" in rendered
    assert "COMPARISON.json" in rendered
    assert "Failed attempts:" not in rendered
    assert "moving camera" in rendered
    assert "whole world" in rendered
    assert "pct = percent" in rendered
