from __future__ import annotations

import json
from pathlib import Path

from camera_rig_calibration.results import index_results


def test_new_and_legacy_results_are_indexed_without_mutation(tmp_path: Path) -> None:
    new_run = tmp_path / "dataset_a" / "runs" / "run_01"
    new_run.mkdir(parents=True)
    (new_run / "run_manifest.json").write_text(
        json.dumps(
            {
                "dataset_id": "dataset_a",
                "run_id": "run_01",
                "status": "completed",
                "enabled_methods": ["ap02"],
            }
        )
    )
    legacy = tmp_path / "legacy_dataset" / "03_ap02_real"
    legacy.mkdir(parents=True)
    status = legacy / "METHOD_STATUS.json"
    original = '{"method":"AP02","status":"OK_FULL"}\n'
    status.write_text(original)
    report_only = tmp_path / "report_only" / "99_FINAL_RESULTS" / "marker_consistency"
    report_only.mkdir(parents=True)
    (report_only / "REAL_DATA_MARKER_CONSISTENCY_SUMMARY.json").write_text(
        '[{"method":"AP03","status":"OK"}]\n'
    )
    migrated = tmp_path / "real_vehicle/historical_missing"
    migrated.mkdir(parents=True)
    (migrated / "legacy_manifest.json").write_text(
        json.dumps(
            {
                "category": "real_vehicle",
                "experiment_id": "historical_missing",
                "input_id": "unavailable_123",
                "status": "input unavailable / not rerunnable",
            }
        ),
        encoding="utf-8",
    )
    entries = index_results(tmp_path)
    assert any(entry.run_id == "run_01" and not entry.legacy for entry in entries)
    assert any(entry.dataset_id == "legacy_dataset" and entry.legacy for entry in entries)
    assert any(
        entry.dataset_id == "report_only"
        and entry.legacy
        and entry.methods == ("AP03",)
        for entry in entries
    )
    assert status.read_text() == original
    migrated_entry = next(
        entry
        for entry in entries
        if entry.experiment_id == "historical_missing"
    )
    assert migrated_entry.category == "real_vehicle"
    assert migrated_entry.status == "input unavailable / not rerunnable"
