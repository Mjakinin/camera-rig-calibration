from __future__ import annotations

import json
from pathlib import Path

from camera_rig_calibration.results import ResultEntry
from camera_rig_calibration.wizard import (
    _delete_incomplete_run,
    _interrupt_incomplete_run,
)


def _incomplete_run(root: Path, dataset: str = "fixture") -> ResultEntry:
    transaction = root / "workspace/temporary_runs/queue_001"
    run = transaction / "jobs/ap02/run_001"
    (run / "00_INPUT").mkdir(parents=True)
    (run / "run_manifest.json").write_text(
        json.dumps(
            {
                "dataset_id": dataset,
                "run_id": "run_001",
                "status": "failed",
                "runner_pid": None,
            }
        ),
        encoding="utf-8",
    )
    (transaction / "queue_transaction.json").write_text(
        json.dumps(
            {
                "schema_version": 5,
                "queue_id": "queue_001",
                "status": "incomplete",
            }
        ),
        encoding="utf-8",
    )
    return ResultEntry(dataset, "queue_001", "failed", transaction)


def test_delete_incomplete_transaction_never_deletes_external_inputs_or_configs(
    tmp_path: Path,
) -> None:
    entry = _incomplete_run(tmp_path)
    private_dataset = tmp_path / "datasets/fixture/hash123"
    private_dataset.mkdir(parents=True)
    (private_dataset / "frame.png").write_bytes(b"frame")
    run = entry.path / "jobs/ap02/run_001"
    (run / "00_INPUT/dataset_pointer.json").write_text(
        json.dumps({"dataset_root": str(private_dataset)}), encoding="utf-8"
    )
    workspace_config = tmp_path / "workspace/fixture/rigcal.yaml"
    workspace_config.parent.mkdir(parents=True)
    workspace_config.write_text("fixture", encoding="utf-8")

    removed = _delete_incomplete_run(
        tmp_path, entry, delete_private_inputs=True
    )

    assert entry.path in removed
    assert not entry.path.exists()
    assert private_dataset.exists()
    assert workspace_config.exists()


def test_delete_incomplete_run_never_removes_reused_historical_input(
    tmp_path: Path,
) -> None:
    entry = _incomplete_run(tmp_path)
    historical = tmp_path / "results/bus_real_data/ablation/world/route/route2"
    historical.mkdir(parents=True)
    (historical / "keep.txt").write_text("historical", encoding="utf-8")
    (entry.path / "jobs/ap02/run_001/00_INPUT/dataset_pointer.json").write_text(
        json.dumps({"dataset_root": str(historical)}), encoding="utf-8"
    )

    _delete_incomplete_run(tmp_path, entry, delete_private_inputs=True)

    assert historical.is_dir()
    assert (historical / "keep.txt").read_text(encoding="utf-8") == "historical"


def test_abort_marks_a_stale_running_stage_resumable(tmp_path: Path) -> None:
    entry = _incomplete_run(tmp_path)
    manifest_path = entry.path / "jobs/ap02/run_001/run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(
        {
            "status": "running",
            "runner_pid": None,
            "stages": [{"id": "detect_markers", "status": "running"}],
        }
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    _interrupt_incomplete_run(entry.path)

    updated = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert updated["status"] == "interrupted"
    assert updated["runner_pid"] is None
    assert updated["stages"][0]["status"] == "interrupted"
