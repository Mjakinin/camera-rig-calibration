from __future__ import annotations

import json
from pathlib import Path

from camera_rig_calibration import migration


def test_migration_materialization_preserves_tree_sha_and_counts(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    (source / "nested").mkdir(parents=True)
    (source / "first.txt").write_text("first\n", encoding="utf-8")
    (source / "nested/second.bin").write_bytes(b"\x00\x01\x02")
    target = tmp_path / "target"

    migration._materialize(source, target)

    assert migration._tree_inventory(source) == migration._tree_inventory(
        target
    )


def test_verified_legacy_cleanup_is_recoverable_and_keeps_incomplete_runs(
    tmp_path: Path, monkeypatch
) -> None:
    results = tmp_path / "results"
    bus = results / "bus_real_data"
    real = results / "real_vehicle_data"
    completed = results / "completed_v1/runs/run_1"
    failed = results / "failed_v1/runs/run_2"
    for root in (bus, real):
        root.mkdir(parents=True)
        (root / "keep.txt").write_text(root.name, encoding="utf-8")
    completed.mkdir(parents=True)
    failed.mkdir(parents=True)
    (completed / "run_manifest.json").write_text(
        json.dumps({"status": "completed"}), encoding="utf-8"
    )
    (failed / "run_manifest.json").write_text(
        json.dumps({"status": "failed"}), encoding="utf-8"
    )
    report = tmp_path / "workspace/migrations/results_v2.json"
    report.parent.mkdir(parents=True)
    report.write_text(
        json.dumps(
            {
                "records": [
                    {"input_verification": {"verified": True}}
                ],
                "legacy_sources_removed": False,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        migration, "_archive_refs_exist", lambda repository: True
    )

    removed = migration.remove_verified_legacy_sources(tmp_path)

    assert set(path.name for path in removed) == {
        "bus_real_data",
        "real_vehicle_data",
        "completed_v1",
    }
    recovery = (
        tmp_path / "workspace/migrations/legacy_sources_v1"
    )
    assert (recovery / "bus_real_data/keep.txt").is_file()
    assert (recovery / "real_vehicle_data/keep.txt").is_file()
    assert (recovery / "completed_v1/runs/run_1").is_dir()
    assert (results / "failed_v1/runs/run_2").is_dir()
