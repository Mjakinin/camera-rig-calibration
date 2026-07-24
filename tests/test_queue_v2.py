from __future__ import annotations

from pathlib import Path
import json
from types import SimpleNamespace

import pytest
import yaml

from camera_rig_calibration.config import save_config
from camera_rig_calibration.config.models import EvaluationSettings
from camera_rig_calibration.queueing import (
    QueueConfig,
    QueueEntry,
    QueueRunner,
    load_queue_partitions,
    save_queue,
)


def test_schema_v4_queue_rejects_multiple_datasets_and_v2_load_partitions_them(
    prepared_config, tmp_path: Path
) -> None:
    first = save_config(prepared_config, tmp_path / "configs" / "first.yaml")
    second_config = prepared_config.model_copy(
        update={
            "dataset": prepared_config.dataset.model_copy(
                update={"id": "second_experiment"}
            )
        },
        deep=True,
    )
    second = save_config(second_config, tmp_path / "configs" / "second.yaml")
    destination = tmp_path / "queues" / "overnight.yaml"
    with pytest.raises(ValueError, match="exactly one dataset"):
        save_queue(
            "overnight",
            [("first", first), ("second", second)],
            destination,
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        yaml.safe_dump(
            {
                "kind": "rigcal_queue",
                "schema_version": 2,
                "id": "legacy_overnight",
                "entries": [
                    {"id": "first", "config": str(first)},
                    {"id": "second", "config": str(second)},
                ],
            }
        ),
        encoding="utf-8",
    )

    partitions = load_queue_partitions(destination)

    assert [queue.id for queue in partitions] == [
        "legacy_overnight__01_generic_fixture",
        "legacy_overnight__02_second_experiment",
    ]
    assert [[entry.id for entry in queue.entries] for queue in partitions] == [
        ["first"],
        ["second"],
    ]
    assert all(
        entry.config.is_absolute()
        for queue in partitions
        for entry in queue.entries
    )


def test_queue_rejects_forward_or_unknown_dependencies() -> None:
    with pytest.raises(ValueError, match="dependencies"):
        QueueConfig.model_validate(
            {
                "kind": "rigcal_queue",
                "schema_version": 2,
                "id": "bad",
                "entries": [
                    {
                        "id": "first",
                        "config": "first.yaml",
                        "depends_on": ["missing"],
                    }
                ],
            }
        )


def test_explicit_evaluation_anchor_runs_evaluator_only_on_existing_methods(
    prepared_config, tmp_path: Path, monkeypatch
) -> None:
    experiment = tmp_path / "results/real_vehicle/paper"
    observations = experiment / "observations/input_1/detection_1"
    observations.mkdir(parents=True)
    (observations / "SELECTION_CANDIDATES.json").write_text(
        json.dumps(
            {
                "evaluation_anchor": {
                    "observation_candidates": [7, 9]
                },
                "ap03_single_scale_marker": {
                    "candidates": [
                        {
                            "id": 7,
                            "moving_frames": 1,
                            "static_camera_count": 1,
                            "moving_median_pnp_reprojection_rmse_px": 1.0,
                            "moving_median_marker_area_px2": 100.0,
                        },
                        {
                            "id": 9,
                            "moving_frames": 2,
                            "static_camera_count": 1,
                            "moving_median_pnp_reprojection_rmse_px": 1.0,
                            "moving_median_marker_area_px2": 100.0,
                        },
                    ]
                },
            }
        ),
        encoding="utf-8",
    )
    dataset = prepared_config.dataset.prepared_root
    assert dataset is not None
    result = (
        experiment
        / "methods/ap02/ref_marker_7/executions/input_1/current"
    )
    (result / "00_INPUT").mkdir(parents=True)
    (result / "03_AP02").mkdir()
    (result / "00_INPUT/dataset_pointer.json").write_text(
        json.dumps({"dataset_root": str(dataset)}),
        encoding="utf-8",
    )
    (result / "run_manifest.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "experiment_root": str(experiment),
                "input_id": "input_1",
                "observations_root": str(observations),
                "method_id": "ap02",
                "variant": "ref_marker_7",
                "method_fingerprint": "method_sha",
            }
        ),
        encoding="utf-8",
    )
    config = prepared_config.model_copy(
        update={
            "evaluation": EvaluationSettings(anchor_marker_id=9)
        },
        deep=True,
    )
    config_path = save_config(config, tmp_path / "ap02.yaml")
    save_config(config, result / "resolved_config.yaml")
    calls: list[list[str]] = []

    def fake_run(argv, **kwargs):
        calls.append(list(argv))
        output = Path(argv[argv.index("--output-root") + 1])
        summary = (
            output
            / "marker_consistency"
            / "REAL_DATA_MARKER_CONSISTENCY_SUMMARY.json"
        )
        summary.parent.mkdir(parents=True)
        summary.write_text(
            json.dumps([{"method": "AP02", "status": "OK"}]),
            encoding="utf-8",
        )
        return SimpleNamespace(returncode=0, stdout="evaluation fixture")

    monkeypatch.setattr(
        "camera_rig_calibration.queueing.subprocess.run", fake_run
    )
    queue = QueueConfig(
        id="evaluation_only",
        entries=[QueueEntry(id="ap02", config=config_path)],
    )

    QueueRunner(tmp_path)._run_common_evaluations(
        queue,
        {
            "ap02": {
                "status": "duplicate_skipped",
                "result": str(result),
            }
        },
        [config],
    )

    assert len(calls) == 1
    argv = calls[0]
    assert argv[argv.index("--anchor-marker-id") + 1] == "9"
    selected = json.loads(
        (
            experiment
            / "evaluations/SELECTED_COMMON_EVALUATION.json"
        ).read_text(encoding="utf-8")
    )
    assert selected["anchor_marker_id"] == 9
