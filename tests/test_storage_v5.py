from __future__ import annotations

import json
from pathlib import Path

import pytest

from camera_rig_calibration.config import save_config
from camera_rig_calibration.config.models import (
    DatasetCategory,
    InputSourceKind,
)
from camera_rig_calibration.experiments import experiment_paths
from camera_rig_calibration.publication import publish_queue_transaction


@pytest.mark.parametrize(
    ("source", "sampling", "expected"),
    [
        (InputSourceKind.VIDEO, 3.0, "video/3hz/generic_fixture"),
        (InputSourceKind.FRAMES, None, "frames/unknown_hz/generic_fixture"),
        (InputSourceKind.ROSBAG, None, "rosbag/stored_rate/generic_fixture"),
        (
            InputSourceKind.PREPARED,
            None,
            "prepared/unknown_hz/generic_fixture",
        ),
    ],
)
def test_real_vehicle_storage_is_grouped_by_source_then_sampling(
    prepared_config,
    source: InputSourceKind,
    sampling: float | None,
    expected: str,
) -> None:
    config = prepared_config.model_copy(
        update={
            "dataset": prepared_config.dataset.model_copy(
                update={"source_kind": source}
            ),
            "sampling": prepared_config.sampling.model_copy(
                update={"target_hz": sampling}
            ),
        },
        deep=True,
    )

    paths = experiment_paths(config)

    assert paths.root == (
        config.project.output_root / "real_vehicle" / expected
    )
    assert paths.dataset_root == (
        config.project.dataset_cache_root / "real_vehicle" / expected
    )


def _simulation_config(prepared_config, **changes):
    simulation = prepared_config.simulation.model_copy(
        update={
            "world_id": "bus",
            "route_name": "route2",
            "target_route_frames": 189,
            **changes,
        },
        deep=True,
    )
    return prepared_config.model_copy(
        update={
            "dataset": prepared_config.dataset.model_copy(
                update={"category": DatasetCategory.SIMULATION}
            ),
            "simulation": simulation,
        },
        deep=True,
    )


def test_simulation_storage_classifies_baseline_single_factor_and_mixed(
    prepared_config,
) -> None:
    baseline = _simulation_config(prepared_config)
    fov = _simulation_config(prepared_config, moving_hfov_deg=100.0)
    mixed = _simulation_config(
        prepared_config,
        moving_hfov_deg=100.0,
        motion_blur_kernel=11,
    )

    assert experiment_paths(baseline).root == (
        baseline.project.output_root / "simulation/baseline/route2"
    )
    assert experiment_paths(fov).root == (
        fov.project.output_root / "simulation/fov/100deg"
    )
    mixed_relative = experiment_paths(mixed).root.relative_to(
        mixed.project.output_root / "simulation"
    )
    assert mixed_relative.parts[0] == "mixed"
    assert mixed_relative.name.startswith("generic_fixture_")


def test_custom_world_uses_its_own_factor_namespace(prepared_config) -> None:
    config = _simulation_config(
        prepared_config,
        world_id="warehouse",
        route_name="survey",
        world_baseline={
            "route_name": "survey",
            "target_route_frames": 189,
        },
    )

    assert experiment_paths(config).root == (
        config.project.output_root
        / "simulation/worlds/warehouse/baseline/survey"
    )


def test_alternative_intrinsics_does_not_move_real_experiment(
    prepared_config,
) -> None:
    first = experiment_paths(prepared_config)
    alternative = prepared_config.model_copy(
        update={
            "moving_camera": prepared_config.moving_camera.model_copy(
                update={"intrinsics_profile": "alternative@abcdef12"}
            )
        },
        deep=True,
    )

    assert experiment_paths(alternative).root == first.root
    assert experiment_paths(alternative).dataset_root == first.dataset_root


def test_transaction_is_invisible_until_atomic_publication(
    prepared_config,
) -> None:
    config = prepared_config
    transaction = (
        config.project.workspace_root / "temporary_runs/queue_fixture"
    )
    input_id = "input_fixture"
    dataset_input = transaction / "dataset/inputs" / input_id
    (dataset_input / "raw_images/moving").mkdir(parents=True)
    (dataset_input / "raw_images/moving/frame.png").write_bytes(b"frame")
    (dataset_input / "SOURCE.json").write_text("{}")

    execution = transaction / "jobs/ap02/completed"
    (execution / "00_INPUT").mkdir(parents=True)
    (execution / "99_FINAL_RESULTS").mkdir()
    (execution / "99_FINAL_RESULTS/SUMMARY.txt").write_text("complete")
    paths = experiment_paths(config)
    target = (
        paths.methods
        / "ap02/ref_marker_3__baseline_fixture"
        / "executions"
        / input_id
        / "current"
    )
    manifest = {
        "schema_version": 5,
        "run_id": "fixture",
        "dataset_id": config.dataset.id,
        "experiment_id": paths.experiment_id,
        "result_category": paths.category,
        "status": "completed",
        "enabled_methods": ["ap02"],
        "input_id": input_id,
        "observation_id": "",
        "intended_result_target": str(target),
    }
    (execution / "run_manifest.json").write_text(json.dumps(manifest))
    save_config(config, execution / "requested_config.yaml")
    save_config(config, execution / "resolved_config.yaml")

    assert not (paths.root / "PUBLISHED.json").exists()
    results = publish_queue_transaction(
        transaction,
        queue_id="queue_fixture",
        configs=[config],
        results={"ap02": {"status": "completed", "result": str(execution)}},
    )

    assert results["ap02"]["result"] == str(target)
    assert (paths.root / "PUBLISHED.json").is_file()
    assert (
        paths.datasets / input_id / "raw_images/moving/frame.png"
    ).read_bytes() == b"frame"
    assert (target / "99_FINAL_RESULTS/SUMMARY.txt").read_text() == "complete"


def test_failed_transaction_never_creates_published_experiment(
    prepared_config,
) -> None:
    transaction = (
        prepared_config.project.workspace_root
        / "temporary_runs/failed_queue"
    )

    publish_queue_transaction(
        transaction,
        queue_id="failed_queue",
        configs=[prepared_config],
        results={"ap02": {"status": "failed"}},
    )

    assert not (
        experiment_paths(prepared_config).root / "PUBLISHED.json"
    ).exists()
    journal = json.loads(
        (transaction / "queue_transaction.json").read_text()
    )
    assert journal["status"] == "incomplete"
