from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from camera_rig_calibration.config import save_config
from camera_rig_calibration.config.models import (
    DatasetCategory,
    InputSourceKind,
)
from camera_rig_calibration.experiments import (
    experiment_fingerprint,
    experiment_paths,
    write_experiment_manifest,
)
from camera_rig_calibration.publication import (
    _publish_success,
    _rename_with_retry,
    publish_preparation_transaction,
    publish_queue_transaction,
)


@pytest.mark.parametrize(
    ("source", "sampling", "expected"),
    [
        (InputSourceKind.VIDEO, 3.0, "3Hz/generic_fixture"),
        (InputSourceKind.FRAMES, None, "native_rate/generic_fixture"),
        (InputSourceKind.ROSBAG, None, "native_rate/generic_fixture"),
        (InputSourceKind.PREPARED, None, "native_rate/generic_fixture"),
    ],
)
def test_real_vehicle_storage_ignores_source_kind(
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

    assert paths.root == config.project.output_root / "real_vehicle" / expected
    assert paths.dataset_root == paths.root


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


def test_foreign_world_id_is_rejected(prepared_config) -> None:
    config = _simulation_config(
        prepared_config,
        world_id="warehouse",
        route_name="survey",
        world_baseline={
            "route_name": "survey",
            "target_route_frames": 189,
        },
    )

    with pytest.raises(
        ValueError,
        match="only the built-in bus Gazebo world is supported",
    ):
        experiment_paths(config)


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


def _transaction_dataset(transaction: Path, fingerprint: str = "input_fixture") -> None:
    dataset = transaction / "dataset"
    for name in ("static", "moving", "camera_info"):
        (dataset / "raw_images" / name).mkdir(parents=True, exist_ok=True)
    (dataset / "raw_images/static/front-left.png").write_bytes(b"static")
    (dataset / "raw_images/moving/frame.png").write_bytes(b"moving")
    (dataset / "raw_images/camera_info/front-left.json").write_text(
        "{}", encoding="utf-8"
    )
    observations = dataset / "observations"
    observations.mkdir()
    for name in (
        "shared_static_aruco_observations.csv",
        "shared_moving_aruco_observations.csv",
        "shared_all_aruco_observations.csv",
    ):
        (observations / name).write_text("frame\n", encoding="utf-8")
    for name in ("SELECTION_CANDIDATES.json", "REFERENCE_SELECTIONS.json"):
        (observations / name).write_text("{}", encoding="utf-8")
    (observations / "SELECTION_CANDIDATES.csv").write_text(
        "selection,rank\n", encoding="utf-8"
    )
    (observations / "REFERENCE_MARKER_ID.txt").write_text(
        "3\n", encoding="utf-8"
    )
    (observations / "PUBLICATION_COMPLETE.json").write_text(
        json.dumps({"status": "complete"}), encoding="utf-8"
    )
    quality = observations / "quality"
    quality.mkdir()
    (quality / "marker_inventory.csv").write_text(
        "marker_id\n", encoding="utf-8"
    )
    (quality / "marker_inventory.json").write_text(
        "[]\n", encoding="utf-8"
    )
    (observations / "debug_images").mkdir()
    (dataset / "metadata").mkdir()
    (dataset / "metadata/source.json").write_text(
        json.dumps(
            {
                "input_id": fingerprint,
                "canonical_source_roots": [
                    str(transaction / "input_working")
                ],
            }
        ),
        encoding="utf-8",
    )
    (dataset / "dataset.json").write_text(
        json.dumps(
            {
                "schema_version": 5,
                "layout_version": 2,
                "input_fingerprint": fingerprint,
            }
        ),
        encoding="utf-8",
    )


def _successful_ap02_execution(
    transaction: Path,
    config,
    *,
    fingerprint: str = "method_fixture",
) -> Path:
    execution = transaction / "jobs/ap02/completed"
    method = execution / "03_AP02"
    poses = (
        method
        / "07_graph_ba/with_moving/"
        "optimized_static_camera_poses_ref_marker.csv"
    )
    poses.parent.mkdir(parents=True)
    poses.write_text("camera_id,tx\nfront-left,0\n", encoding="utf-8")
    (method / "METHOD_STATUS.json").write_text(
        json.dumps(
            {
                "primary_result": "combined",
                "reference_marker_id": 3,
                "available_static_cameras": ["front-left"],
            }
        ),
        encoding="utf-8",
    )
    (execution / "logs").mkdir()
    (execution / "logs/ap02.log").write_text("complete\n", encoding="utf-8")
    (execution / "run_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 5,
                "run_id": "fixture",
                "status": "completed",
                "method_id": "ap02",
                "enabled_methods": ["ap02"],
                "input_id": "input_fixture",
                "method_fingerprint": fingerprint,
            }
        ),
        encoding="utf-8",
    )
    save_config(config, execution / "requested_config.yaml")
    save_config(config, execution / "resolved_config.yaml")
    return execution


def test_transaction_publishes_flat_dataset_and_method_result_atomically(
    prepared_config,
) -> None:
    transaction = (
        prepared_config.project.workspace_root / "temporary_runs/queue_fixture"
    )
    _transaction_dataset(transaction)
    execution = _successful_ap02_execution(transaction, prepared_config)
    paths = experiment_paths(prepared_config)

    results = publish_queue_transaction(
        transaction,
        queue_id="queue_fixture",
        configs=[prepared_config],
        results={"baseline": {"status": "completed", "result": str(execution)}},
    )

    target = paths.methods / "ap02/baseline"
    assert results["baseline"]["result"] == str(target.resolve())
    assert (paths.dataset_root / "raw_images/moving/frame.png").read_bytes() == b"moving"
    assert (paths.dataset_root / "observations/shared_all_aruco_observations.csv").is_file()
    descriptor = json.loads(
        (paths.dataset_root / "dataset.json").read_text()
    )
    assert descriptor["storage"]["dataset_root"] == str(
        paths.dataset_root.resolve()
    )
    source = json.loads(
        (paths.dataset_root / "metadata/source.json").read_text()
    )
    assert source["canonical_dataset_root"] == str(
        paths.dataset_root.resolve()
    )
    assert "canonical_source_roots" not in source
    assert (target / "RESULT.txt").is_file()
    assert (target / "RESULT.json").is_file()
    assert (target / "camera_extrinsics.csv").is_file()
    assert (target / "diagnostics/method/graph_ba").is_dir()
    assert (target / "logs/ap02.log").is_file()
    assert (target / "provenance/resolved_config.yaml").is_file()
    assert (paths.root / "SUMMARY.json").is_file()
    assert (paths.root / "COMPARISON.csv").is_file()
    assert not list(paths.root.rglob("executions"))
    assert not list(paths.root.rglob("current"))
    assert not list(paths.root.rglob("99_FINAL_RESULTS"))


def test_detector_retry_rekeys_complete_dataset_manifest(
    prepared_config,
) -> None:
    """A post-capture detector variant must own one consistent experiment ID."""

    transaction = (
        prepared_config.project.workspace_root
        / "temporary_runs/detector_retry_fixture"
    )
    _transaction_dataset(transaction)
    source_descriptor = json.loads(
        (transaction / "dataset/dataset.json").read_text(encoding="utf-8")
    )
    source_descriptor.update(
        {
            "id": prepared_config.dataset.id,
            "experiment_fingerprint": experiment_fingerprint(prepared_config),
            "created_at": "2026-07-28T19:25:32+00:00",
        }
    )
    (transaction / "dataset/dataset.json").write_text(
        json.dumps(source_descriptor),
        encoding="utf-8",
    )
    experiment_id = f"{prepared_config.dataset.id}__aruco_high_sensitivity"
    final_config = prepared_config.model_copy(
        update={
            "dataset": prepared_config.dataset.model_copy(
                update={"id": experiment_id}
            ),
            "project": prepared_config.project.model_copy(
                update={"experiment_id": experiment_id}
            ),
            "markers": prepared_config.markers.model_copy(
                update={"detection_mode": "high_sensitivity"}
            ),
        },
        deep=True,
    )

    published = publish_preparation_transaction(
        transaction,
        queue_id="detector_retry_fixture",
        config=final_config,
        preparation=transaction / "jobs/queue_preflight/prepared",
    )

    descriptor = json.loads(
        (published / "dataset.json").read_text(encoding="utf-8")
    )
    assert published == experiment_paths(final_config).dataset_root
    assert descriptor["id"] == experiment_id
    assert descriptor["storage"]["canonical_id"] == experiment_id
    assert (
        descriptor["experiment_fingerprint"]
        == experiment_fingerprint(final_config)
    )
    assert descriptor["input_fingerprint"] == "input_fixture"
    assert descriptor["created_at"] == "2026-07-28T19:25:32+00:00"

    transaction_descriptor = json.loads(
        (transaction / "dataset/dataset.json").read_text(encoding="utf-8")
    )
    assert transaction_descriptor["id"] == experiment_id
    assert (
        transaction_descriptor["experiment_fingerprint"]
        == experiment_fingerprint(final_config)
    )

    # Method startup validates the reused transaction view, not only the
    # canonical copy. Both must therefore be idempotent.
    transaction_paths = replace(
        experiment_paths(final_config),
        dataset_root=transaction / "dataset",
        datasets=transaction / "dataset",
    )
    write_experiment_manifest(
        final_config,
        transaction_paths,
        "input_fixture",
    )
    write_experiment_manifest(
        final_config,
        experiment_paths(final_config),
        "input_fixture",
    )


def test_directory_publication_retries_transient_permission_error(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = tmp_path / "incoming"
    target = tmp_path / "published"
    source.mkdir()
    (source / "RESULT.txt").write_text("complete\n", encoding="utf-8")
    original_rename = Path.rename
    failures = 0

    def flaky_rename(path: Path, destination: Path):
        nonlocal failures
        if path == source and failures < 2:
            failures += 1
            raise PermissionError("temporary Windows directory lock")
        return original_rename(path, destination)

    monkeypatch.setattr(Path, "rename", flaky_rename)
    monkeypatch.setattr(
        "camera_rig_calibration.publication.time.sleep",
        lambda _seconds: None,
    )

    _rename_with_retry(source, target)

    assert failures == 2
    assert (target / "RESULT.txt").is_file()
    assert not source.exists()


def test_completed_staging_directory_is_reused_after_failed_rename(
    prepared_config,
) -> None:
    config = prepared_config.model_copy(
        update={
            "project": prepared_config.project.model_copy(
                update={"run_label": "ap02_variant2"},
                deep=True,
            )
        },
        deep=True,
    )
    transaction = config.project.workspace_root / "temporary_runs/staged"
    execution = _successful_ap02_execution(transaction, config)
    target = experiment_paths(config).methods / "ap02/baseline"
    staged = target.with_name(".incoming_baseline_interrupted")
    (staged / "provenance").mkdir(parents=True)
    (staged / "RESULT.json").write_text(
        json.dumps(
            {
                "status": "available",
                "method": "ap02",
                    "label": "baseline",
                "method_fingerprint": "method_fixture",
            }
        ),
        encoding="utf-8",
    )
    (staged / "RESULT.txt").write_text("complete\n", encoding="utf-8")
    (staged / "camera_extrinsics.csv").write_text(
        "camera_id,tx\nfront-left,0\n",
        encoding="utf-8",
    )

    published, outcome = _publish_success(
        execution,
        config=config,
        canonical_root=experiment_paths(config).root,
        queue_id="staged",
    )

    assert outcome == "completed"
    assert published == target
    assert (target / "RESULT.json").is_file()
    assert not staged.exists()


def test_same_label_and_fingerprint_is_skipped_but_changed_config_conflicts(
    prepared_config,
) -> None:
    first = prepared_config.project.workspace_root / "temporary_runs/first"
    _transaction_dataset(first)
    execution = _successful_ap02_execution(first, prepared_config)
    publish_queue_transaction(
        first,
        queue_id="first",
        configs=[prepared_config],
        results={"baseline": {"status": "completed", "result": str(execution)}},
    )

    repeated = prepared_config.project.workspace_root / "temporary_runs/repeated"
    _transaction_dataset(repeated)
    repeated_execution = _successful_ap02_execution(repeated, prepared_config)
    rows = publish_queue_transaction(
        repeated,
        queue_id="repeated",
        configs=[prepared_config],
        results={
            "baseline": {
                "status": "completed",
                "result": str(repeated_execution),
            }
        },
    )
    assert rows["baseline"]["status"] == "duplicate_skipped"

    changed = prepared_config.project.workspace_root / "temporary_runs/changed"
    _transaction_dataset(changed)
    changed_execution = _successful_ap02_execution(
        changed, prepared_config, fingerprint="different_method"
    )
    with pytest.raises(RuntimeError, match="Result label conflict"):
        publish_queue_transaction(
            changed,
            queue_id="changed",
            configs=[prepared_config],
            results={
                "baseline": {
                    "status": "completed",
                    "result": str(changed_execution),
                }
            },
        )


def test_failed_transaction_is_archived_as_non_authoritative_attempt(
    prepared_config,
) -> None:
    transaction = (
        prepared_config.project.workspace_root / "temporary_runs/failed_queue"
    )
    _transaction_dataset(transaction)

    results = publish_queue_transaction(
        transaction,
        queue_id="failed_queue",
        configs=[prepared_config],
        results={"baseline": {"status": "failed", "error": "optimizer failed"}},
    )

    root = experiment_paths(prepared_config).root
    assert results["baseline"]["status"] == "failed_published"
    attempt = Path(results["baseline"]["attempt"])
    summary = json.loads((attempt / "FAILURE.json").read_text())
    assert summary["scientific_validity"] == "incomplete/non-authoritative"
    assert summary["cause_code"] == "optimizer_failed"
    experiment_summary = json.loads((root / "SUMMARY.json").read_text())
    assert experiment_summary["status"] == "failed"
    assert not any(root.glob("methods/**/*"))
    journal = json.loads(
        (transaction / "queue_transaction.json").read_text()
    )
    assert journal["status"] == "published"
