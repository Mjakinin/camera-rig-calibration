from __future__ import annotations

from pathlib import Path
import json
from types import SimpleNamespace

import pytest
import yaml

from camera_rig_calibration.config import save_config
from camera_rig_calibration.config.models import (
    DatasetCategory,
    EvaluationSettings,
    MethodSettings,
    RigConfig,
    SceneType,
    SimulationSettings,
)
from camera_rig_calibration.observations import ResolvedSelections
from camera_rig_calibration.experiments import experiment_paths
from camera_rig_calibration.queueing import (
    BatchConfig,
    QueueConfig,
    QueueEntry,
    QueueRunner,
    _bind_prepared_dataset,
    _configured_selection_summary,
    _method_selection_summary,
    _method_result_summary,
    load_batch,
    load_queue_partitions,
    save_batch,
    save_queue,
)
from camera_rig_calibration.input.preparation import build_preparation_plan


def test_method_result_summary_reports_runtime_metrics_and_logs(
    tmp_path: Path,
) -> None:
    result = tmp_path / "method_result"
    (result / "provenance").mkdir(parents=True)
    (result / "logs").mkdir()
    (result / "provenance/timings.json").write_text(
        json.dumps(
            {
                "_structured": {
                    "method_ap02": {
                        "stage_elapsed_seconds": 125.4
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    (result / "RESULT.json").write_text(
        json.dumps(
            {
                "layout_version": 2,
                "runtime_seconds": 125.4,
                "primary_result": "combined",
                "static_camera_count": 2,
                "reference_marker_id": 3,
            }
        ),
        encoding="utf-8",
    )

    summary, logs = _method_result_summary(result)

    assert summary == (
        "method=2m05s, primary=combined, cameras=2, marker=3"
    )
    assert logs == str(result / "logs")


def test_preflight_selection_summaries_name_auto_and_frozen_values(
    prepared_config: RigConfig,
) -> None:
    resolved = ResolvedSelections(
        root_camera="front-left",
        ap02_reference_marker_id=7,
        ap03_single_scale_marker_id=9,
        ap03_multi_marker_ids=(7, 9),
        evaluation_anchor_marker_id=None,
        marker_ids=(7, 9),
        payload={},
    )
    ap01 = prepared_config.model_copy(
        update={"methods": MethodSettings(enabled=["ap01"])},
        deep=True,
    )
    ap02 = prepared_config.model_copy(
        update={
            "methods": MethodSettings(enabled=["ap02"]).model_copy(
                update={
                    "ap02": MethodSettings().ap02.model_copy(
                        update={"reference_marker_id": 9}
                    )
                },
                deep=True,
            )
        },
        deep=True,
    )

    assert (
        _method_selection_summary(ap01, resolved)
        == "root camera=front-left"
    )
    assert (
        _method_selection_summary(ap02, resolved)
        == "reference marker=7"
    )
    assert (
        _configured_selection_summary(ap02)
        == "reference marker=9"
    )


def test_queue_method_uses_preflight_dataset_without_intrinsics_recomposition(
    prepared_config,
) -> None:
    prepared_root = prepared_config.dataset.prepared_root
    assert prepared_root is not None
    moving_info = (
        prepared_root
        / "raw_images"
        / "camera_info"
        / f"{prepared_config.moving_camera.id}.json"
    )
    candidate = prepared_config.model_copy(
        update={
            "moving_camera": prepared_config.moving_camera.model_copy(
                update={
                    "intrinsics": moving_info,
                    "intrinsics_profile": "fixture@deadbeef",
                },
                deep=True,
            )
        },
        deep=True,
    )

    bound = _bind_prepared_dataset(candidate, prepared_root)
    plan = build_preparation_plan(
        bound, Path(__file__).resolve().parents[1]
    )

    assert bound.dataset.prepared_root == prepared_root.resolve()
    assert bound.moving_camera.intrinsics is None
    assert bound.moving_camera.intrinsics_profile is None
    assert bound.moving_camera.intrinsic_calibration_video is None
    assert bound.moving_camera.intrinsic_calibration_images is None
    assert plan.dataset_root == prepared_root.resolve()
    assert plan.commands == []


def test_simulation_queue_method_reuses_capture_without_recapture(
    prepared_config,
    tmp_path: Path,
) -> None:
    prepared_root = prepared_config.dataset.prepared_root
    assert prepared_root is not None
    world = tmp_path / "bus.sdf"
    route = tmp_path / "route.json"
    world.write_text(
        '<sdf version="1.8"><world name="bus"/></sdf>',
        encoding="utf-8",
    )
    route.write_text('{"frames": []}\n', encoding="utf-8")
    capturing = RigConfig.model_validate(
        prepared_config.model_copy(
            update={
                "dataset": prepared_config.dataset.model_copy(
                    update={
                        "category": DatasetCategory.SIMULATION,
                        "scene_type": SceneType.SIMULATION,
                        "prepared_root": None,
                    },
                    deep=True,
                ),
                "static_cameras": [
                    camera.model_copy(
                        update={
                            "image_topic": f"/{camera.id}/image",
                            "camera_info_topic": (
                                f"/{camera.id}/camera_info"
                            ),
                        },
                        deep=True,
                    )
                    for camera in prepared_config.static_cameras
                ],
                "moving_camera": (
                    prepared_config.moving_camera.model_copy(
                        update={
                            "image_topic": "/moving/image",
                            "camera_info_topic": "/moving/camera_info",
                        },
                        deep=True,
                    )
                ),
                "simulation": SimulationSettings(
                    enabled=True,
                    world=world,
                    route=route,
                    route_name="route2",
                    target_route_frames=189,
                    world_baseline={"route": "route2"},
                ),
            },
            deep=True,
        ).model_dump(mode="python")
    )

    bound = _bind_prepared_dataset(capturing, prepared_root)
    plan = build_preparation_plan(
        bound, Path(__file__).resolve().parents[1]
    )

    assert capturing.simulation.enabled
    assert not bound.simulation.enabled
    assert bound.dataset.category.value == "simulation"
    assert bound.dataset.prepared_root == prepared_root.resolve()
    assert bound.simulation.route_name == "route2"
    assert bound.simulation.target_route_frames == 189
    assert bound.simulation.world == world
    assert bound.simulation.route == route
    assert bound.simulation.world_baseline == {"route": "route2"}
    assert plan.dataset_root == prepared_root.resolve()
    assert plan.commands == []


def test_queue_rejects_multiple_experiments_and_legacy_schema(
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
    legacy = tmp_path / "queues/legacy.yaml"
    legacy.parent.mkdir(parents=True, exist_ok=True)
    legacy.write_text(
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
    with pytest.raises(ValueError, match="Only schema_version 5"):
        load_queue_partitions(legacy)

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        yaml.safe_dump(
            {
                "kind": "rigcal_queue",
                "schema_version": 5,
                "id": "invalid_multi_experiment",
                "entries": [
                    {"id": "first", "config": str(first)},
                    {"id": "second", "config": str(second)},
                ],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="exactly one experiment"):
        load_queue_partitions(destination)


def test_queue_rejects_forward_or_unknown_dependencies() -> None:
    with pytest.raises(ValueError, match="dependencies"):
        QueueConfig.model_validate(
            {
                "kind": "rigcal_queue",
                "schema_version": 5,
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


def test_batch_manifest_round_trips_ordered_experiment_queues(
    tmp_path: Path,
) -> None:
    first = tmp_path / "batch/01_route2/queue/queue.yaml"
    second = tmp_path / "batch/02_fov/queue/queue.yaml"
    for path, queue_id in ((first, "route2"), (second, "fov")):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            yaml.safe_dump(
                {
                    "kind": "rigcal_queue",
                    "schema_version": 5,
                    "id": queue_id,
                    "entries": [
                        {"id": "ap02", "config": "ap02.yaml"}
                    ],
                }
            ),
            encoding="utf-8",
        )
    path = save_batch(
        "paper_batch",
        [("route2", first), ("fov_100deg", second)],
        tmp_path / "batch/batch.yaml",
    )

    loaded = load_batch(path)

    assert isinstance(loaded, BatchConfig)
    assert [entry.experiment_id for entry in loaded.queues] == [
        "route2",
        "fov_100deg",
    ]
    assert [entry.queue for entry in loaded.queues] == [
        first.resolve(),
        second.resolve(),
    ]


def test_prepare_only_uses_one_dedicated_pipeline_without_queue_preflight(
    prepared_config,
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = prepared_config.model_copy(
        update={
            "project": prepared_config.project.model_copy(
                update={"execution_mode": "prepare_only"}
            )
        },
        deep=True,
    )
    config_path = save_config(config, tmp_path / "prepare.yaml")
    queue = QueueConfig(
        id="prepare_once",
        entries=[QueueEntry(id="prepare_input", config=config_path)],
    )
    calls: list[object] = []
    published = tmp_path / "results/real_vehicle/native_rate/input_1"
    published.mkdir(parents=True)

    class DummyOrchestrator:
        def __init__(self, *args, **kwargs):
            calls.append(("init", kwargs))

        def show_dry_run(self, config):
            calls.append(("dry_run", config))

        def validate_ready(self, config):
            calls.append(("validate", config))

        def run(self, config):
            calls.append(("run", config))
            return published

    monkeypatch.setattr(
        "camera_rig_calibration.queueing.PipelineOrchestrator",
        DummyOrchestrator,
    )
    monkeypatch.setattr(
        "camera_rig_calibration.queueing.run_queue_preflight",
        lambda *args, **kwargs: pytest.fail(
            "prepare-only must not invoke queue preflight"
        ),
    )
    monkeypatch.setattr(
        "camera_rig_calibration.queueing.publish_preparation_transaction",
        lambda *args, **kwargs: published,
    )

    result = QueueRunner(tmp_path).run(queue)

    assert result["prepare_input"]["status"] == "completed"
    assert result["prepare_input"]["execution_mode"] == "prepare_only"
    assert [item[0] for item in calls].count("run") == 1
    assert [item[0] for item in calls].count("validate") == 1


def test_explicit_evaluation_anchor_runs_evaluator_only_on_existing_methods(
    prepared_config, tmp_path: Path, monkeypatch
) -> None:
    config = prepared_config.model_copy(
        update={
            "evaluation": EvaluationSettings(anchor_marker_id=9)
        },
        deep=True,
    )
    paths = experiment_paths(config)
    experiment = paths.root
    observations = paths.dataset_root / "observations"
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
    result = experiment / "methods/ap02/baseline"
    (result / "diagnostics/method").mkdir(parents=True)
    (result / "provenance").mkdir()
    (result / "provenance/run_manifest.json").write_text(
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
    config_path = save_config(config, tmp_path / "ap02.yaml")
    save_config(config, result / "provenance/resolved_config.yaml")
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
            config.project.workspace_root
            / "temporary_runs/evaluation_only/results/evaluations/"
            "SELECTED_COMMON_EVALUATION.json"
        ).read_text(encoding="utf-8")
    )
    assert selected["anchor_marker_id"] == 9
