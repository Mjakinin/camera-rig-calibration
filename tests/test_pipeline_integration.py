from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import pytest
from pydantic import BaseModel, ConfigDict

import camera_rig_calibration.queueing as queueing_module
from camera_rig_calibration.config import load_config
from camera_rig_calibration.config import save_config
from camera_rig_calibration.config.models import (
    DatasetSettings,
    EvaluationSettings,
    MovingCameraSettings,
    ProjectSettings,
    RigConfig,
    SelectionSettings,
    StaticCameraSettings,
)
from camera_rig_calibration.contracts import CommandSpec, RequirementResult, RunContext
from camera_rig_calibration.registry import calibration_methods
from camera_rig_calibration.queueing import QueueConfig, QueueEntry, QueueRunner
from camera_rig_calibration.runtime import PipelineOrchestrator

from conftest import real_method_settings, write_intrinsics


class PipelineDummyOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")


@dataclass(frozen=True)
class PipelineDummyMethod:
    id: str = "pipeline_dummy"
    display_name: str = "Pipeline contract fixture"
    config_model: type[BaseModel] = PipelineDummyOptions

    def requirements(self, context: RunContext) -> RequirementResult:
        return RequirementResult.ok()

    def commands(self, context: RunContext) -> Sequence[CommandSpec]:
        return ()

    def collect(self, context: RunContext) -> dict:
        return {
            "status": "FIXTURE_OK",
            "success": True,
            "available_static_cameras": [
                camera.id for camera in context.config.static_cameras
            ],
            "runtime_seconds": 0.0,
            "directory": str(context.run_directory / "07_COMPARISON"),
        }


def _marker_image(path: Path) -> None:
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    if hasattr(cv2.aruco, "generateImageMarker"):
        marker = cv2.aruco.generateImageMarker(
            dictionary, 7, 240, borderBits=1
        )
    else:
        marker = np.zeros((240, 240), dtype=np.uint8)
        cv2.aruco.drawMarker(dictionary, 7, 240, marker, 1)
    canvas = np.full((480, 640), 255, dtype=np.uint8)
    canvas[120:360, 200:440] = marker
    path.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(path), canvas)


@pytest.mark.integration
def test_complete_orchestrator_contract_without_running_ap_methods(tmp_path: Path) -> None:
    if "pipeline_dummy" not in calibration_methods:
        calibration_methods.register(PipelineDummyMethod())
    dataset = tmp_path / "prepared"
    raw = dataset / "raw_images"
    camera_ids = ["outside-left", "roof.camera"]
    for camera_id in camera_ids:
        _marker_image(raw / "static" / f"{camera_id}.png")
        write_intrinsics(raw / "camera_info" / f"{camera_id}.json", camera_id)
    _marker_image(raw / "moving" / "frame_000000.png")
    write_intrinsics(
        raw / "camera_info" / "calibration-camera.json", "calibration-camera"
    )
    config = RigConfig(
        project=ProjectSettings(
            workspace_root=tmp_path / "workspace",
            dataset_cache_root=tmp_path / "datasets",
            output_root=tmp_path / "results",
        ),
        dataset=DatasetSettings(id="outside_contract", prepared_root=dataset),
        static_cameras=[StaticCameraSettings(id=value) for value in camera_ids],
        moving_camera=MovingCameraSettings(id="calibration-camera"),
        methods=real_method_settings(["pipeline_dummy"]),
        selection=SelectionSettings(mode="auto"),
        evaluation=EvaluationSettings(enabled=False),
    )
    run = PipelineOrchestrator(Path(__file__).resolve().parents[1]).run(config)
    manifest = json.loads((run / "run_manifest.json").read_text())
    assert manifest["status"] == "completed"
    assert all(stage["status"] == "completed" for stage in manifest["stages"])
    resolved = load_config(run / "resolved_config.yaml", resolve_paths=False)
    assert resolved.methods.ap01.root_camera in camera_ids
    assert resolved.methods.ap02.reference_marker_id == 7
    assert resolved.methods.ap03_single.scale_marker_id == 7
    assert resolved.evaluation.anchor_marker_id == "auto"
    assert resolved.markers.accepted_ids == "all_detected"
    dataset_manifest = json.loads(
        (run / "00_INPUT" / "dataset_manifest.json").read_text()
    )
    assert len(dataset_manifest["automatic_selections"]) == 3
    commands = (run / "commands.txt").read_text()
    assert (
        "camera_rig_calibration.observation_services.detection" in commands
    )
    assert "run/real_vehicle_data" not in commands.replace("\\", "/")

    prepare_only = config.model_copy(
        update={
            "project": config.project.model_copy(
                update={"run_label": "capture_test", "execution_mode": "prepare_only"}
            ),
            "methods": real_method_settings(["ap02"]),
            "evaluation": EvaluationSettings(enabled=True),
        },
        deep=True,
    )
    prepared_run = PipelineOrchestrator(Path(__file__).resolve().parents[1]).run(
        prepare_only
    )
    prepared_manifest = json.loads(
        (
            prepared_run
            / "metadata/preparation/run_manifest.json"
        ).read_text()
    )
    assert prepared_manifest["status"] == "completed"
    assert prepared_manifest["execution_mode"] == "prepare_only"
    assert not any(
        stage["id"].startswith("method_")
        for stage in prepared_manifest["stages"]
    )
    assert (prepared_run / "dataset.json").is_file()
    assert not (prepared_run / "RESULT.json").exists()


@pytest.mark.integration
def test_queue_auto_freezes_automatic_selections_and_reuses_shared_preparation(
    tmp_path: Path,
) -> None:
    first_method = PipelineDummyMethod(id="pipeline_dummy_first")
    second_method = PipelineDummyMethod(id="pipeline_dummy_second")
    for method in (first_method, second_method):
        if method.id not in calibration_methods:
            calibration_methods.register(method)

    dataset = tmp_path / "prepared"
    raw = dataset / "raw_images"
    camera_ids = ["front-left", "rear.camera"]
    for camera_id in camera_ids:
        _marker_image(raw / "static" / f"{camera_id}.png")
        write_intrinsics(
            raw / "camera_info" / f"{camera_id}.json", camera_id
        )
    _marker_image(raw / "moving" / "frame_000000.png")
    write_intrinsics(
        raw / "camera_info" / "calibration-camera.json",
        "calibration-camera",
    )
    common = {
        "project": ProjectSettings(
            workspace_root=tmp_path / "workspace",
            dataset_cache_root=tmp_path / "datasets",
            output_root=tmp_path / "results",
            experiment_id="shared_queue",
        ),
        "dataset": DatasetSettings(
            id="shared_queue", input_root=dataset
        ),
        "static_cameras": [
            StaticCameraSettings(
                id=value,
                images=[raw / "static" / f"{value}.png"],
                intrinsics=raw / "camera_info" / f"{value}.json",
            )
            for value in camera_ids
        ],
        "moving_camera": MovingCameraSettings(
            id="calibration-camera",
            frames=raw / "moving",
            intrinsics=(
                raw
                / "camera_info"
                / "calibration-camera.json"
            ),
        ),
        "selection": SelectionSettings(mode="auto"),
        "evaluation": EvaluationSettings(enabled=False),
    }
    configs = [
        RigConfig(
            methods=real_method_settings(
                [method.id], extensions={method.id: {}}
            ),
            **common,
        )
        for method in (first_method, second_method)
    ]
    paths = [
        save_config(config, tmp_path / "queue" / f"{index}.yaml")
        for index, config in enumerate(configs, 1)
    ]
    queue = QueueConfig(
        id="shared_preparation",
        entries=[
            QueueEntry(id=f"job_{index}", config=path)
            for index, path in enumerate(paths, 1)
        ],
    )
    reviews = 0

    def reviewer(config, resolved, run):
        nonlocal reviews
        reviews += 1
        return {
            "root_camera": resolved.root_camera,
            "ap02_reference_marker_id": (
                resolved.ap02_reference_marker_id
            ),
            "ap03_single_scale_marker_id": (
                resolved.ap03_single_scale_marker_id
            ),
            "ap03_multi_marker_ids": list(
                resolved.ap03_multi_marker_ids
            ),
        }

    automatic = QueueRunner(
        Path(__file__).resolve().parents[1]
    ).run(queue)
    assert {
        entry: row["status"] for entry, row in automatic.items()
    } == {
        "job_1": "completed",
        "job_2": "completed",
    }

    results = QueueRunner(
        Path(__file__).resolve().parents[1],
        selection_reviewer=reviewer,
    ).run(queue)

    assert reviews == 0
    assert {row["status"] for row in results.values()} == {
        "duplicate_skipped"
    }
    resolved_configs = [
        load_config(
            Path(row["result"]) / "provenance/resolved_config.yaml"
        )
        for row in results.values()
    ]
    requested_configs = [
        load_config(
            Path(row["result"]) / "provenance/requested_config.yaml"
        )
        for row in results.values()
    ]
    assert len(resolved_configs) == 2
    assert all(
        config.selection.mode == "explicit"
        for config in resolved_configs
    )
    assert all(
        config.moving_camera.intrinsics is None
        and config.moving_camera.intrinsics_profile is None
        for config in requested_configs
    )
    assert all(
        config.moving_camera.intrinsics is not None
        and config.dataset.prepared_root is not None
        and config.dataset.prepared_root
        in config.moving_camera.intrinsics.parents
        for config in resolved_configs
    )
    detector_commands = []
    for row in results.values():
        detector_commands.append(
            "observation_detection.py"
            in (
                Path(row["result"]) / "provenance/commands.txt"
            ).read_text()
        )
    assert detector_commands == [False, False]
    assert not (
        tmp_path / "workspace/temporary_runs/shared_preparation"
    ).exists()
    assert len(
        {
            config.dataset.prepared_root
            for config in resolved_configs
        }
    ) == 1


@pytest.mark.integration
def test_queue_review_freezes_one_decision_mapping_per_variant(
    tmp_path: Path,
) -> None:
    methods = (
        PipelineDummyMethod(id="pipeline_review_first"),
        PipelineDummyMethod(id="pipeline_review_second"),
    )
    for method in methods:
        if method.id not in calibration_methods:
            calibration_methods.register(method)
    dataset = tmp_path / "prepared_review"
    raw = dataset / "raw_images"
    camera_ids = ["front", "rear"]
    for camera_id in camera_ids:
        _marker_image(raw / "static" / f"{camera_id}.png")
        write_intrinsics(
            raw / "camera_info" / f"{camera_id}.json", camera_id
        )
    _marker_image(raw / "moving/frame_000000.png")
    write_intrinsics(
        raw / "camera_info/calibration-camera.json",
        "calibration-camera",
    )
    common = {
        "project": ProjectSettings(
            workspace_root=tmp_path / "workspace",
            dataset_cache_root=tmp_path / "datasets",
            output_root=tmp_path / "results",
            experiment_id="review_queue",
        ),
        "dataset": DatasetSettings(
            id="review_queue", input_root=dataset
        ),
        "static_cameras": [
            StaticCameraSettings(
                id=camera_id,
                images=[raw / "static" / f"{camera_id}.png"],
                intrinsics=(
                    raw / "camera_info" / f"{camera_id}.json"
                ),
            )
            for camera_id in camera_ids
        ],
        "moving_camera": MovingCameraSettings(
            id="calibration-camera",
            frames=raw / "moving",
            intrinsics=raw / "camera_info/calibration-camera.json",
        ),
        "selection": SelectionSettings(mode="review_once"),
        "evaluation": EvaluationSettings(enabled=False),
    }
    configs = [
        RigConfig(
            methods=real_method_settings(
                [method.id], extensions={method.id: {}}
            ),
            **common,
        )
        for method in methods
    ]
    config_paths = [
        save_config(config, tmp_path / "review_queue" / f"{index}.yaml")
        for index, config in enumerate(configs, 1)
    ]
    queue = QueueConfig(
        id="review_queue",
        entries=[
            QueueEntry(id=f"variant_{index}", config=path)
            for index, path in enumerate(config_paths, 1)
        ],
    )
    review_calls = 0

    def reviewer(jobs, run_directory):
        nonlocal review_calls
        review_calls += 1
        assert len(jobs) == 2
        roots = ("front", "rear")
        return {
            job.entry_id: {
                "root_camera": root,
                "ap02_reference_marker_id": (
                    job.selections.ap02_reference_marker_id
                ),
                "ap03_single_scale_marker_id": (
                    job.selections.ap03_single_scale_marker_id
                ),
                "ap03_multi_marker_ids": list(
                    job.selections.ap03_multi_marker_ids
                ),
            }
            for job, root in zip(jobs, roots, strict=True)
        }

    waiting = QueueRunner(
        Path(__file__).resolve().parents[1]
    ).run(queue)
    assert {
        row["status"] for row in waiting.values()
    } == {"waiting_for_selection"}

    results = QueueRunner(
        Path(__file__).resolve().parents[1],
        selection_reviewer=reviewer,
    ).run(queue)

    assert review_calls == 1
    assert {
        row["status"] for row in results.values()
    } == {"completed"}
    resolved_roots = [
        load_config(
            Path(results[f"variant_{index}"]["result"])
            / "provenance/resolved_config.yaml"
        ).methods.ap01.root_camera
        for index in (1, 2)
    ]
    assert resolved_roots == ["front", "rear"]


@pytest.mark.integration
def test_resume_retries_only_publication_after_completed_method(
    tmp_path: Path,
    monkeypatch,
) -> None:
    method = PipelineDummyMethod(id="pipeline_dummy_publication")
    if method.id not in calibration_methods:
        calibration_methods.register(method)
    dataset = tmp_path / "prepared"
    raw = dataset / "raw_images"
    camera_ids = ["front-left", "rear.camera"]
    for camera_id in camera_ids:
        _marker_image(raw / "static" / f"{camera_id}.png")
        write_intrinsics(
            raw / "camera_info" / f"{camera_id}.json",
            camera_id,
        )
    _marker_image(raw / "moving" / "frame_000000.png")
    write_intrinsics(
        raw / "camera_info" / "calibration-camera.json",
        "calibration-camera",
    )
    config = RigConfig(
        project=ProjectSettings(
            workspace_root=tmp_path / "workspace",
            dataset_cache_root=tmp_path / "datasets",
            output_root=tmp_path / "results",
            run_label="pipeline_dummy_publication_baseline",
        ),
        dataset=DatasetSettings(
            id="publication_resume",
            prepared_root=dataset,
        ),
        static_cameras=[
            StaticCameraSettings(id=camera_id)
            for camera_id in camera_ids
        ],
        moving_camera=MovingCameraSettings(
            id="calibration-camera"
        ),
        methods=real_method_settings(
            [method.id], extensions={method.id: {}}
        ),
        selection=SelectionSettings(mode="auto"),
        evaluation=EvaluationSettings(enabled=False),
    )
    config_path = save_config(config, tmp_path / "queue/method.yaml")
    queue = QueueConfig(
        id="publication_resume",
        entries=[
            QueueEntry(
                id="pipeline_dummy_publication_baseline",
                config=config_path,
            )
        ],
    )
    real_publish = queueing_module.publish_queue_transaction
    fail_once = True

    def fail_first_publication(*args, **kwargs):
        nonlocal fail_once
        if kwargs.get("finalize") is False and fail_once:
            fail_once = False
            raise PermissionError("temporary Windows directory lock")
        return real_publish(*args, **kwargs)

    monkeypatch.setattr(
        queueing_module,
        "publish_queue_transaction",
        fail_first_publication,
    )
    first = QueueRunner(Path(__file__).resolve().parents[1]).run(queue)

    entry_id = "pipeline_dummy_publication_baseline"
    assert first[entry_id]["status"] == "publication_failed"
    completed = Path(first[entry_id]["result"])
    assert (completed / "run_manifest.json").is_file()

    def forbidden_method_rerun(*_args, **_kwargs):
        pytest.fail("completed method must not run again during publication retry")

    monkeypatch.setattr(PipelineOrchestrator, "run", forbidden_method_rerun)
    resumed = QueueRunner(Path(__file__).resolve().parents[1]).run(queue)

    assert resumed[entry_id]["status"] == "completed"
    assert resumed[entry_id]["published"] is True
    assert Path(resumed[entry_id]["result"]).joinpath("RESULT.json").is_file()


@pytest.mark.integration
def test_queue_runs_ready_job_when_an_independent_job_fails_preflight(
    tmp_path: Path,
) -> None:
    method = PipelineDummyMethod(id="pipeline_dummy_independent")
    if method.id not in calibration_methods:
        calibration_methods.register(method)
    dataset = tmp_path / "prepared"
    raw = dataset / "raw_images"
    for camera_id in ("cam_a", "cam_b"):
        _marker_image(raw / "static" / f"{camera_id}.png")
        write_intrinsics(
            raw / "camera_info" / f"{camera_id}.json", camera_id
        )
    _marker_image(raw / "moving" / "frame_000000.png")
    write_intrinsics(
        raw / "camera_info" / "moving.json", "moving"
    )
    common = {
        "project": ProjectSettings(
            workspace_root=tmp_path / "workspace",
            dataset_cache_root=tmp_path / "datasets",
            output_root=tmp_path / "results",
            experiment_id="independent_queue",
        ),
        "dataset": DatasetSettings(
            id="independent_queue", prepared_root=dataset
        ),
        "static_cameras": [
            StaticCameraSettings(id=value)
            for value in ("cam_a", "cam_b")
        ],
        "moving_camera": MovingCameraSettings(id="moving"),
        "methods": real_method_settings(
            [method.id], extensions={method.id: {}}
        ),
        "selection": SelectionSettings(mode="auto"),
        "evaluation": EvaluationSettings(enabled=False),
    }
    ready = RigConfig(**common)
    rejected = RigConfig(
        **common,
        observation_quality={"minimum_marker_area_ratio": 0.99},
    )
    ready_path = save_config(ready, tmp_path / "queue/ready.yaml")
    rejected_path = save_config(
        rejected, tmp_path / "queue/rejected.yaml"
    )
    queue = QueueConfig(
        id="independent_preflight",
        entries=[
            QueueEntry(id="ready", config=ready_path),
            QueueEntry(id="rejected", config=rejected_path),
        ],
    )

    results = QueueRunner(
        Path(__file__).resolve().parents[1]
    ).run(queue)

    assert results["ready"]["status"] == "completed"
    assert results["rejected"]["status"] == "failed_published"
    assert (
        results["rejected"]["failure"]["cause_code"]
        == "preflight_failed"
    )
    assert Path(results["ready"]["result"]).is_dir()


@pytest.mark.integration
def test_required_camera_gate_waits_and_diagnostic_override_is_published(
    tmp_path: Path,
) -> None:
    method = PipelineDummyMethod(id="pipeline_dummy_partial_coverage")
    if method.id not in calibration_methods:
        calibration_methods.register(method)
    dataset = tmp_path / "prepared"
    raw = dataset / "raw_images"
    _marker_image(raw / "static/cam_a.png")
    blank = np.full((480, 640), 255, dtype=np.uint8)
    (raw / "static").mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(raw / "static/cam_b.png"), blank)
    for camera_id in ("cam_a", "cam_b"):
        write_intrinsics(
            raw / "camera_info" / f"{camera_id}.json",
            camera_id,
        )
    _marker_image(raw / "moving/frame_000000.png")
    write_intrinsics(raw / "camera_info/moving.json", "moving")
    config = RigConfig(
        project=ProjectSettings(
            workspace_root=tmp_path / "workspace",
            dataset_cache_root=tmp_path / "datasets",
            output_root=tmp_path / "results",
            experiment_id="partial_coverage",
        ),
        dataset=DatasetSettings(
            id="partial_coverage",
            prepared_root=dataset,
        ),
        static_cameras=[
            StaticCameraSettings(id="cam_a"),
            StaticCameraSettings(id="cam_b"),
        ],
        moving_camera=MovingCameraSettings(id="moving"),
        methods=real_method_settings(
            [method.id], extensions={method.id: {}}
        ),
        selection=SelectionSettings(mode="auto"),
        evaluation=EvaluationSettings(enabled=False),
    )
    config_path = save_config(config, tmp_path / "queue/method.yaml")
    queue = QueueConfig(
        id="partial_coverage",
        entries=[QueueEntry(id="method", config=config_path)],
    )

    waiting = QueueRunner(
        Path(__file__).resolve().parents[1]
    ).run(queue)

    assert waiting["method"]["status"] == "waiting_for_observation_review"
    assert waiting["method"]["missing_required_cameras"] == ["cam_b"]
    reviews = 0

    def reviewer(preflight, output_directory):
        nonlocal reviews
        reviews += 1
        assert preflight.missing_required_cameras == ("cam_b",)
        assert output_directory.is_dir()
        return True

    completed = QueueRunner(
        Path(__file__).resolve().parents[1],
        observation_reviewer=reviewer,
    ).run(queue)

    assert reviews == 1
    assert completed["method"]["status"] == "completed"
    result = Path(completed["method"]["result"])
    payload = json.loads((result / "RESULT.json").read_text())
    assert payload["quality_status"] == "partial_coverage"
    assert payload["metrics"]["missing_required_cameras"] == ["cam_b"]
    assert (
        result
        / "provenance"
        / "observation_detection_config.json"
    ).is_file()
