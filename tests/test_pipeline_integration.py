from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import pytest
from pydantic import BaseModel, ConfigDict

from camera_rig_calibration.config import load_config
from camera_rig_calibration.config import save_config
from camera_rig_calibration.config.models import (
    DatasetSettings,
    EvaluationSettings,
    MethodSettings,
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

from conftest import write_intrinsics


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
        methods=MethodSettings(enabled=["pipeline_dummy"]),
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
    assert resolved.evaluation.anchor_marker_id == "auto_common"
    assert resolved.markers.allowed_ids == "auto"
    dataset_manifest = json.loads(
        (run / "00_INPUT" / "dataset_manifest.json").read_text()
    )
    assert len(dataset_manifest["automatic_selections"]) == 3
    commands = (run / "commands.txt").read_text()
    assert "02_detect_shared_aruco_observations.py" in commands
    assert "07_run_ap01_real.py" not in commands
    assert "08_run_ap02_real.py" not in commands
    assert "09_run_ap03_real.py" not in commands

    prepare_only = config.model_copy(
        update={
            "project": config.project.model_copy(
                update={"run_label": "capture_test", "execution_mode": "prepare_only"}
            ),
            "methods": MethodSettings(enabled=["ap02"]),
            "evaluation": EvaluationSettings(enabled=True),
        },
        deep=True,
    )
    prepared_run = PipelineOrchestrator(Path(__file__).resolve().parents[1]).run(
        prepare_only
    )
    prepared_manifest = json.loads(
        (prepared_run / "run_manifest.json").read_text()
    )
    assert prepared_manifest["status"] == "completed"
    assert prepared_manifest["execution_mode"] == "prepare_only"
    assert not any(
        stage["id"].startswith("method_") for stage in prepared_manifest["stages"]
    )
    summary = (prepared_run / "99_FINAL_RESULTS" / "SUMMARY.txt").read_text()
    assert "Calibration methods executed: none" in summary


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
            id="shared_queue", prepared_root=dataset
        ),
        "static_cameras": [
            StaticCameraSettings(id=value) for value in camera_ids
        ],
        "moving_camera": MovingCameraSettings(id="calibration-camera"),
        "selection": SelectionSettings(mode="review_once"),
        "evaluation": EvaluationSettings(enabled=False),
    }
    configs = [
        RigConfig(
            methods=MethodSettings(
                enabled=[method.id], extensions={method.id: {}}
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
    assert {row["status"] for row in results.values()} == {"completed"}
    resolved_queue = (
        tmp_path
        / "workspace/queues/shared_preparation.resolved/queue.yaml"
    )
    assert resolved_queue.is_file()
    resolved_configs = [
        load_config(path)
        for path in resolved_queue.parent.glob("*.yaml")
        if path.name != "queue.yaml"
    ]
    assert len(resolved_configs) == 2
    assert all(
        config.selection.mode == "explicit"
        for config in resolved_configs
    )
    detector_commands = []
    for row in results.values():
        detector_commands.append(
            "02_detect_shared_aruco_observations.py"
            in (Path(row["result"]) / "commands.txt").read_text()
        )
    assert detector_commands == [False, False]
    queue_state = json.loads(
        (
            tmp_path / "workspace/queues/shared_preparation.state.json"
        ).read_text(encoding="utf-8")
    )
    preparation = Path(queue_state["preflight_preparation"])
    assert (
        (preparation / "commands.txt")
        .read_text(encoding="utf-8")
        .count("02_detect_shared_aruco_observations.py")
        == 1
    )


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
        "methods": MethodSettings(
            enabled=[method.id], extensions={method.id: {}}
        ),
        "selection": SelectionSettings(mode="auto"),
        "evaluation": EvaluationSettings(enabled=False),
    }
    ready = RigConfig(**common)
    rejected = RigConfig(
        **common,
        observation_quality={"minimum_marker_area_px2": 1_000_000.0},
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
    assert results["rejected"]["status"] == "failed_preflight"
    assert Path(results["ready"]["result"]).is_dir()
