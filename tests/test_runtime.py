from __future__ import annotations

import io
import json
from pathlib import Path

import pytest
from rich.console import Console

from camera_rig_calibration.config.models import ColmapSettings, MethodSettings
from camera_rig_calibration.runtime import PipelineOrchestrator, planned_stages


def test_dry_run_does_not_create_output(prepared_config, tmp_path: Path) -> None:
    stream = io.StringIO()
    orchestrator = PipelineOrchestrator(
        Path(__file__).resolve().parents[1], Console(file=stream, force_terminal=False)
    )
    orchestrator.show_dry_run(prepared_config)
    assert "Calibration pipeline plan" in stream.getvalue()
    assert "no directories or method processes" in stream.getvalue()
    assert not prepared_config.project.output_root.exists()


def test_failed_stage_is_persisted_and_can_be_retried(prepared_config) -> None:
    orchestrator = PipelineOrchestrator(Path(__file__).resolve().parents[1])
    run = orchestrator._new_run(prepared_config)

    def fail() -> None:
        raise RuntimeError("fixture failure")

    with pytest.raises(RuntimeError, match="fixture failure"):
        orchestrator._execute_stage("prepare_inputs", fail)
    payload = json.loads((run / "run_manifest.json").read_text())
    assert payload["status"] == "failed"
    assert payload["stages"][0]["status"] == "failed"

    orchestrator._execute_stage("prepare_inputs", lambda: None)
    payload = json.loads((run / "run_manifest.json").read_text())
    assert payload["stages"][0]["status"] == "completed"


def test_prepare_only_plan_never_schedules_a_calibration_method(
    prepared_config,
) -> None:
    config = prepared_config.model_copy(
        update={
            "project": prepared_config.project.model_copy(
                update={"execution_mode": "prepare_only"}
            )
        },
        deep=True,
    )
    stage_ids = [stage_id for stage_id, _ in planned_stages(config)]
    assert stage_ids == [
        "prepare_inputs",
        "detect_markers",
        "build_debug_gallery",
        "validate_dataset",
        "observation_quality",
        "analyze_selections",
        "finalize",
    ]
    assert not any(stage_id.startswith("method_") for stage_id in stage_ids)


def test_missing_colmap_is_rejected_before_a_run_directory_is_created(
    prepared_config,
) -> None:
    config = prepared_config.model_copy(
        update={
            "methods": MethodSettings(enabled=["ap01"]),
            "colmap": ColmapSettings(executable="definitely_missing_rigcal_colmap"),
        },
        deep=True,
    )
    orchestrator = PipelineOrchestrator(Path(__file__).resolve().parents[1])

    with pytest.raises(RuntimeError, match="before any run was created"):
        orchestrator.run(config=config)

    assert not config.project.output_root.exists()


def test_new_result_dataset_contains_a_shared_input_view(prepared_config) -> None:
    orchestrator = PipelineOrchestrator(Path(__file__).resolve().parents[1])
    run = orchestrator._new_run(prepared_config)
    source = prepared_config.dataset.prepared_root
    assert source is not None

    input_id = orchestrator._publish_input_view(prepared_config, source)

    from camera_rig_calibration.experiments import experiment_paths

    input_root = experiment_paths(prepared_config).datasets / input_id
    assert (input_root / "raw_images/static/front-left.png").is_file()
    assert (input_root / "raw_images/moving/frame_000000.png").is_file()
    assert (input_root / "SOURCE.json").is_file()
    assert (run / "00_INPUT/raw_images").is_dir()
    assert (run / "03_AP02").is_dir()
    assert not (run / "02_AP01").exists()
