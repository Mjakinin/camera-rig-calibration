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
        "validate_dataset",
        "detect_markers",
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

    input_root = experiment_paths(prepared_config).datasets
    assert (input_root / "raw_images/static/front-left.png").is_file()
    assert (input_root / "raw_images/moving/frame_000000.png").is_file()
    assert (input_root / "metadata/source.json").is_file()
    assert (run / "00_INPUT/raw_images").is_dir()
    assert (run / "03_AP02").is_dir()
    assert not (run / "02_AP01").exists()


def test_transaction_input_cache_is_separate_and_reuses_obsolete_working_data(
    prepared_config,
    tmp_path: Path,
) -> None:
    transaction = tmp_path / "transaction"
    obsolete = transaction / "dataset" / ".working"
    extracted = obsolete / "_acquisitions" / "capture" / "frame.png"
    extracted.parent.mkdir(parents=True)
    extracted.write_bytes(b"already extracted")
    stream = io.StringIO()
    orchestrator = PipelineOrchestrator(
        Path(__file__).resolve().parents[1],
        Console(file=stream, force_terminal=False),
        transaction_root=transaction,
    )

    working = orchestrator._input_working_root(prepared_config)

    assert working == (transaction / "input_working").resolve()
    assert (working / "_acquisitions/capture/frame.png").read_bytes() == (
        b"already extracted"
    )
    assert not (transaction / "dataset").exists()
    assert "Reusing the already extracted input" in stream.getvalue()


def test_completed_dataset_observation_evidence_is_not_rewritten_by_method(
    prepared_config, tmp_path: Path
) -> None:
    transaction = tmp_path / "transaction"
    observations = transaction / "dataset" / "observations"
    observations.mkdir(parents=True)
    (observations / "PUBLICATION_COMPLETE.json").write_text(
        json.dumps({"status": "complete"}) + "\n",
        encoding="utf-8",
    )
    candidates = observations / "SELECTION_CANDIDATES.json"
    candidates.write_text(
        json.dumps({"scope": "queue_preflight"}) + "\n",
        encoding="utf-8",
    )
    run = tmp_path / "method_run"
    run.mkdir()
    orchestrator = PipelineOrchestrator(
        Path(__file__).resolve().parents[1],
        transaction_root=transaction,
    )
    orchestrator.run_directory = run
    orchestrator.manifest = {}

    orchestrator._finalize_dataset_observations(
        prepared_config,
        quality_observations_root=tmp_path / "method_specific",
    )

    assert json.loads(candidates.read_text(encoding="utf-8")) == {
        "scope": "queue_preflight"
    }
    manifest = json.loads(
        (run / "run_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["dataset_observation_evidence_reused"] is True
