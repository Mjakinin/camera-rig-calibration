from __future__ import annotations

import json
import sys
import time
from io import StringIO
from pathlib import Path

from rich.console import Console

from camera_rig_calibration.contracts import CommandSpec
from camera_rig_calibration.progress import (
    ProgressClock,
    progress_text,
    terminal_lines,
)
from camera_rig_calibration.runtime import PipelineOrchestrator


def test_progress_event_contains_elapsed_seconds_and_counts(tmp_path: Path) -> None:
    clock = ProgressClock(job_id="ap03_exhaustive", job_index=2, job_count=3)
    clock.begin_stage()
    assert clock.update_counts("Registered images: 142") == {
        "registered_images": 142
    }

    event = clock.event(
        event="stage_completed",
        stage_id="colmap_mapping",
        stage_name="COLMAP mapping",
        stage_index=3,
        stage_count=8,
        log=tmp_path / "mapping.log",
    )

    lines = terminal_lines(event)
    assert lines[0] == (
        "[ap03_exhaustive] Job 2/3, Step 3/8: COLMAP mapping"
    )
    assert "Registered Images: 142" in lines
    assert all("ETA" not in line for line in lines)


def test_structured_substage_is_flushed_to_log_and_timings(
    prepared_config,
) -> None:
    orchestrator = PipelineOrchestrator(Path(__file__).resolve().parents[1])
    run = orchestrator._new_run(prepared_config)
    command = CommandSpec(
        stage_id="method_ap03",
        display_name="AP03 progress fixture",
        argv=(
            sys.executable,
            "-c",
            (
                "print('RIGCAL_STAGE_START colmap_mapping', flush=True); "
                "print('Registered images: 12', flush=True); "
                "print('RIGCAL_STAGE_END colmap_mapping "
                "elapsed_seconds=1.250', flush=True)"
            ),
        ),
        cwd=Path(__file__).resolve().parents[1],
    )

    orchestrator._run_command(command)

    log = (run / "logs/method_ap03.log").read_text(encoding="utf-8")
    assert "Registered images: 12" in log
    assert (
        orchestrator.timings["_sub_stages"]["method_ap03"]["colmap_mapping"]
        == 1.25
    )
    persisted = json.loads((run / "timings.json").read_text(encoding="utf-8"))
    assert persisted["_sub_stages"]["method_ap03"]["colmap_mapping"] == 1.25


def test_frame_and_optimizer_progress_are_compact() -> None:
    clock_start = time.monotonic()
    clock = ProgressClock(
        job_id="ap02",
        batch_started_monotonic=clock_start,
    )

    counts = clock.update_counts(
        "RIGCAL_PROGRESS current=45 total=180 unit=frames "
        "label=ArUco observations"
    )
    assert progress_text(counts) == (
        "ArUco observations: 45/180 frames (25%)"
    )
    clock.begin_stage()
    counts = clock.update_counts(
        "       8             17         1.1203e+09      1.96e+06"
    )
    assert progress_text(counts) == "optimizer iteration 8"
    event = clock.event(
        event="stage_progress",
        stage_id="ap02_ba",
        stage_name="AP02 bundle adjustment",
        stage_index=2,
        stage_count=3,
    )
    assert event.batch_elapsed_seconds is not None
    assert event.batch_elapsed_seconds >= 0
    assert clock_start <= time.monotonic()


def test_silent_subprocess_prints_periodic_runtime_heartbeat(
    prepared_config, monkeypatch
) -> None:
    monkeypatch.setattr(
        "camera_rig_calibration.runtime.COMMAND_HEARTBEAT_SECONDS",
        0.05,
    )
    stream = StringIO()
    orchestrator = PipelineOrchestrator(
        Path(__file__).resolve().parents[1],
        console=Console(file=stream, force_terminal=False, width=240),
    )
    orchestrator._new_run(prepared_config)
    command = CommandSpec(
        stage_id="silent_fixture",
        display_name="Silent progress fixture",
        argv=(
            sys.executable,
            "-c",
            "import time; time.sleep(0.65)",
        ),
        cwd=Path(__file__).resolve().parents[1],
    )

    orchestrator._run_command(command)

    output = stream.getvalue()
    assert "Running Silent progress fixture" in output
    assert "Still running: Silent progress fixture" in output
    assert "Stage " in output
    assert "Method/job " in output
    assert "Experiment " in output
    assert "ETA" not in output
