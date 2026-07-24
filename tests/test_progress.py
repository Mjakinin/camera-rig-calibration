from __future__ import annotations

import json
import sys
from io import StringIO
from pathlib import Path

from rich.console import Console

from camera_rig_calibration.contracts import CommandSpec
from camera_rig_calibration.progress import ProgressClock, terminal_lines
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
    assert "Started PID" in output
    assert "Still running: Silent progress fixture" in output
    assert "Stage " in output
    assert "Queue " in output
    assert "ETA" not in output
