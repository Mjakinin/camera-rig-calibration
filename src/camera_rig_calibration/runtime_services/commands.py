"""Runtime implementation grouped by one cohesive responsibility."""

from __future__ import annotations

import json
import hashlib
import importlib.util
import csv
import os
import platform
import queue
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, TypeVar

from rich.console import Console
from rich.table import Table

from ..components import register_builtin_components
from ..config import config_fingerprint, load_config, save_config
from ..config.models import RigConfig, effective_observation_quality
from ..contracts import CommandSpec, RunContext
from ..dataset.manifest import AutoSelection, load_dataset_manifest, save_dataset_manifest
from ..dataset.validation import validate_dataset
from ..input.preparation import build_preparation_plan, finalize_dataset
from ..input.topics import resolve_rosbag_source
from ..intrinsics_profiles import resolve_intrinsic_profile
from ..methods.common.aruco_utils import (
    DETECTOR_CONTRACT,
    effective_detector_config,
)
from ..experiments import (
    colmap_artifact_fingerprint,
    evaluation_fingerprint,
    experiment_paths,
    input_fingerprint,
    method_config_diff,
    method_fingerprint,
    method_result_label,
    write_experiment_manifest,
)
from ..observations import (
    ResolvedSelections,
    freeze_selections,
    resolve_selections,
)
from ..observation_quality import ObservationQualityError, filter_observations
from ..progress import ProgressClock, progress_text, terminal_lines
from ..pipeline import StageContract, validate_stage_dag
from ..registry import calibration_methods, evaluators, input_adapters
from ..results import write_comparison


from .common import (
    T,
    COMMAND_HEARTBEAT_SECONDS,
    TERMINAL_PREFIXES,
    _now,
)
from .bindings import current_runtime_bindings


class CommandMixin:
    def _execute_stage(self, stage_id: str, action: Callable[[], T]) -> T | None:
        stage = self._stage_record(stage_id)
        if stage["status"] == "completed":
            self.console.print(f"[dim]Resume: skipping completed stage {stage_id}[/dim]")
            return None
        started = time.monotonic()
        self.progress.begin_stage()
        stage_index = next(
            index
            for index, candidate in enumerate(self.manifest["stages"], 1)
            if candidate["id"] == stage_id
        )
        stage_count = len(self.manifest["stages"])
        stage.update({"status": "running", "started_at": _now()})
        stage.pop("error", None)
        self._save_state()
        started_event = self.progress.event(
            event="stage_started",
            stage_id=stage_id,
            stage_name=stage["display_name"],
            stage_index=stage_index,
            stage_count=stage_count,
        )
        self.console.print(
            "\n[bold cyan]"
            + terminal_lines(started_event)[0]
            + "[/bold cyan]"
        )
        try:
            result = action()
        except Exception as exc:
            elapsed = time.monotonic() - started
            stage.update(
                {
                    "status": "failed",
                    "finished_at": _now(),
                    "runtime_seconds": elapsed,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            self.timings[stage_id] = elapsed
            self.timings.setdefault("_structured", {})[stage_id] = (
                self.progress.event(
                    event="stage_failed",
                    stage_id=stage_id,
                    stage_name=stage["display_name"],
                    stage_index=stage_index,
                    stage_count=stage_count,
                ).payload()
            )
            self.manifest["status"] = "failed"
            self.manifest["runner_pid"] = None
            self.manifest["error"] = stage["error"]
            self._save_state()
            raise
        elapsed = time.monotonic() - started
        stage.update(
            {
                "status": "completed",
                "finished_at": _now(),
                "runtime_seconds": elapsed,
            }
        )
        self.timings[stage_id] = elapsed
        event = self.progress.event(
            event="stage_completed",
            stage_id=stage_id,
            stage_name=stage["display_name"],
            stage_index=stage_index,
            stage_count=stage_count,
        )
        self.timings.setdefault("_structured", {})[stage_id] = event.payload()
        self._save_state()
        self.console.print(
            "[green]"
            + "\n".join(terminal_lines(event)[:4])
            + "[/green]"
        )
        return result

    def _run_command(self, spec: CommandSpec) -> None:
        COMMAND_HEARTBEAT_SECONDS = (
            current_runtime_bindings().command_heartbeat_seconds
        )
        assert self.run_directory is not None
        stage_manifest = (
            spec.output_directory / "stage_manifest.json"
            if spec.output_directory is not None
            else None
        )
        if stage_manifest is not None and stage_manifest.is_file():
            try:
                stage_state = json.loads(
                    stage_manifest.read_text(encoding="utf-8")
                )
            except (OSError, json.JSONDecodeError):
                stage_state = {}
            if stage_state.get("status") == "COMPLETED":
                self.console.print(
                    f"[dim]Resume: {spec.display_name} already completed; "
                    f"reusing {spec.output_directory}[/dim]"
                )
                return
        command_text = spec.shell_display()
        with (self.run_directory / "commands.txt").open("a", encoding="utf-8") as handle:
            handle.write(f"# {spec.stage_id}: {spec.display_name}\n{command_text}\n\n")
        log_path = self.run_directory / "logs" / f"{spec.stage_id}.log"
        environment = os.environ.copy()
        environment.update(spec.environment)
        environment.setdefault("PYTHONUNBUFFERED", "1")
        started = time.monotonic()
        structured_stage_starts: dict[str, float] = {}
        with log_path.open("a", encoding="utf-8") as log:
            process = subprocess.Popen(
                spec.argv,
                cwd=spec.cwd,
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert process.stdout is not None
            self.console.print(
                f"[dim]Running {spec.display_name} | PID {process.pid} | "
                f"full log: {log_path}[/dim]"
            )
            output: queue.Queue[str | None] = queue.Queue()

            def read_output() -> None:
                assert process.stdout is not None
                for child_line in process.stdout:
                    output.put(child_line)
                output.put(None)

            reader = threading.Thread(
                target=read_output,
                name=f"rigcal-output-{spec.stage_id}",
                daemon=True,
            )
            reader.start()
            last_terminal_activity = started
            try:
                while True:
                    try:
                        queued_line = output.get(timeout=0.5)
                    except queue.Empty:
                        queued_line = ""
                    now = time.monotonic()
                    if queued_line is None:
                        break
                    if not queued_line:
                        if (
                            now - last_terminal_activity
                            >= COMMAND_HEARTBEAT_SECONDS
                        ):
                            stage_elapsed = now - self.progress.stage_started
                            job_elapsed = now - self.progress.job_started
                            queue_elapsed = now - self.progress.queue_started
                            batch_text = ""
                            if self.progress.batch_started is not None:
                                batch_text = (
                                    " | Batch "
                                    f"{now - self.progress.batch_started:.1f} s"
                                )
                            counts_text = progress_text(self.progress.counts)
                            counts_suffix = (
                                f" | {counts_text}" if counts_text else ""
                            )
                            self.console.print(
                                f"Still running: "
                                f"{spec.display_name} | PID {process.pid} | "
                                f"Stage {stage_elapsed:.1f} s | "
                                f"Method/job {job_elapsed:.1f} s | "
                                f"Experiment {queue_elapsed:.1f} s"
                                f"{batch_text}{counts_suffix} | Log: {log_path}",
                                style="cyan",
                                markup=False,
                            )
                            last_terminal_activity = now
                        if process.poll() is not None and not reader.is_alive():
                            break
                        continue
                    elapsed = time.monotonic() - started
                    line = queued_line.rstrip("\n")
                    log.write(line + "\n")
                    log.flush()
                    self.progress.update_counts(line)
                    displayed = False
                    if line.startswith("RIGCAL_STAGE_START "):
                        substage = line.split(maxsplit=1)[1].strip()
                        structured_stage_starts[substage] = time.monotonic()
                        self.console.print(
                            f"[bold cyan][{self.progress.job_id}] "
                            f"{substage.replace('_', ' ')}[/bold cyan]"
                        )
                        displayed = True
                    elif line.startswith("RIGCAL_STAGE_END "):
                        pieces = line.split()
                        substage = pieces[1] if len(pieces) > 1 else "unknown"
                        measured = next(
                            (
                                float(piece.split("=", 1)[1])
                                for piece in pieces[2:]
                                if piece.startswith("elapsed_seconds=")
                            ),
                            time.monotonic()
                            - structured_stage_starts.get(
                                substage, time.monotonic()
                            ),
                        )
                        self.timings.setdefault("_sub_stages", {}).setdefault(
                            spec.stage_id, {}
                        )[substage] = measured
                        self._save_state()
                        self.console.print(
                            f"[green]{substage.replace('_', ' ')}: "
                            f"{measured:.1f} s[/green]"
                        )
                        displayed = True
                    elif line.startswith("RIGCAL_STAGE_WARNING "):
                        self.console.print(f"[yellow]{line}[/yellow]", markup=False)
                        displayed = True
                    elif line.startswith("RIGCAL_STAGE_FAILED "):
                        self.console.print(f"[red]{line}[/red]", markup=False)
                        displayed = True
                    elif line.startswith("RIGCAL_PROGRESS "):
                        summary = progress_text(self.progress.counts)
                        if summary:
                            self.console.print(
                                f"[{elapsed:8.1f}s] {spec.display_name}: {summary}",
                                markup=False,
                            )
                            displayed = True
                    elif line.startswith(TERMINAL_PREFIXES):
                        self.console.print(
                            f"[{elapsed:8.1f}s] {line}", markup=False
                        )
                        displayed = True
                    if displayed:
                        last_terminal_activity = time.monotonic()
            except KeyboardInterrupt:
                process.terminate()
                process.wait(timeout=10)
                raise
            code = process.wait()
        if code != 0:
            raise RuntimeError(
                f"{spec.display_name} exited with code {code}; log: {log_path}"
            )
        elapsed = time.monotonic() - started
        self.console.print(
            f"[green]{spec.display_name}: completed in {elapsed:.1f} s[/green] "
            f"[dim]| log: {log_path}[/dim]"
        )

    def _detector_command(self, config: RigConfig, dataset_root: Path) -> CommandSpec:
        assert self.run_directory is not None
        argv = (
            sys.executable,
            "-m",
            "camera_rig_calibration.observation_services.detection",
            "--dataset",
            str(dataset_root / "raw_images"),
            "--out",
            str(self.run_directory / "01_OBSERVATIONS"),
            "--cameras",
            ",".join(camera.id for camera in config.static_cameras),
            "--moving-camera-id",
            config.moving_camera.id,
            "--marker-length-m",
            str(config.markers.length_m),
            "--dictionary",
            config.markers.dictionary,
            "--detection-mode",
            config.markers.detection_mode,
            "--allowed-marker-ids",
            "auto",
            "--minimum-area-px2",
            "0",
        )
        return CommandSpec(
            "detect_markers",
            "Shared ArUco detection",
            argv,
            self.repository_root,
            self.run_directory / "01_OBSERVATIONS",
        )



__all__ = ['CommandMixin']
