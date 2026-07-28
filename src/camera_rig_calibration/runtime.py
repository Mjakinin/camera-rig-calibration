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

from .components import register_builtin_components
from .config import config_fingerprint, load_config, save_config
from .config.models import RigConfig
from .contracts import CommandSpec, RunContext
from .dataset.manifest import AutoSelection, load_dataset_manifest, save_dataset_manifest
from .dataset.validation import validate_dataset
from .input.preparation import build_preparation_plan, finalize_dataset
from .input.topics import resolve_rosbag_source
from .intrinsics_profiles import resolve_intrinsic_profile
from .methods.common.aruco_utils import (
    DETECTOR_CONTRACT,
    effective_detector_config,
)
from .experiments import (
    colmap_artifact_fingerprint,
    evaluation_fingerprint,
    experiment_paths,
    input_fingerprint,
    method_config_diff,
    method_fingerprint,
    method_result_label,
    write_experiment_manifest,
)
from .observations import (
    ResolvedSelections,
    ap03_candidate_rank,
    freeze_selections,
    resolve_selections,
)
from .observation_quality import ObservationQualityError, filter_observations
from .progress import ProgressClock, progress_text, terminal_lines
from .pipeline import StageContract, validate_stage_dag
from .registry import calibration_methods, evaluators, input_adapters
from .results import write_comparison


T = TypeVar("T")
COMMAND_HEARTBEAT_SECONDS = 10.0
BASE_RUN_DIRECTORIES = (
    "00_INPUT",
    "01_OBSERVATIONS",
    "preflight",
    "06_EVALUATION",
    "07_COMPARISON",
    "99_FINAL_RESULTS",
    "logs",
)
METHOD_DIRECTORIES = {
    "ap01": "02_AP01",
    "ap02": "03_AP02",
    "ap03": "04_AP03",
}
TERMINAL_PREFIXES = (
    "[OK]",
    "[WARN]",
    "[WARNING]",
    "[ERROR]",
    "[INFO]",
    "[REUSE]",
    "ERROR:",
    "WARNING:",
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _run_id(config: RigConfig) -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{stamp}_{config.project.run_label}_{config_fingerprint(config)[:8]}"


def _run_directories(config: RigConfig) -> tuple[str, ...]:
    selected = tuple(
        METHOD_DIRECTORIES[method_id]
        for method_id in config.methods.enabled
        if method_id in METHOD_DIRECTORIES
    )
    return (*BASE_RUN_DIRECTORIES[:2], *selected, *BASE_RUN_DIRECTORIES[2:])


def _automatic_scientific_selections(config: RigConfig) -> bool:
    return (
        config.methods.ap01.root_camera == "auto"
        and config.methods.ap02.reference_marker_id == "auto"
        and config.methods.ap03_single.scale_marker_id == "auto"
        and config.methods.ap03_multi.marker_ids == "auto"
        and config.evaluation.anchor_marker_id == "auto_common"
    )


def _materialize_tree(source: Path, destination: Path) -> dict[str, int]:
    """Place immutable input files in results, using hardlinks when possible."""
    counts = {"hardlinked": 0, "copied": 0, "existing": 0}
    if not source.is_dir():
        return counts
    for item in sorted(source.rglob("*")):
        relative = item.relative_to(source)
        target = destination / relative
        if item.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        if not item.is_file():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if target.stat().st_size != item.stat().st_size:
                raise RuntimeError(
                    f"Published input conflicts with the canonical dataset: {target}"
                )
            counts["existing"] += 1
            continue
        try:
            os.link(item, target)
            counts["hardlinked"] += 1
        except OSError:
            shutil.copy2(item, target)
            counts["copied"] += 1
    return counts


def planned_stages(
    config: RigConfig, *, defer_evaluation: bool = False
) -> list[tuple[str, str]]:
    stages = [
        ("prepare_inputs", "Prepare canonical inputs and provenance"),
        ("validate_dataset", "Validate the canonical dataset"),
        ("detect_markers", "Detect shared ArUco observations and debug images"),
        (
            "observation_quality",
            "Apply immutable checks and job-specific observation quality",
        ),
        ("analyze_selections", "Analyze root-camera and method-marker candidates"),
    ]
    if config.project.execution_mode == "prepare_only":
        stages.append(("finalize", "Write input-preparation report"))
        return stages
    stages.extend(
        (f"method_{method_id}", calibration_methods.get(method_id).display_name)
        for method_id in config.methods.enabled
    )
    if config.evaluation.enabled and not defer_evaluation:
        stages.append(
            (
                "resolve_evaluation_anchor",
                "Resolve a common post-method evaluation anchor",
            )
        )
        stages.append(("evaluation", "Common method evaluation"))
    stages.extend(
        [
            ("comparison", "Normalize and compare method results"),
            ("finalize", "Write final report"),
        ]
    )
    return stages


def observation_id(config: RigConfig) -> str:
    """Content ID for one versioned ArUco observation contract."""
    payload = {
        "dictionary": config.markers.dictionary,
        "length_m": config.markers.length_m,
        "detection": effective_detector_config(
            config.markers.detection_mode,
            config.markers.dictionary,
        ),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return (
        "detection_"
        + hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]
    )


class PipelineOrchestrator:
    def __init__(
        self,
        repository_root: Path,
        console: Console | None = None,
        selection_reviewer: Callable[
            [RigConfig, ResolvedSelections, Path], dict[str, Any]
        ]
        | None = None,
        defer_evaluation: bool = False,
        job_id: str | None = None,
        job_index: int = 1,
        job_count: int = 1,
        queue_started_monotonic: float | None = None,
        batch_started_monotonic: float | None = None,
        transaction_root: Path | None = None,
    ) -> None:
        self.repository_root = repository_root.resolve()
        self.console = console or Console()
        self.selection_reviewer = selection_reviewer
        self.defer_evaluation = defer_evaluation
        self.transaction_root = (
            transaction_root.resolve()
            if transaction_root is not None
            else None
        )
        self.progress = ProgressClock(
            job_id=job_id or "rigcal",
            job_index=job_index,
            job_count=job_count,
            queue_started_monotonic=queue_started_monotonic,
            batch_started_monotonic=batch_started_monotonic,
        )
        register_builtin_components()
        self.run_directory: Path | None = None
        self.manifest: dict[str, Any] = {}
        self.timings: dict[str, Any] = {}

    def _working_paths(self, config: RigConfig):
        paths = experiment_paths(config)
        if self.transaction_root is None:
            return paths
        result_root = self.transaction_root / "results"
        dataset_root = self.transaction_root / "dataset"
        return replace(
            paths,
            root=result_root,
            dataset_root=dataset_root,
            datasets=dataset_root,
            methods=result_root / "methods",
            evaluations=result_root / "evaluations",
            comparisons=result_root,
            attempts=result_root / "attempts",
            artifacts=self.transaction_root / "artifacts",
            staging=self.transaction_root / "jobs",
        )

    def _input_working_root(self, config: RigConfig) -> Path:
        """Keep extraction caches outside the dataset publication directory."""
        if self.transaction_root is None:
            return (
                self._working_paths(config).staging / "input_working"
            ).resolve()

        working = (self.transaction_root / "input_working").resolve()
        legacy = (self.transaction_root / "dataset" / ".working").resolve()
        if legacy.is_dir():
            if working.exists():
                raise RuntimeError(
                    "Both the current and obsolete input working directories "
                    f"exist: {working} and {legacy}. Refusing to merge them."
                )
            working.parent.mkdir(parents=True, exist_ok=True)
            legacy.rename(working)
            legacy_parent = legacy.parent
            if legacy_parent.is_dir() and not any(legacy_parent.iterdir()):
                legacy_parent.rmdir()
            self.console.print(
                "[green]Reusing the already extracted input from the "
                "interrupted queue preflight.[/green]"
            )
        return working

    def _validate_components(self, config: RigConfig) -> None:
        builtin_methods = {"ap01", "ap02", "ap03"}
        if (
            config.project.execution_mode == "complete"
            and len(config.methods.enabled) != 1
        ):
            raise RuntimeError(
                "A schema-v5 execution contains exactly one method job. "
                "Put multiple method configurations into a rigcal queue."
            )
        if builtin_methods.intersection(config.methods.enabled) and len(
            config.static_cameras
        ) < 2:
            raise RuntimeError(
                "AP01/AP02/AP03 require at least two declared static cameras"
            )
        matching = [adapter for adapter in input_adapters if adapter.matches(config)]
        if not matching:
            raise RuntimeError("No registered input adapter accepts this configuration")
        adapter_result = matching[0].requirements(config)
        if not adapter_result.compatible:
            raise RuntimeError("Input is incompatible: " + "; ".join(adapter_result.reasons))
        for method_id in config.methods.enabled:
            method = calibration_methods.get(method_id)
            if method_id not in {"ap01", "ap02", "ap03"}:
                method.config_model.model_validate(
                    config.methods.extensions.get(method_id, {})
                )

    def validate_ready(self, config: RigConfig) -> None:
        """Fail before creating a run when an executable dependency is missing."""
        self._validate_components(config)
        missing: list[str] = []
        if (
            config.moving_camera.intrinsics_profile
            and config.moving_camera.intrinsics is None
            and config.moving_camera.intrinsic_calibration_video is None
            and config.moving_camera.intrinsic_calibration_images is None
        ):
            try:
                resolve_intrinsic_profile(
                    self.repository_root,
                    config.moving_camera.intrinsics_profile,
                )
            except (FileNotFoundError, ValueError) as exc:
                missing.append(str(exc))
        if any(
            method_id in {"ap01", "ap03"}
            for method_id in config.methods.enabled
        ) and config.project.execution_mode == "complete":
            executable = (
                "colmap"
                if config.colmap.executable == "auto"
                else config.colmap.executable
            )
            explicit = Path(executable).expanduser()
            available = (
                explicit.is_file()
                if explicit.is_absolute() or "/" in executable
                else shutil.which(executable) is not None
            )
            if not available:
                missing.append(
                    f"COLMAP executable '{executable}' (needed by AP01/AP03)"
                )
            if (
                config.colmap.gpu_mode == "true"
                and not self._compatible_gpu_available()
            ):
                missing.append(
                    "a compatible NVIDIA GPU/driver (COLMAP gpu_mode=true)"
                )
        if config.simulation.enabled:
            for command in ("ign", "ros2"):
                if shutil.which(command) is None:
                    missing.append(
                        f"'{command}' command (needed for a new Gazebo capture)"
                    )
            for module in ("rclpy", "sensor_msgs", "cv_bridge"):
                if importlib.util.find_spec(module) is None:
                    missing.append(
                        f"Python module '{module}' (needed for a new Gazebo capture)"
                    )
        if config.mcap.path is not None:
            for module in ("rosbag2_py", "rclpy", "sensor_msgs", "cv_bridge"):
                if importlib.util.find_spec(module) is None:
                    missing.append(
                        f"Python module '{module}' (needed to extract the ROS recording)"
                    )
            if importlib.util.find_spec("rosbag2_py") is not None:
                try:
                    import rosbag2_py

                    storage_id = resolve_rosbag_source(
                        config.mcap.path
                    ).storage_id
                    readers = set(rosbag2_py.get_registered_readers())
                    if storage_id and storage_id not in readers:
                        missing.append(
                            f"ROS 2 storage plugin '{storage_id}' (install "
                            f"ros-humble-rosbag2-storage-{storage_id})"
                        )
                except (ImportError, AttributeError, RuntimeError) as exc:
                    missing.append(
                        f"readable ROS 2 storage plugin information ({exc})"
                    )
        if missing:
            raise RuntimeError(
                "Installation check failed before any run was created:\n- "
                + "\n- ".join(missing)
                + "\nUse rigcal → Check installation, install the missing component, "
                "or remove the incompatible queue job."
            )

    @staticmethod
    def _compatible_gpu_available() -> bool:
        executable = shutil.which("nvidia-smi")
        if executable is None:
            return False
        probe = subprocess.run(
            [executable, "-L"],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
        return probe.returncode == 0 and "GPU" in probe.stdout

    def _resolve_colmap_environment(self, config: RigConfig) -> RigConfig:
        if not any(
            method_id in {"ap01", "ap03"}
            for method_id in config.methods.enabled
        ):
            return config
        requested = config.colmap
        executable_text = (
            "colmap" if requested.executable == "auto" else requested.executable
        )
        explicit = Path(executable_text).expanduser()
        executable = (
            explicit.resolve()
            if explicit.is_file()
            else Path(shutil.which(executable_text) or executable_text).resolve()
        )
        if not executable.is_file():
            raise RuntimeError(
                f"COLMAP executable could not be resolved: {requested.executable}"
            )
        gpu_available = self._compatible_gpu_available()
        if requested.gpu_mode == "true" and not gpu_available:
            raise RuntimeError(
                "COLMAP gpu_mode=true but the lightweight NVIDIA capability "
                "probe found no compatible GPU"
            )
        resolved_gpu = (
            "true"
            if requested.gpu_mode == "true"
            or (requested.gpu_mode == "auto" and gpu_available)
            else "false"
        )
        version_probe = subprocess.run(
            [str(executable), "--version"],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
        version = (
            version_probe.stdout.strip()
            or version_probe.stderr.strip()
            or "unknown"
        ).splitlines()[0]
        method_id = next(iter(config.methods.enabled), "")
        requested_label = method_result_label(config, method_id)
        resolved = config.model_copy(
            update={
                "project": config.project.model_copy(
                    update={"run_label": requested_label}
                ),
                "colmap": requested.model_copy(
                    update={
                        "executable": str(executable),
                        "gpu_mode": resolved_gpu,
                    }
                )
            },
            deep=True,
        )
        self.manifest["colmap_resolution"] = {
            "requested_executable": requested.executable,
            "resolved_executable": str(executable),
            "version": version,
            "requested_gpu_mode": requested.gpu_mode,
            "resolved_gpu_mode": resolved_gpu,
            "gpu_probe_available": gpu_available,
        }
        self._save_state()
        return RigConfig.model_validate(resolved.model_dump(mode="python"))

    def show_dry_run(self, config: RigConfig) -> None:
        self._validate_components(config)
        paths = experiment_paths(config)
        run = paths.staging / (
            f"<timestamp>_{config.project.run_label}_{config_fingerprint(config)[:8]}"
        )
        table = Table(title="Calibration pipeline plan")
        table.add_column("Stage")
        table.add_column("Action")
        for stage_id, display in planned_stages(
            config, defer_evaluation=self.defer_evaluation
        ):
            table.add_row(stage_id, display)
        self.console.print(table)
        self.console.print(f"Dataset: {config.dataset.id}")
        self.console.print(
            "Static cameras: " + ", ".join(camera.id for camera in config.static_cameras)
        )
        self.console.print("Methods: " + ", ".join(config.methods.enabled))
        self.console.print("Observations: all passing quality checks")
        self.console.print(f"Selection handling: {config.selection.mode}")
        self.console.print(f"Planned run directory: {run}")
        self.console.print(
            "Completed result root: "
            f"{paths.methods}/<method>/<queue-label>"
        )
        self.console.print(
            "Dry run complete: no directories or method processes were created for a "
            "run. The saved configuration remains available."
        )

    def _new_run(self, config: RigConfig) -> Path:
        paths = self._working_paths(config)
        base = paths.staging
        if self.transaction_root is not None:
            base = base / self.progress.job_id
        run_id = _run_id(config)
        run = base / run_id
        suffix = 2
        while run.exists():
            run = base / f"{run_id}_{suffix}"
            suffix += 1
        for directory in _run_directories(config):
            (run / directory).mkdir(parents=True, exist_ok=False)
        save_config(config, run / "requested_config.yaml")
        save_config(config, run / "resolved_config.yaml")
        stages = [
            {"id": stage_id, "display_name": display, "status": "pending"}
            for stage_id, display in planned_stages(
                config, defer_evaluation=self.defer_evaluation
            )
        ]
        self.manifest = {
            "schema_version": 5,
            "run_id": run.name,
            "execution_id": run.name,
            "experiment_id": paths.experiment_id,
            "canonical_experiment_root": str(
                experiment_paths(config).root
            ),
            "transaction_root": (
                str(self.transaction_root)
                if self.transaction_root is not None
                else None
            ),
            "result_category": paths.category,
            "dataset_id": config.dataset.id,
            "scene_type": config.dataset.scene_type.value,
            "status": "running",
            "created_at": _now(),
            "updated_at": _now(),
            "config_sha256": config_fingerprint(config),
            "requested_config_sha256": config_fingerprint(config),
            "enabled_methods": config.methods.enabled,
            "observation_input_contract": "all_quality_passed_v1",
            "execution_mode": config.project.execution_mode,
            "selection_mode": config.selection.mode,
            "evaluation_deferred_to_queue": self.defer_evaluation,
            "runner_pid": os.getpid(),
            "simulation_parameters": (
                {
                    "route": config.simulation.route_name,
                    "moving_width": config.simulation.moving_width,
                    "moving_height": config.simulation.moving_height,
                    "moving_hfov_deg": config.simulation.moving_hfov_deg,
                    "lighting": config.simulation.lighting,
                    "lighting_scale": config.simulation.lighting_scale,
                    "motion_blur_kernel": config.simulation.motion_blur_kernel,
                    "motion_blur_angle_deg": config.simulation.motion_blur_angle_deg,
                    "target_route_frames": config.simulation.target_route_frames,
                    "route_sampling_strategy": config.simulation.route_sampling_strategy,
                    "settle_seconds": config.simulation.settle_seconds,
                    "post_pose_skip": config.simulation.post_pose_skip,
                    "frame_timeout_seconds": (
                        config.simulation.frame_timeout_seconds
                    ),
                    "startup_timeout_seconds": (
                        config.simulation.startup_timeout_seconds
                    ),
                }
                if config.dataset.category.value == "simulation"
                else {}
            ),
            "stages": stages,
        }
        self.run_directory = run
        self._save_state()
        _write_json(run / "environment.json", self._environment())
        (run / "commands.txt").write_text("", encoding="utf-8")
        (run / "RUN_LAYOUT.txt").write_text(
            "RIGCAL TEMPORARY METHOD WORKSPACE\n"
            "=================\n\n"
            "This directory is an internal, resumable workspace. Its numbered "
            "sub-stages describe execution order and are never exposed as the "
            "public result layout.\n\n"
            "After validation, publication exports:\n"
            "  RESULT.txt/json and camera_extrinsics.csv\n"
            "  diagnostics/ (preflight, method and evaluation internals)\n"
            "  logs/ (complete child-process output)\n"
            "  provenance/ (configs, commands, environment and timings)\n\n"
            f"Canonical dataset: {paths.datasets}\n",
            encoding="utf-8",
        )
        return run

    def _publish_input_view(
        self,
        config: RigConfig,
        dataset_root: Path,
        dataset_manifest=None,
    ) -> str:
        """Publish an immutable, content-addressed input once per experiment."""
        assert self.run_directory is not None
        run = self.run_directory
        paths = self._working_paths(config)
        input_id = input_fingerprint(dataset_manifest, dataset_root)
        published = paths.datasets
        pointer_path = published / "metadata" / "source.json"
        descriptor_path = published / "dataset.json"
        source = dataset_root.resolve()
        reuse_existing = False
        if descriptor_path.is_file():
            descriptor = json.loads(
                descriptor_path.read_text(encoding="utf-8")
            )
            existing_input = str(
                descriptor.get("input_fingerprint", "")
            )
            if existing_input != input_id:
                raise RuntimeError(
                    f"Experiment '{paths.experiment_id}' already contains a "
                    "different immutable dataset. Choose a new experiment ID."
                )
            if not pointer_path.is_file() or not (
                published / "raw_images"
            ).is_dir():
                raise RuntimeError(
                    f"Existing dataset is incomplete: {published}"
                )
            reuse_existing = True
        elif published.is_dir() and any(published.iterdir()):
            raise RuntimeError(
                f"Dataset directory has files but no dataset descriptor: {published}. "
                "Refusing to mix an unknown input."
            )
        if not reuse_existing:
            published.mkdir(parents=True, exist_ok=True)
            totals = {"hardlinked": 0, "copied": 0, "existing": 0}
            for directory in ("raw_images", "metadata"):
                counts = _materialize_tree(
                    source / directory, published / directory
                )
                for key, value in counts.items():
                    totals[key] += value
            _write_json(
                pointer_path,
                {
                    "input_id": input_id,
                    "layout_version": 2,
                    "canonical_source_roots": [str(source)],
                    "status": "ready",
                    "published_at": _now(),
                    "storage": (
                        "hardlinks where supported, byte copies otherwise; "
                        "deleting the source does not remove files from this "
                        "dataset"
                    ),
                    "file_counts": totals,
                    "content_addressing": "normalized input SHA-256",
                },
            )
            if dataset_manifest is not None:
                save_dataset_manifest(
                    dataset_manifest,
                    published / "metadata" / "dataset_manifest.json",
                )
            (published / "README.txt").write_text(
                "Canonical immutable rigcal dataset (layout version 2).\n"
                "raw_images/ contains static, moving and camera_info inputs.\n"
                "observations/ contains shared ArUco CSVs, quality decisions "
                "and the single debug_images collection. metadata/ contains provenance and "
                "validation details.\n"
                "Calibration methods never modify this directory.\n",
                encoding="utf-8",
            )
        run_view = run / "00_INPUT" / "raw_images"
        if not run_view.exists() and not run_view.is_symlink():
            try:
                run_view.symlink_to(
                    (published / "raw_images").resolve(), target_is_directory=True
                )
            except OSError:
                (run / "00_INPUT" / "RAW_IMAGES_LOCATION.txt").write_text(
                    str(published / "raw_images") + "\n", encoding="utf-8"
                )
        write_experiment_manifest(config, paths, input_id)
        self.manifest["input_id"] = input_id
        self.manifest["experiment_root"] = str(paths.root)
        self._save_state()
        return input_id

    def _observation_id(self, config: RigConfig) -> str:
        return observation_id(config)

    def detect_observations_only(
        self,
        config: RigConfig,
        *,
        dataset_root: Path,
        run_directory: Path,
    ) -> Path:
        """Re-run only ArUco detection on an already normalized dataset."""
        run = run_directory.resolve()
        run.mkdir(parents=True, exist_ok=True)
        (run / "logs").mkdir(exist_ok=True)
        commands = run / "commands.txt"
        if not commands.is_file():
            commands.write_text("", encoding="utf-8")
        self.run_directory = run
        self._run_command(
            self._detector_command(config, dataset_root.resolve())
        )
        return run / "01_OBSERVATIONS"

    def _bind_observations_view(
        self, config: RigConfig, input_id: str
    ) -> Path:
        assert self.run_directory is not None
        paths = self._working_paths(config)
        shared = paths.datasets / "observations"
        shared.mkdir(parents=True, exist_ok=True)
        observation_id = self._observation_id(config)
        existing_config = _read_json(shared / "detection_config.json")
        existing_observation_id = existing_config.get("observation_id")
        existing_csv = shared / "shared_all_aruco_observations.csv"
        if (
            existing_csv.is_file()
            and existing_observation_id
            and existing_observation_id != observation_id
        ):
            raise RuntimeError(
                "This experiment already contains observations generated with "
                "a different ArUco detector contract. Use a distinct experiment "
                f"ID (recommended suffix: __aruco_{config.markers.detection_mode}) "
                "instead of overwriting scientific evidence."
            )
        _write_json(
            shared / "detection_config.json",
            {
                "schema_version": 5,
                "layout_version": 2,
                "input_id": input_id,
                "observation_id": observation_id,
                "markers": config.markers.model_dump(mode="json"),
                "effective_detector": effective_detector_config(
                    config.markers.detection_mode,
                    config.markers.dictionary,
                ),
                "detector_contract": DETECTOR_CONTRACT,
                "observation_input_contract": "all_quality_passed_v1",
            },
        )
        view = self.run_directory / "01_OBSERVATIONS"
        if view.is_symlink():
            if view.resolve() != shared.resolve():
                raise RuntimeError(
                    f"Run is already bound to different observations: {view.resolve()}"
                )
        else:
            if view.is_dir() and any(view.iterdir()):
                _materialize_tree(view, shared)
                shutil.rmtree(view)
            if view.is_dir():
                view.rmdir()
            view.symlink_to(shared.resolve(), target_is_directory=True)
        self.manifest["observation_id"] = observation_id
        self.manifest["observations_root"] = str(shared)
        self._save_state()
        return shared

    def _finalize_dataset_observations(
        self,
        config: RigConfig,
        *,
        quality_observations_root: Path,
    ) -> None:
        """Freeze selection and quality evidence into the immutable dataset."""
        assert self.run_directory is not None
        dataset_root = self._working_paths(config).datasets
        observations = dataset_root / "observations"
        observations.mkdir(parents=True, exist_ok=True)
        required_selection = (
            "SELECTION_CANDIDATES.json",
            "REFERENCE_SELECTIONS.json",
            "REFERENCE_MARKER_ID.txt",
        )
        missing = [
            name
            for name in required_selection
            if not (quality_observations_root / name).is_file()
        ]
        if missing:
            raise RuntimeError(
                "Selection analysis completed without publishable evidence: "
                + ", ".join(missing)
            )
        for name in required_selection:
            shutil.copy2(
                quality_observations_root / name,
                observations / name,
            )

        quality = observations / "quality"
        quality.mkdir(parents=True, exist_ok=True)
        for name in (
            "accepted_observations.csv",
            "rejected_observations.csv",
            "observation_filter_summary.json",
            "preflight_summary.json",
        ):
            source = self.run_directory / "preflight" / name
            if source.is_file():
                shutil.copy2(source, quality / name)
        manifest = self.run_directory / "00_INPUT" / "dataset_manifest.json"
        if manifest.is_file():
            destination = dataset_root / "metadata" / "dataset_manifest.json"
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(manifest, destination)
        completion = observations / "PUBLICATION_COMPLETE.json"
        # A queue shares one immutable prepared dataset across all method
        # jobs.  Its first/preflight finalization is authoritative; rewriting
        # the timestamp for every method would make byte-identical
        # publication look like a dataset conflict.
        existing_completion = _read_json(completion)
        if existing_completion.get("status") != "complete":
            _write_json(
                completion,
                {
                    "schema_version": 5,
                    "layout_version": 2,
                    "status": "complete",
                    "selection_files": list(required_selection),
                    "quality_directory": "quality",
                    "debug_images": (
                        "debug_images"
                        if (observations / "debug_images").is_dir()
                        else None
                    ),
                    "finalized_at": _now(),
                },
            )

    @staticmethod
    def _observation_contract_ready(
        root: Path,
        expected_observation_id: str | None = None,
    ) -> bool:
        paths = [
            root / name
            for name in (
                "shared_static_aruco_observations.csv",
                "shared_moving_aruco_observations.csv",
                "shared_all_aruco_observations.csv",
            )
        ]
        if not all(path.is_file() for path in paths):
            return False
        if expected_observation_id is not None:
            config = _read_json(root / "detection_config.json")
            if config.get("observation_id") != expected_observation_id:
                return False
        try:
            with paths[-1].open(newline="", encoding="utf-8") as handle:
                fields = set(next(csv.reader(handle)))
        except (OSError, StopIteration):
            return False
        return {
            "detection_success",
            "detection_mode",
            "detection_source",
            "detector_contract",
            "opencv_version",
            "pnp_reprojection_rmse_px",
            "corner0_u",
            "corner3_v",
        }.issubset(fields)

    def _execution_target(
        self,
        config: RigConfig,
        input_id: str,
        resolved: ResolvedSelections,
    ) -> tuple[Path, str, str]:
        method_id = config.methods.enabled[0]
        variant = method_result_label(config, method_id)
        target = (
            experiment_paths(config).methods
            / method_id
            / variant
        )
        return target, method_id, variant

    def _colmap_artifact_paths(
        self, config: RigConfig, input_id: str, method_id: str
    ) -> tuple[Path, Path] | None:
        if method_id == "ap01":
            relative = Path("02_AP01/01_moving_colmap")
            family = "ap01_moving"
        elif method_id == "ap03":
            relative = Path("04_AP03/colmap")
            family = "ap03_grouped"
        else:
            return None
        fingerprint = colmap_artifact_fingerprint(
            config, method_id, input_id
        )
        cache = (
            self._working_paths(config).artifacts
            / "colmap"
            / family
            / fingerprint
        )
        assert self.run_directory is not None
        return cache, self.run_directory / relative

    def _seed_colmap_artifact(
        self, config: RigConfig, input_id: str, method_id: str
    ) -> bool:
        paths = self._colmap_artifact_paths(config, input_id, method_id)
        if paths is None:
            return False
        cache, destination = paths
        complete = cache / "ARTIFACT.json"
        if (
            not complete.is_file()
            and self.transaction_root is not None
        ):
            family = cache.parent.name
            canonical_cache = (
                experiment_paths(config).artifacts
                / "colmap"
                / family
                / cache.name
            )
            if (canonical_cache / "ARTIFACT.json").is_file():
                _materialize_tree(canonical_cache, cache)
        if not complete.is_file():
            return False
        _materialize_tree(cache / "data", destination)
        self.manifest["reused_artifacts"] = {
            "colmap": str(cache),
            "reason": (
                "same normalized input and COLMAP configuration; root/marker "
                "selection is downstream"
            ),
        }
        self._save_state()
        self.console.print(
            f"[dim]Reusing compatible COLMAP artifact: {cache}[/dim]"
        )
        return True

    def _store_colmap_artifact(
        self, config: RigConfig, input_id: str, method_id: str
    ) -> None:
        paths = self._colmap_artifact_paths(config, input_id, method_id)
        if paths is None:
            return
        cache, source = paths
        if not source.is_dir():
            return
        data = cache / "data"
        _materialize_tree(source, data)
        _write_json(
            cache / "ARTIFACT.json",
            {
                "schema_version": 5,
                "stage": "colmap",
                "method_family": (
                    "ap01_moving"
                    if method_id == "ap01"
                    else "ap03_grouped"
                ),
                "input_id": input_id,
                "fingerprint": cache.name,
                "source_execution": str(self.run_directory),
                "stored_at": _now(),
            },
        )

    def _matching_completed_execution(
        self,
        target: Path,
        *,
        method_sha: str,
        input_id: str,
    ) -> bool:
        for manifest_path in (
            target / "RESULT.json",
            target / "provenance" / "run_manifest.json",
            target / "run_manifest.json",
        ):
            if not manifest_path.is_file():
                continue
            try:
                payload = json.loads(
                    manifest_path.read_text(encoding="utf-8")
                )
            except Exception:
                continue
            status = str(payload.get("status", "")).lower()
            fingerprint = payload.get(
                "method_fingerprint", payload.get("config_fingerprint")
            )
            stored_input = payload.get(
                "input_fingerprint", payload.get("input_id")
            )
            return (
                status in {"available", "completed"}
                and fingerprint == method_sha
                and stored_input == input_id
            )
        return False

    @staticmethod
    def _archive_compact_history(current: Path, history: Path) -> None:
        history.mkdir(parents=True, exist_ok=False)
        for name in (
            "run_manifest.json",
            "requested_config.yaml",
            "resolved_config.yaml",
            "timings.json",
            "commands.txt",
            "environment.json",
        ):
            source = current / name
            if source.is_file():
                shutil.copy2(source, history / name)
        for source in (
            current / "99_FINAL_RESULTS" / "SUMMARY.json",
            current / "99_FINAL_RESULTS" / "SUMMARY.txt",
        ):
            if source.is_file():
                destination = history / "summary" / source.name
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)

    def _publish_completed_execution(
        self,
        config: RigConfig,
        input_id: str,
        resolved: ResolvedSelections,
    ) -> Path:
        assert self.run_directory is not None
        staging = self.run_directory
        canonical_target, method_id, variant = self._execution_target(
            config, input_id, resolved
        )
        target = canonical_target
        if self.transaction_root is not None:
            target = (
                self.transaction_root
                / "jobs"
                / self.progress.job_id
                / "completed"
            )
            self.manifest["intended_result_target"] = str(
                canonical_target
            )
        target.parent.mkdir(parents=True, exist_ok=True)
        old: Path | None = None
        if target.exists():
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            history_root = target.parent / "run_history"
            history = history_root / f"{stamp}_{target.name}"
            suffix = 2
            while history.exists():
                history = history_root / f"{stamp}_{target.name}_{suffix}"
                suffix += 1
            self._archive_compact_history(target, history)
            old = target.with_name(f".previous_{os.getpid()}_{time.time_ns()}")
            target.rename(old)
        try:
            staging.rename(target)
        except Exception:
            if old is not None and old.exists() and not target.exists():
                old.rename(target)
            raise
        if old is not None and old.exists():
            shutil.rmtree(old)
        self.run_directory = target
        self.manifest["published_result"] = str(target)
        self.manifest["method_id"] = method_id
        self.manifest["variant"] = variant
        self._save_state()
        return target

    def _publish_preparation(
        self, config: RigConfig, input_id: str
    ) -> Path:
        assert self.run_directory is not None
        staging = self.run_directory
        if self.transaction_root is not None:
            root = (
                self.transaction_root
                / "jobs"
                / self.progress.job_id
                / "prepared"
            )
            root.parent.mkdir(parents=True, exist_ok=True)
            if root.exists():
                shutil.rmtree(root)
            staging.rename(root)
            self.run_directory = root
            self.manifest["published_result"] = str(root)
            self._save_state()
            return root

        dataset = experiment_paths(config).dataset_root
        provenance = dataset / "metadata" / "preparation"
        provenance.mkdir(parents=True, exist_ok=True)
        for name in (
            "run_manifest.json",
            "requested_config.yaml",
            "resolved_config.yaml",
            "commands.txt",
            "environment.json",
            "timings.json",
        ):
            source = staging / name
            if source.is_file():
                shutil.copy2(source, provenance / name)
        self.manifest["published_result"] = str(dataset)
        self.manifest["status"] = "completed"
        _write_json(provenance / "run_manifest.json", self.manifest)
        shutil.rmtree(staging)
        self.run_directory = dataset
        return dataset

    def _load_run(self, run: Path) -> RigConfig:
        self.run_directory = run.resolve()
        manifest_path = self.run_directory / "run_manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Run manifest not found: {manifest_path}")
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        previous_pid = int(self.manifest.get("runner_pid") or 0)
        if previous_pid > 0 and previous_pid != os.getpid():
            try:
                os.kill(previous_pid, 0)
                command = (
                    Path(f"/proc/{previous_pid}/cmdline")
                    .read_bytes()
                    .replace(b"\0", b" ")
                )
            except OSError:
                command = b""
            if b"rigcal" in command or b"camera_rig_calibration" in command:
                raise RuntimeError(
                    f"Run is already active in process {previous_pid}; refusing a "
                    "second concurrent resume."
                )
        timings_path = self.run_directory / "timings.json"
        self.timings = (
            json.loads(timings_path.read_text(encoding="utf-8"))
            if timings_path.is_file()
            else {}
        )
        resolved_path = self.run_directory / "resolved_config.yaml"
        config = load_config(resolved_path)
        expected = config_fingerprint(config)
        if expected != self.manifest.get("config_sha256"):
            if self.manifest.get("resolution_update_pending"):
                self.manifest["config_sha256"] = expected
                self.manifest["resolved_config_sha256"] = expected
                self.manifest.pop("resolution_update_pending", None)
            else:
                raise RuntimeError(
                    "Resolved configuration differs from the run manifest; refusing an "
                    "ambiguous resume. Create an experiment run instead."
                )
        self.validate_ready(config)
        self.manifest["status"] = "running"
        self.manifest["runner_pid"] = os.getpid()
        self.manifest.pop("error", None)
        self._save_state()
        return config

    def _environment(self) -> dict[str, Any]:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=self.repository_root,
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()
        branch = subprocess.run(
            ["git", "branch", "--show-current"],
            cwd=self.repository_root,
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()
        versions = {}
        for module_name in ("numpy", "scipy", "cv2"):
            try:
                module = __import__(module_name)
                versions[module_name] = getattr(module, "__version__", "unknown")
            except Exception:
                versions[module_name] = None
        return {
            "created_at": _now(),
            "python": sys.version,
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "working_directory": str(self.repository_root),
            "git_commit": commit or None,
            "git_branch": branch or None,
            "scientific_packages": versions,
            "colmap": subprocess.run(
                ["bash", "-lc", "command -v colmap || true"],
                capture_output=True,
                text=True,
                check=False,
            ).stdout.strip() or None,
        }

    def _save_state(self) -> None:
        if self.run_directory is None:
            return
        self.manifest["updated_at"] = _now()
        _write_json(self.run_directory / "run_manifest.json", self.manifest)
        _write_json(self.run_directory / "timings.json", self.timings)

    def _stage_record(self, stage_id: str) -> dict[str, Any]:
        for stage in self.manifest["stages"]:
            if stage["id"] == stage_id:
                return stage
        # Additive stages remain resumable for manifests created by older
        # rigcal versions.
        stage = {
            "id": stage_id,
            "display_name": stage_id.replace("_", " ").title(),
            "status": "pending",
        }
        self.manifest["stages"].append(stage)
        self._save_state()
        return stage

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
            str(
                self.repository_root
                / "src/camera_rig_calibration/observation_detection.py"
            ),
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

    def run(
        self,
        config: RigConfig | None = None,
        *,
        resume_directory: Path | None = None,
    ) -> Path:
        if resume_directory is not None:
            config = self._load_run(resume_directory)
        elif config is not None:
            self.validate_ready(config)
            self._new_run(config)
        else:
            raise ValueError("Provide a configuration or a run directory to resume")
        assert config is not None and self.run_directory is not None
        run = self.run_directory

        pointer_path = run / "00_INPUT" / "dataset_pointer.json"
        preparation = None

        def prepare() -> None:
            nonlocal config, preparation
            if config.dataset.prepared_root is None:
                project = config.project.model_copy(
                    update={
                        "dataset_cache_root": self._input_working_root(config)
                    }
                )
                config = RigConfig.model_validate(
                    config.model_copy(
                        update={"project": project}, deep=True
                    ).model_dump(mode="python")
                )
                save_config(config, run / "resolved_config.yaml")
                self.manifest["dataset_working_root"] = str(
                    project.dataset_cache_root.resolve()
                )
                self._save_state()
            preparation = build_preparation_plan(config, self.repository_root)
            for command in preparation.commands:
                self._run_command(command)
            manifest = finalize_dataset(config, preparation)
            moving_info = (
                preparation.dataset_root
                / "raw_images"
                / "camera_info"
                / f"{config.moving_camera.id}.json"
            )
            if moving_info.is_file():
                try:
                    moving_payload = json.loads(
                        moving_info.read_text(encoding="utf-8")
                    )
                except (OSError, json.JSONDecodeError):
                    moving_payload = {}
                profile_key = (
                    moving_payload.get("rigcal_intrinsics_profile")
                    or config.moving_camera.intrinsics_profile
                )
                config = config.model_copy(
                    update={
                        "moving_camera": config.moving_camera.model_copy(
                            update={
                                "intrinsics": moving_info.resolve(),
                                "intrinsics_profile": profile_key,
                                "intrinsic_calibration_video": None,
                                "intrinsic_calibration_images": None,
                            },
                            deep=True,
                        )
                    },
                    deep=True,
                )
                save_config(config, run / "resolved_config.yaml")
                self.manifest["resolved_config_sha256"] = (
                    config_fingerprint(config)
                )
                self.manifest["moving_intrinsics"] = {
                    "profile": profile_key,
                    "path": str(moving_info.resolve()),
                    "sha256": hashlib.sha256(
                        moving_info.read_bytes()
                    ).hexdigest(),
                    "width": moving_payload.get(
                        "width", moving_payload.get("image_width")
                    ),
                    "height": moving_payload.get(
                        "height", moving_payload.get("image_height")
                    ),
                    "distortion_model": moving_payload.get(
                        "distortion_model"
                    ),
                }
                self._save_state()
            save_dataset_manifest(manifest, run / "00_INPUT" / "dataset_manifest.json")
            _write_json(
                pointer_path,
                {
                    "dataset_root": str(preparation.dataset_root.resolve()),
                    "prepared_input": preparation.prepared_input,
                },
            )

        self._execute_stage("prepare_inputs", prepare)
        if not pointer_path.is_file():
            raise RuntimeError(f"Completed input stage has no dataset pointer: {pointer_path}")
        dataset_root = Path(
            json.loads(pointer_path.read_text(encoding="utf-8"))["dataset_root"]
        )
        dataset_manifest = load_dataset_manifest(
            run / "00_INPUT" / "dataset_manifest.json"
        )
        input_id = self._publish_input_view(
            config, dataset_root, dataset_manifest
        )
        authoritative_dataset_root = self._working_paths(
            config
        ).datasets.resolve()
        _write_json(
            pointer_path,
            {
                "dataset_root": str(authoritative_dataset_root),
                "prepared_source_root": str(dataset_root.resolve()),
                "prepared_input": bool(
                    preparation and preparation.prepared_input
                ),
                "layout_version": 2,
                "input_id": input_id,
            },
        )
        observations_root = self._bind_observations_view(config, input_id)

        validation_holder: dict[str, Any] = {}

        def validate() -> None:
            validation = validate_dataset(config, dataset_root)
            validation_holder["value"] = validation
            _write_json(
                run / "00_INPUT" / "validation.json",
                {
                    "valid": validation.valid,
                    "errors": validation.errors,
                    "warnings": validation.warnings,
                    "static_camera_count": validation.static_camera_count,
                    "moving_frame_count": validation.moving_frame_count,
                },
            )
            validation.require_valid()

        self._execute_stage("validate_dataset", validate)

        def detect_markers() -> None:
            if self._observation_contract_ready(
                observations_root,
                self._observation_id(config),
            ):
                self.console.print(
                    f"[dim]Reusing compatible observations: {observations_root}[/dim]"
                )
                return
            self._run_command(self._detector_command(config, dataset_root))

        self._execute_stage("detect_markers", detect_markers)

        filtered_holder: dict[str, Path] = {}

        def apply_observation_quality() -> None:
            try:
                result = filter_observations(
                    observations_root
                    / "shared_all_aruco_observations.csv",
                    run / "preflight",
                    job_id=run.name,
                    marker_settings=config.markers,
                    quality=config.observation_quality,
                )
            except ObservationQualityError as exc:
                _write_json(
                    run / "preflight" / "preflight_summary.json",
                    {
                        "schema_version": 5,
                        "status": "FAILED_PREFLIGHT",
                        "job_id": run.name,
                        "reason": str(exc),
                    },
                )
                raise RuntimeError(f"Observation preflight failed: {exc}") from exc
            if result.accepted_count == 0:
                raise RuntimeError(
                    "Observation preflight rejected every observation; adjust "
                    "this job's quality limits before running a method"
                )
            filtered_holder["root"] = result.filtered_observations_root
            _write_json(
                run / "preflight" / "preflight_summary.json",
                {
                    "schema_version": 5,
                    "status": "READY",
                    "job_id": run.name,
                    "filter": result.summary,
                    "filtered_observations_root": str(
                        result.filtered_observations_root
                    ),
                },
            )

        self._execute_stage("observation_quality", apply_observation_quality)
        quality_observations_root = filtered_holder.get(
            "root", run / "preflight" / "observations"
        )

        observations_root = quality_observations_root

        resolved_holder: dict[str, ResolvedSelections] = {}

        def analyze_selections() -> None:
            resolved_holder["value"] = resolve_selections(
                config, observations_root
            )

        self._execute_stage("analyze_selections", analyze_selections)
        resolved = resolved_holder.get("value") or resolve_selections(
            config, observations_root
        )
        selection_payload = resolved.payload
        manifest_path = run / "00_INPUT" / "dataset_manifest.json"
        dataset_manifest = load_dataset_manifest(manifest_path)
        dataset_manifest.automatic_selections = [
            AutoSelection(
                kind="ap01_root_camera",
                selected=resolved.root_camera,
                candidates=[
                    candidate["id"]
                    for candidate in selection_payload["ap01_root_camera"][
                        "candidates"
                    ]
                ],
                reason=str(
                    selection_payload["ap01_root_camera"]["reason"]
                ),
            ),
            AutoSelection(
                kind="ap02_reference_marker",
                selected=resolved.ap02_reference_marker_id,
                candidates=[
                    candidate["id"]
                    for candidate in selection_payload[
                        "ap02_reference_marker"
                    ]["candidates"]
                ],
                reason=str(
                    selection_payload["ap02_reference_marker"]["reason"]
                ),
            ),
            AutoSelection(
                kind="ap03_single_scale_marker",
                selected=resolved.ap03_single_scale_marker_id,
                candidates=[
                    candidate["id"]
                    for candidate in selection_payload[
                        "ap03_single_scale_marker"
                    ]["candidates"]
                ],
                reason=str(
                    selection_payload["ap03_single_scale_marker"]["reason"]
                ),
            ),
        ]
        save_dataset_manifest(dataset_manifest, manifest_path)
        self._finalize_dataset_observations(
            config,
            quality_observations_root=observations_root,
        )

        if config.project.execution_mode == "prepare_only":
            def finalize_preparation() -> None:
                manifest = load_dataset_manifest(
                    run / "00_INPUT" / "dataset_manifest.json"
                )
                final = run / "99_FINAL_RESULTS"
                lines = [
                    "CAMERA RIG INPUT PREPARATION",
                    "=" * 72,
                    "",
                    f"Run: {run.name}",
                    f"Dataset: {manifest.dataset_id}",
                    f"Static cameras: {len(manifest.static_cameras)}",
                    f"Moving frames: {manifest.moving_camera.image_count}",
                    f"Observations: {observations_root}",
                    "Calibration methods executed: none",
                    "",
                    "Repeat this saved setup with execution_mode=complete to run methods.",
                    "",
                ]
                (final / "SUMMARY.txt").write_text(
                    "\n".join(lines), encoding="utf-8"
                )
                _write_json(
                    final / "SUMMARY.json",
                    {
                        "run": run.name,
                        "dataset": manifest.dataset_id,
                        "execution_mode": "prepare_only",
                        "static_camera_count": len(manifest.static_cameras),
                        "moving_frame_count": manifest.moving_camera.image_count,
                        "methods_executed": [],
                        "selection_candidates": str(
                            observations_root / "SELECTION_CANDIDATES.json"
                        ),
                    },
                )
                self.manifest["status"] = "completed"
                self.manifest["runner_pid"] = None
                self.manifest["completed_at"] = _now()
                self.manifest.pop("error", None)

            self._execute_stage("finalize", finalize_preparation)
            self._save_state()
            published = self._publish_preparation(config, input_id)
            if self.transaction_root is None:
                self.console.print(
                    "\n[bold green]Input preparation completed:[/bold green] "
                    f"{published}"
                )
            else:
                self.console.print(
                    "\n[green]Input preparation validated; atomic dataset "
                    "publication is being finalized.[/green]"
                )
            return published

        if config.selection.mode == "review_once":
            if self.selection_reviewer is None:
                self.manifest["status"] = "waiting_for_selection"
                self.manifest["runner_pid"] = None
                self.manifest["selection_candidates"] = str(
                    observations_root / "SELECTION_CANDIDATES.json"
                )
                self._save_state()
                self.console.print(
                    "\n[yellow]Input and observations are ready. This run is waiting "
                    "for the one-time root/marker review.[/yellow]"
                )
                self.console.print(f"Resume: rigcal --resume {run}")
                return run
            overrides = self.selection_reviewer(config, resolved, run)
            config = freeze_selections(config, resolved, overrides)
        else:
            config = freeze_selections(config, resolved)
        resolved = resolve_selections(config, observations_root)
        config = self._resolve_colmap_environment(config)

        self.manifest["resolution_update_pending"] = True
        self._save_state()
        save_config(config, run / "resolved_config.yaml")
        self.manifest["config_sha256"] = config_fingerprint(config)
        self.manifest["resolved_config_sha256"] = config_fingerprint(config)
        self.manifest["resolved_selections"] = {
            "ap01_root_camera": resolved.root_camera,
            "ap02_reference_marker_id": resolved.ap02_reference_marker_id,
            "ap03_single_scale_marker_id": resolved.ap03_single_scale_marker_id,
            "ap03_multi_marker_ids": list(
                resolved.ap03_multi_marker_ids
            ),
            "evaluation_anchor_marker_id": None,
        }
        self.manifest.pop("resolution_update_pending", None)
        self._save_state()

        method_id = config.methods.enabled[0]
        method_sha = method_fingerprint(config, method_id, resolved)
        target, _, variant = self._execution_target(
            config, input_id, resolved
        )
        self.manifest["method_id"] = method_id
        self.manifest["method_fingerprint"] = method_sha
        self.manifest["variant"] = variant
        self.manifest["variant_target"] = str(target)
        _write_json(
            run / "method_config_diff.json",
            method_config_diff(config, method_id, resolved),
        )
        if self._matching_completed_execution(
            target, method_sha=method_sha, input_id=input_id
        ):
            if config.project.duplicate_policy == "skip":
                self.manifest["status"] = "duplicate_skipped"
                self.manifest["runner_pid"] = None
                self.manifest["duplicate_of"] = str(target)
                self._save_state()
                self.console.print(
                    f"[yellow]Identical method configuration and input already "
                    f"exist in results; skipped without recomputation: "
                    f"{target}[/yellow]"
                )
                return target
            if config.project.duplicate_policy == "error":
                raise RuntimeError(
                    f"Exact completed result already exists: {target}"
                )
        elif target.exists():
            raise RuntimeError(
                "Variant target exists but does not match this method/input "
                f"fingerprint: {target}"
            )

        context = RunContext(
            repository_root=self.repository_root,
            config=config,
            dataset_root=dataset_root,
            observations_root=observations_root,
            run_directory=run,
            resolved_root_camera=resolved.root_camera,
            resolved_ap02_reference_marker_id=(
                resolved.ap02_reference_marker_id
            ),
            resolved_ap03_single_scale_marker_id=(
                resolved.ap03_single_scale_marker_id
            ),
            resolved_ap03_multi_marker_ids=(
                resolved.ap03_multi_marker_ids
            ),
            resolved_marker_ids=resolved.marker_ids,
            reuse_colmap_artifact=self._seed_colmap_artifact(
                config, input_id, method_id
            ),
        )
        method_results: dict[str, dict[str, Any]] = {}
        for method_id in config.methods.enabled:
            method = calibration_methods.get(method_id)
            requirement = method.requirements(context)
            if not requirement.compatible:
                raise RuntimeError(
                    f"{method.display_name} is incompatible with this dataset: "
                    + "; ".join(requirement.reasons)
                )

            def run_method(method=method) -> None:
                commands = tuple(method.commands(context))
                validate_stage_dag(
                    StageContract(
                        command.stage_id,
                        command.depends_on,
                        command.diagnostic,
                    )
                    for command in commands
                )
                for command in commands:
                    try:
                        self._run_command(command)
                    except RuntimeError as exc:
                        if not command.diagnostic:
                            raise
                        failure = {
                            "stage_id": command.stage_id,
                            "display_name": command.display_name,
                            "error": str(exc),
                        }
                        self.manifest.setdefault(
                            "diagnostic_stage_failures", []
                        ).append(failure)
                        self._save_state()
                        self.console.print(
                            f"[yellow]Diagnostic stage failed and the "
                            f"independent primary branch continues: "
                            f"{command.display_name}: {exc}[/yellow]"
                        )

            self._execute_stage(f"method_{method_id}", run_method)
            self._store_colmap_artifact(
                config, input_id, method_id
            )
            method_results[method_id] = method.collect(context)

        if config.evaluation.enabled and not self.defer_evaluation:
            anchor_holder: dict[str, int] = {}

            def resolve_evaluation_anchor() -> None:
                configured = config.evaluation.anchor_marker_id
                if configured != "auto_common":
                    anchor = int(configured)
                    reason = "explicit user configuration"
                else:
                    eligible = set(
                        int(value)
                        for value in resolved.payload["evaluation_anchor"][
                            "observation_candidates"
                        ]
                    )
                    candidates = {
                        int(item["id"]): item
                        for item in resolved.payload[
                            "ap03_single_scale_marker"
                        ]["candidates"]
                        if int(item["id"]) in eligible
                    }
                    if not candidates:
                        raise RuntimeError(
                            "No evaluation marker is shared by the AP01 root "
                            "camera and moving-camera observations"
                        )
                    best_rank = max(
                        ap03_candidate_rank(item)
                        for item in candidates.values()
                    )
                    anchor = min(
                        marker
                        for marker, item in candidates.items()
                        if ap03_candidate_rank(item) == best_rank
                    )
                    reason = (
                        "best observation-supported candidate; every method is "
                        "evaluated with this same marker and failures stay visible"
                    )
                anchor_holder["value"] = anchor
                _write_json(
                    observations_root / "EVALUATION_ANCHOR_SELECTION.json",
                    {
                        "schema_version": 5,
                        "configured": configured,
                        "selected": anchor,
                        "resolved_after_methods": True,
                        "reason": reason,
                    },
                )

            self._execute_stage(
                "resolve_evaluation_anchor", resolve_evaluation_anchor
            )
            if "value" in anchor_holder:
                evaluation_anchor = anchor_holder["value"]
            else:
                evaluation_anchor = int(
                    json.loads(
                        (
                            observations_root
                            / "EVALUATION_ANCHOR_SELECTION.json"
                        ).read_text(encoding="utf-8")
                    )["selected"]
                )
            evaluation_settings = config.evaluation.model_copy(
                update={"anchor_marker_id": evaluation_anchor}
            )
            config = RigConfig.model_validate(
                config.model_copy(
                    update={"evaluation": evaluation_settings}, deep=True
                ).model_dump(mode="python")
            )
            save_config(config, run / "resolved_config.yaml")
            self.manifest["config_sha256"] = config_fingerprint(config)
            self.manifest["resolved_config_sha256"] = config_fingerprint(config)
            self.manifest["resolved_selections"][
                "evaluation_anchor_marker_id"
            ] = evaluation_anchor
            self.manifest["evaluation_fingerprint"] = evaluation_fingerprint(
                config, evaluation_anchor
            )
            self._save_state()
            context = RunContext(
                repository_root=self.repository_root,
                config=config,
                dataset_root=dataset_root,
                observations_root=observations_root,
                run_directory=run,
                resolved_root_camera=resolved.root_camera,
                resolved_ap02_reference_marker_id=(
                    resolved.ap02_reference_marker_id
                ),
                resolved_ap03_single_scale_marker_id=(
                    resolved.ap03_single_scale_marker_id
                ),
                resolved_ap03_multi_marker_ids=(
                    resolved.ap03_multi_marker_ids
                ),
                resolved_evaluation_anchor_marker_id=evaluation_anchor,
                resolved_marker_ids=resolved.marker_ids,
            )
            evaluator = evaluators.get("marker_consistency")
            requirement = evaluator.requirements(context)
            if not requirement.compatible:
                raise RuntimeError(
                    f"Evaluation is incompatible: {'; '.join(requirement.reasons)}"
                )

            def evaluate() -> None:
                for command in evaluator.commands(context):
                    self._run_command(command)

            self._execute_stage("evaluation", evaluate)

        self._execute_stage(
            "comparison", lambda: write_comparison(run, method_results)
        )

        def finalize() -> None:
            self.manifest["status"] = "completed"
            self.manifest["runner_pid"] = None
            self.manifest["completed_at"] = _now()
            self.manifest.pop("error", None)

        self._execute_stage("finalize", finalize)
        self._save_state()
        published = self._publish_completed_execution(
            config, input_id, resolved
        )
        if config.evaluation.enabled and not self.defer_evaluation:
            anchor = int(config.evaluation.anchor_marker_id)
            evaluation_name = (
                f"anchor_marker_{anchor}_"
                f"{evaluation_fingerprint(config, anchor)[:8]}"
            )
            evaluation_view = (
                self._working_paths(config).evaluations
                / evaluation_name
                / method_id
                / variant
            )
            evaluation_view.parent.mkdir(parents=True, exist_ok=True)
            if not evaluation_view.exists() and not evaluation_view.is_symlink():
                evaluation_view.symlink_to(
                    (published / "06_EVALUATION").resolve(),
                    target_is_directory=True,
                )
        if self.transaction_root is None:
            self.console.print(
                f"\n[bold green]Calibration run completed:[/bold green] "
                f"{published}"
            )
        else:
            self.console.print(
                "\n[green]Method execution completed; canonical publication "
                "is being finalized.[/green]"
            )
        return published

    def mark_interrupted(self) -> None:
        """Persist Ctrl+C as a resumable state instead of leaving `running`."""
        if self.run_directory is None or not self.manifest:
            return
        self.manifest["status"] = "interrupted"
        self.manifest["runner_pid"] = None
        self.manifest["interrupted_at"] = _now()
        for stage in self.manifest.get("stages", []):
            if stage.get("status") == "running":
                stage["status"] = "interrupted"
                stage["finished_at"] = _now()
                stage["error"] = "Interrupted by user"
        self._save_state()


def find_run(output_root: Path, run_id_or_path: str) -> Path:
    candidate = Path(run_id_or_path).expanduser()
    if candidate.is_dir():
        return candidate.resolve()
    root = output_root.resolve()
    matches: list[Path] = []
    for manifest_path in root.rglob("run_manifest.json"):
        directory = manifest_path.parent
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            payload = {}
        if (
            directory.name == run_id_or_path
            or payload.get("run_id") == run_id_or_path
            or payload.get("execution_id") == run_id_or_path
        ):
            matches.append(directory)
    if not matches:
        raise FileNotFoundError(f"Run not found: {run_id_or_path}")
    if len(matches) > 1:
        raise RuntimeError(
            f"Run ID is ambiguous across datasets: {run_id_or_path}; use its path"
        )
    return matches[0].resolve()
