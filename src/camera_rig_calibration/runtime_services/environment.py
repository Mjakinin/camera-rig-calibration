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
    _now,
    _write_json,
    _run_id,
    _run_directories,
    _materialize_tree,
    planned_stages,
)
from .bindings import current_runtime_bindings


class EnvironmentMixin:
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
                config.colmap.compute_mode == "gpu"
                and not self._compatible_gpu_available()
            ):
                missing.append(
                    "a compatible NVIDIA GPU/driver "
                    "(COLMAP compute_mode=gpu)"
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
        if requested.compute_mode == "gpu" and not gpu_available:
            raise RuntimeError(
                "COLMAP compute_mode=gpu but the lightweight NVIDIA capability "
                "probe found no compatible GPU"
            )
        resolved_compute_mode = (
            "gpu"
            if requested.compute_mode == "gpu"
            or (requested.compute_mode == "auto" and gpu_available)
            else "cpu_baseline"
        )
        version_probe = subprocess.run(
            [str(executable), "--version"],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
        version_text = version_probe.stdout.strip()
        if version_probe.returncode != 0 or not version_text.startswith("COLMAP"):
            help_probe = subprocess.run(
                [str(executable), "-h"],
                capture_output=True,
                text=True,
                check=False,
                timeout=10,
            )
            version_text = help_probe.stdout.strip() or help_probe.stderr.strip()
        version = (version_text or "unknown").splitlines()[0]
        method_id = next(iter(config.methods.enabled), "")
        ap03_explicit_limits = (
            method_id == "ap03"
            and config.methods.ap03.feature_limit_policy
            == "wizard_explicit_limits_v1"
        )
        requested_label = method_result_label(config, method_id)
        resolved = config.model_copy(
            update={
                "project": config.project.model_copy(
                    update={"run_label": requested_label}
                ),
                "colmap": requested.model_copy(
                    update={
                        "executable": str(executable),
                        "compute_mode": resolved_compute_mode,
                    }
                )
            },
            deep=True,
        )
        self.manifest["colmap_resolution"] = {
            "requested_executable": requested.executable,
            "resolved_executable": str(executable),
            "version": version,
            "configured_compute_mode": requested.compute_mode,
            "resolved_compute_mode": resolved_compute_mode,
            # Retain the former report keys for published-schema consumers.
            "requested_gpu_mode": (
                "true"
                if requested.compute_mode == "gpu"
                else (
                    "auto"
                    if requested.compute_mode == "auto"
                    else "false"
                )
            ),
            "resolved_gpu_mode": (
                "true" if resolved_compute_mode == "gpu" else "false"
            ),
            "gpu_probe_available": gpu_available,
            "matcher": requested.matcher,
            "maximum_image_size": (
                (
                    requested.ap03_maximum_image_size
                    or requested.maximum_image_size
                )
                if ap03_explicit_limits
                else None
                if method_id == "ap03"
                else requested.maximum_image_size
            ),
            "maximum_features": (
                (
                    requested.ap03_maximum_features
                    or requested.maximum_features
                )
                if ap03_explicit_limits
                else None
                if method_id == "ap03"
                else requested.maximum_features
            ),
            "configured_ap03_maximum_image_size": (
                requested.ap03_maximum_image_size
            ),
            "configured_ap03_maximum_features": requested.ap03_maximum_features,
            "ap03_feature_limit_policy": (
                config.methods.ap03.feature_limit_policy
                if method_id == "ap03"
                else None
            ),
            "mapper_minimum_matches": requested.mapper_minimum_matches,
            "intrinsics_refinement": {
                "focal_length": False,
                "principal_point": False,
                "extra_parameters": False,
            },
        }
        self._save_state()
        self.console.print(
            "[dim]COLMAP resolved: "
            f"matcher={requested.matcher}, compute={requested.compute_mode}"
            f" -> {resolved_compute_mode}, "
            f"image_size={self.manifest['colmap_resolution']['maximum_image_size'] or 'COLMAP-default'}, "
            f"features={self.manifest['colmap_resolution']['maximum_features'] or 'COLMAP-default'}, "
            f"mapper_matches={requested.mapper_minimum_matches}[/dim]"
        )
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
            "observation_input_contract": "observation_quality_v2",
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
        if self.rerun_metadata:
            self.manifest.update(self.rerun_metadata)
            self.manifest["single_method_rerun"] = True
        self.manifest["explicit_method_rerun"] = self.explicit_method_rerun
        self.run_directory = run
        if (
            self.reuse_intermediates_from is not None
            and config.methods.enabled == ["ap01"]
        ):
            source = self.reuse_intermediates_from
            destination = run / "02_AP01"
            reused: list[str] = []
            for stage, public_name in (
                ("01_moving_colmap", "moving_colmap"),
                ("02_metric_scale", "metric_scale"),
            ):
                stage_source = (
                    source / stage
                    if (source / stage).is_dir()
                    else source / public_name
                )
                stage_target = destination / stage
                if not stage_source.is_dir():
                    raise RuntimeError(
                        "Requested AP01 intermediate reuse is incomplete: "
                        f"{stage_source}"
                    )
                _materialize_tree(stage_source, stage_target)
                reused.append(stage)
            self.reused_method_stages = tuple(reused)
            self.manifest["reused_stages"] = reused
            self.manifest["reuse_source"] = str(source)
            self.manifest["reuse_validation"] = (
                "method/input/COLMAP fingerprint validated by rerun-method"
            )
            self.manifest["rerun_stages"] = [
                "03_candidates",
                "03_static_extrinsics",
                "05_report",
                "publication",
                "anchor_export",
                "gt_evaluation",
                "reporting",
                "rviz_derivation",
            ]
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



__all__ = ['EnvironmentMixin']
