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
    _read_json,
    _materialize_tree,
)
from .bindings import current_runtime_bindings


class ArtifactMixin:
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
        colmap_artifact_fingerprint = (
            current_runtime_bindings().colmap_artifact_fingerprint
        )
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

    def _validate_conflicting_existing_target(
        self, target: Path, *, input_id: str
    ) -> None:
        """Allow only an explicit same-dataset rerun past a public target."""

        if not self.explicit_method_rerun:
            raise RuntimeError(
                "Variant target exists but does not match this method/input "
                f"fingerprint: {target}"
            )
        existing_result = _read_json(target / "RESULT.json")
        if existing_result.get("input_fingerprint") != input_id:
            raise RuntimeError(
                "Explicit method rerun refused: the existing public result "
                "belongs to a different immutable dataset: "
                f"{target}"
            )

    def _validate_explicit_rerun_dataset_identity(
        self, actual_dataset_identity: dict[str, Any]
    ) -> None:
        if not self.explicit_method_rerun:
            return
        from ..dataset_identity import identities_match

        declared_dataset_identity = self.rerun_metadata.get(
            "dataset_identity"
        )
        if not isinstance(declared_dataset_identity, dict) or not (
            identities_match(
                declared_dataset_identity, actual_dataset_identity
            )
        ):
            raise RuntimeError(
                "Explicit method rerun refused: the prepared input no "
                "longer matches the exact immutable dataset identity."
            )

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



__all__ = ['ArtifactMixin']
