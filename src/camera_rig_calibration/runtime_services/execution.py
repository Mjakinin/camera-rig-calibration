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
)
from .bindings import current_runtime_bindings


class ExecutionMixin:
    def run(
        self,
        config: RigConfig | None = None,
        *,
        resume_directory: Path | None = None,
    ) -> Path:
        bindings = current_runtime_bindings()
        freeze_selections = bindings.freeze_selections
        resolve_selections = bindings.resolve_selections
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
        from ..dataset_identity import build_dataset_identity

        actual_dataset_identity = build_dataset_identity(
            authoritative_dataset_root
        )
        self.manifest["dataset_identity"] = actual_dataset_identity
        self._validate_explicit_rerun_dataset_identity(
            actual_dataset_identity
        )
        self.manifest["algorithm_version"] = {
            "ap01": "ap01_main_compat_hierarchical_v1",
            "ap02": "ap02_main_compat_widest_path_v1",
            "ap03": "ap03_shared_colmap_single_multi_v1",
        }.get(next(iter(config.methods.enabled), ""), "extension_v1")
        self._save_state()
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
            expected_observation_id = str(
                self.manifest.get("observation_id")
                or self._observation_id(config)
            )
            if self._observation_contract_ready(
                observations_root,
                expected_observation_id,
            ):
                self.console.print(
                    "[dim]Reusing frozen compatible observations: "
                    f"{observations_root}[/dim]"
                )
                return
            self._run_command(self._detector_command(config, dataset_root))

        self._execute_stage("detect_markers", detect_markers)

        filtered_holder: dict[str, Path] = {}

        def apply_observation_quality() -> None:
            try:
                method_id = config.methods.enabled[0]
                effective_quality, quality_sources = (
                    effective_observation_quality(config, method_id)
                )
                result = filter_observations(
                    observations_root
                    / "shared_all_aruco_observations.csv",
                    run / "preflight",
                    job_id=run.name,
                    marker_settings=config.markers,
                    quality=effective_quality,
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
                    "effective_observation_quality": (
                        effective_quality.model_dump(mode="json")
                    ),
                    "observation_quality_sources": quality_sources,
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

        if (
            config.selection.mode == "review_once"
            or (
                config.evaluation.enabled
                and config.evaluation.anchor_selection_mode == "review_once"
            )
        ):
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
        selection_payload = resolved.payload
        config = self._resolve_colmap_environment(config)

        self.manifest["resolution_update_pending"] = True
        self._save_state()
        save_config(config, run / "resolved_config.yaml")
        self.manifest["config_sha256"] = config_fingerprint(config)
        self.manifest["resolved_config_sha256"] = config_fingerprint(config)
        self.manifest["resolved_selections"] = {
            "ap01_root_camera": resolved.root_camera,
            "ap02_reference_marker_id": resolved.ap02_reference_marker_id,
            "ap02_reference_marker_selection_mode": (
                config.methods.ap02.reference_marker_selection_mode
            ),
            "ap02_reference_marker_reason": selection_payload[
                "ap02_reference_marker"
            ].get("reason"),
            "ap02_reference_marker_evidence": selection_payload[
                "ap02_reference_marker"
            ].get("evidence"),
            "ap03_single_scale_marker_id": resolved.ap03_single_scale_marker_id,
            "ap03_multi_marker_ids": list(
                resolved.ap03_multi_marker_ids
            ),
            "evaluation_anchor_marker_id": (
                resolved.evaluation_anchor_marker_id
            ),
        }
        effective_quality, quality_sources = (
            effective_observation_quality(
                config, config.methods.enabled[0]
            )
        )
        self.manifest["effective_observation_quality"] = (
            effective_quality.model_dump(mode="json")
        )
        self.manifest["observation_quality_sources"] = quality_sources
        self.manifest["automatic_recommendations"] = (
            selection_payload["automatic_recommendations"]
        )
        self.manifest["final_decisions"] = dict(
            self.manifest["resolved_selections"]
        )
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
            self._validate_conflicting_existing_target(
                target, input_id=input_id
            )
            self.manifest["superseding_public_target_after_validation"] = str(
                target
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
            resolved_marker_ids=resolved.marker_ids,
            reuse_colmap_artifact=self._seed_colmap_artifact(
                config, input_id, method_id
            ),
            reused_method_stages=self.reused_method_stages,
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
            if resolved.evaluation_anchor_marker_id is None:
                raise RuntimeError(
                    "Evaluation is enabled, but no evaluation anchor was "
                    "frozen during preflight."
                )
            evaluation_anchor = int(
                resolved.evaluation_anchor_marker_id
            )
            _write_json(
                observations_root / "EVALUATION_ANCHOR_SELECTION.json",
                {
                    "schema_version": 5,
                    "configured": config.evaluation.anchor_marker_id,
                    "selected": evaluation_anchor,
                    "resolved_after_methods": False,
                    "resolution_stage": "preflight",
                    "reason": (
                        resolved.payload["evaluation_anchor"]["reason"]
                    ),
                },
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


__all__ = ['ExecutionMixin']
