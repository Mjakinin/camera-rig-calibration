from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .components import register_builtin_components
from .config.models import RigConfig
from .contracts import RunContext
from .observation_quality import (
    ObservationFilterResult,
    ObservationQualityError,
    filter_observations,
)
from .observations import ResolvedSelections, resolve_selections
from .registry import calibration_methods


@dataclass(frozen=True)
class PreflightJob:
    job_id: str
    config: RigConfig


@dataclass(frozen=True)
class PreflightJobResult:
    job_id: str
    status: str
    errors: tuple[str, ...]
    warnings: tuple[str, ...]
    details: tuple[str, ...]
    filter_result: ObservationFilterResult | None
    selections: ResolvedSelections | None
    output_directory: Path

    @property
    def runnable(self) -> bool:
        return self.status in {
            "READY",
            "READY_WITH_WARNINGS",
            "READY_PARTIAL",
        }


@dataclass(frozen=True)
class QueuePreflightResult:
    status: str
    jobs: tuple[PreflightJobResult, ...]
    output_directory: Path

    @property
    def ready(self) -> bool:
        return any(job.runnable for job in self.jobs)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _copy_filter_artifacts(source: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for name in (
        "observation_filter_summary.json",
        "accepted_observations.csv",
        "rejected_observations.csv",
    ):
        path = source / name
        if path.is_file():
            shutil.copy2(path, destination / name)


def run_queue_preflight(
    jobs: Iterable[PreflightJob],
    *,
    raw_observations_csv: Path,
    dataset_root: Path,
    output_directory: Path,
    repository_root: Path,
) -> QueuePreflightResult:
    """Resolve all inexpensive job readiness checks before any method starts."""
    register_builtin_components()
    destination = output_directory.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    results: list[PreflightJobResult] = []
    for job in jobs:
        job_root = destination / "jobs" / job.job_id
        errors: list[str] = []
        warnings: list[str] = []
        details: list[str] = []
        filtered = None
        selections = None
        try:
            filtered = filter_observations(
                raw_observations_csv,
                job_root,
                job_id=job.job_id,
                marker_settings=job.config.markers,
                quality=job.config.observation_quality,
            )
            if filtered.accepted_count == 0:
                errors.append("Every raw observation was rejected")
            else:
                details.append(
                    "Observation quality: "
                    f"{filtered.accepted_count} accepted, "
                    f"{filtered.rejected_count} rejected; configured limits "
                    "were not altered"
                )
                selections = resolve_selections(
                    job.config, filtered.filtered_observations_root
                )
                context = RunContext(
                    repository_root=repository_root.resolve(),
                    config=job.config,
                    dataset_root=dataset_root.resolve(),
                    observations_root=filtered.filtered_observations_root,
                    run_directory=job_root,
                    resolved_root_camera=selections.root_camera,
                    resolved_ap02_reference_marker_id=(
                        selections.ap02_reference_marker_id
                    ),
                    resolved_ap03_single_scale_marker_id=(
                        selections.ap03_single_scale_marker_id
                    ),
                    resolved_ap03_multi_marker_ids=(
                        selections.ap03_multi_marker_ids
                    ),
                    resolved_marker_ids=selections.marker_ids,
                )
                method_id = job.config.methods.enabled[0]
                requirement = calibration_methods.get(method_id).requirements(
                    context
                )
                if not requirement.compatible:
                    errors.extend(requirement.reasons)
                if method_id == "ap02":
                    candidate = next(
                        (
                            item
                            for item in selections.payload[
                                "ap02_reference_marker"
                            ]["candidates"]
                            if int(item["id"])
                            == selections.ap02_reference_marker_id
                        ),
                        {},
                    )
                    expected = len(job.config.static_cameras)
                    combined = int(
                        candidate.get(
                            "combined_graph_reachable_static_count", 0
                        )
                    )
                    if combined < expected:
                        if not candidate.get(
                            "ap02_partial_compatible", False
                        ):
                            errors.append(
                                "Combined input graph has no usable AP02 "
                                f"component ({combined}/{expected} static "
                                "cameras)"
                            )
                    combined_missing = sorted(
                        set(camera.id for camera in job.config.static_cameras)
                        - set(
                            candidate.get(
                                "combined_graph_reachable_static_cameras", []
                            )
                        )
                    )
                    static_only = int(
                        candidate.get(
                            "static_graph_reachable_count", 0
                        )
                    )
                    static_missing = sorted(
                        set(camera.id for camera in job.config.static_cameras)
                        - set(
                            candidate.get(
                                "static_graph_reachable_cameras", []
                            )
                        )
                    )
                    details.extend(
                        [
                            f"Static-only coverage: {static_only}/{expected} cameras",
                            (
                                "Missing from diagnostic branch: "
                                + ", ".join(static_missing)
                                if static_missing
                                else "Missing from diagnostic branch: none"
                            ),
                            f"Combined coverage: {combined}/{expected} cameras",
                            (
                                "Missing from primary branch: "
                                + ", ".join(combined_missing)
                                if combined_missing
                                else "Missing from primary branch: none"
                            ),
                            (
                                "Coverage checks input connectivity only; they "
                                "do not predict Bundle Adjustment success"
                            ),
                        ]
                    )
                if method_id == "ap03":
                    candidates = {
                        int(item["id"]): item
                        for item in selections.payload[
                            "ap03_single_scale_marker"
                        ]["candidates"]
                    }
                    single = candidates.get(
                        selections.ap03_single_scale_marker_id, {}
                    )
                    if not single.get("ap03_compatible", False):
                        warnings.append(
                            "AP03 single-scale diagnostic is unsupported; "
                            "multi-scale remains the primary result"
                        )
        except (ObservationQualityError, RuntimeError, ValueError) as exc:
            errors.append(str(exc))

        partial = False
        if (
            not errors
            and selections is not None
            and job.config.methods.enabled[0] == "ap02"
        ):
            selected = next(
                (
                    item
                    for item in selections.payload[
                        "ap02_reference_marker"
                    ]["candidates"]
                    if int(item["id"])
                    == selections.ap02_reference_marker_id
                ),
                {},
            )
            partial = (
                int(
                    selected.get(
                        "combined_graph_reachable_static_count", 0
                    )
                )
                < len(job.config.static_cameras)
            )
            if partial:
                details.append(
                    "AP02 will run only the connected component as a "
                    "diagnostic partial result; this is a connectivity limit, "
                    "not an automatic reprojection-threshold fallback"
                )
        status = (
            "FAILED_PREFLIGHT"
            if errors
            else "READY_PARTIAL"
            if partial
            else "READY_WITH_WARNINGS"
            if warnings
            else "READY"
        )
        if job.config.methods.enabled[0] == "ap02":
            details.append(f"Status: {status}")
        summary = {
            "schema_version": 5,
            "job_id": job.job_id,
            "method_id": job.config.methods.enabled[0],
            "status": status,
            "errors": errors,
            "warnings": warnings,
            "details": details,
            "observation_quality": job.config.observation_quality.model_dump(
                mode="json"
            ),
            "resolved_selections": (
                {
                    "root_camera": selections.root_camera,
                    "ap02_reference_marker_id": (
                        selections.ap02_reference_marker_id
                    ),
                    "ap03_single_scale_marker_id": (
                        selections.ap03_single_scale_marker_id
                    ),
                    "ap03_multi_marker_ids": list(
                        selections.ap03_multi_marker_ids
                    ),
                }
                if selections is not None
                else None
            ),
        }
        _write_json(job_root / "preflight_summary.json", summary)
        results.append(
            PreflightJobResult(
                job_id=job.job_id,
                status=status,
                errors=tuple(errors),
                warnings=tuple(warnings),
                details=tuple(details),
                filter_result=filtered,
                selections=selections,
                output_directory=job_root,
            )
        )

    runnable = [result for result in results if result.runnable]
    failed = [result for result in results if not result.runnable]
    queue_status = (
        "FAILED_PREFLIGHT"
        if not runnable
        else "READY_PARTIAL"
        if failed or any(result.status == "READY_PARTIAL" for result in runnable)
        else "READY_WITH_WARNINGS"
        if any(result.status == "READY_WITH_WARNINGS" for result in runnable)
        else "READY"
    )
    _write_json(
        destination / "queue_preflight_summary.json",
        {
            "schema_version": 5,
            "status": queue_status,
            "raw_observations": str(raw_observations_csv.resolve()),
            "dataset_root": str(dataset_root.resolve()),
            "jobs": [
                {
                    "job_id": result.job_id,
                    "status": result.status,
                    "errors": list(result.errors),
                    "warnings": list(result.warnings),
                    "details": list(result.details),
                    "preflight_summary": str(
                        result.output_directory / "preflight_summary.json"
                    ),
                }
                for result in results
            ],
            "methods_may_start": bool(runnable),
            "runnable_jobs": [result.job_id for result in runnable],
            "skipped_jobs": [result.job_id for result in failed],
        },
    )
    return QueuePreflightResult(
        status=queue_status,
        jobs=tuple(results),
        output_directory=destination,
    )
