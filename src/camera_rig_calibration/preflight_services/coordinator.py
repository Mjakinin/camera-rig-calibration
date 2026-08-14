"""Short coordinator for the queue-preflight phases."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

from .bindings import PreflightDependencies
from .common_anchor import resolve_common_evaluation_anchor
from .core import (
    PreflightJob,
    QueuePreflightResult,
    build_queue_camera_coverage,
)
from .job import run_job_preflight
from .marker_inventory import enrich_raw_marker_inventory
from .report import finalize_queue_preflight


def run_queue_preflight(
    jobs: Iterable[PreflightJob],
    *,
    raw_observations_csv: Path,
    dataset_root: Path,
    output_directory: Path,
    repository_root: Path,
) -> QueuePreflightResult:
    """Resolve all inexpensive job readiness checks before any method starts."""

    dependencies = PreflightDependencies.current()
    dependencies.register_builtin_components()
    destination = output_directory.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    job_list = list(jobs)
    if not job_list:
        raise ValueError("Queue preflight requires at least one method job")

    raw_rows = dependencies.read_observation_rows(raw_observations_csv)
    first_config = job_list[0].config
    queue_camera_coverage, missing_required = build_queue_camera_coverage(
        first_config, raw_rows
    )
    results = [
        run_job_preflight(
            job,
            raw_observations_csv=raw_observations_csv,
            raw_rows=raw_rows,
            dataset_root=dataset_root,
            destination=destination,
            repository_root=repository_root,
            queue_camera_coverage=queue_camera_coverage,
            dependencies=dependencies,
        )
        for job in job_list
    ]
    results = enrich_raw_marker_inventory(results, raw_rows, dependencies)
    results, common_evaluation_anchor = resolve_common_evaluation_anchor(
        job_list, results, dependencies
    )
    return finalize_queue_preflight(
        results,
        destination=destination,
        raw_observations_csv=raw_observations_csv,
        dataset_root=dataset_root,
        first_config=first_config,
        queue_camera_coverage=queue_camera_coverage,
        missing_required=missing_required,
        common_evaluation_anchor=common_evaluation_anchor,
        dependencies=dependencies,
    )
