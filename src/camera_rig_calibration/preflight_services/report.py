"""Queue-level preflight status and report finalization."""
from __future__ import annotations

from pathlib import Path

from ..config.models import RigConfig
from .bindings import PreflightDependencies
from .core import (
    CameraObservationCoverage,
    PreflightJobResult,
    QueuePreflightResult,
)


def finalize_queue_preflight(
    results: list[PreflightJobResult],
    *,
    destination: Path,
    raw_observations_csv: Path,
    dataset_root: Path,
    first_config: RigConfig,
    queue_camera_coverage: tuple[CameraObservationCoverage, ...],
    missing_required: tuple[str, ...],
    common_evaluation_anchor: int | None,
    dependencies: PreflightDependencies,
) -> QueuePreflightResult:
    """Persist and return the aggregate queue-preflight result."""
    _write_json = dependencies.write_json
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
    review_reasons: list[str] = []
    if missing_required:
        review_reasons.append("required_camera_without_detection")
    if any(
        result.ap02_graph_diagnosis is not None
        and not result.ap02_graph_diagnosis.complete
        for result in results
    ):
        review_reasons.append("ap02_combined_graph_incomplete")
    if review_reasons:
        queue_status = "REVIEW_REQUIRED"
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
            "detection_mode": first_config.markers.detection_mode,
            "camera_coverage": [
                {
                    "camera_id": item.camera_id,
                    "required": item.required,
                    "raw_detection_count": item.raw_detection_count,
                    "marker_ids": list(item.marker_ids),
                }
                for item in queue_camera_coverage
            ],
            "missing_required_cameras": list(missing_required),
            "review_required": bool(review_reasons),
            "review_reasons": review_reasons,
            "common_evaluation_anchor_marker_id": (
                common_evaluation_anchor
            ),
            "runnable_jobs": [result.job_id for result in runnable],
            "skipped_jobs": [result.job_id for result in failed],
        },
    )
    return QueuePreflightResult(
        status=queue_status,
        jobs=tuple(results),
        output_directory=destination,
        camera_coverage=queue_camera_coverage,
        missing_required_cameras=missing_required,
        review_reasons=tuple(review_reasons),
        common_evaluation_anchor_marker_id=common_evaluation_anchor,
    )
