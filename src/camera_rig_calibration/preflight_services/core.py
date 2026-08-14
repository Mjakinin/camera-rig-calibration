from __future__ import annotations

import csv
import json
import shutil
from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable

from ..methods.ap02.graph_diagnostics import (
    AP02GraphDiagnosis,
    diagnose_ap02_graph,
    graph_components,
)
from ..components import register_builtin_components
from ..config.models import RigConfig, effective_observation_quality
from ..contracts import RunContext
from ..observation_quality import (
    ObservationFilterResult,
    ObservationQualityError,
    filter_observations,
)
from ..observations import (
    ResolvedSelections,
    resolve_selections,
    write_selection_candidates_csv,
)
from ..registry import calibration_methods
from ..methods.ap02.frame_selection import (
    AP02FrameSelectionError,
    select_ap02_frames,
    write_ap02_frame_selection,
)


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
    camera_coverage: tuple["CameraObservationCoverage", ...] = ()
    ap02_graph_diagnosis: AP02GraphDiagnosis | None = None

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
    camera_coverage: tuple["CameraObservationCoverage", ...] = ()
    missing_required_cameras: tuple[str, ...] = ()
    review_reasons: tuple[str, ...] = ()
    common_evaluation_anchor_marker_id: int | None = None

    @property
    def ready(self) -> bool:
        return any(job.runnable for job in self.jobs)

    @property
    def review_required(self) -> bool:
        return bool(self.review_reasons)


@dataclass(frozen=True)
class CameraObservationCoverage:
    camera_id: str
    required: bool
    raw_detection_count: int
    accepted_observation_count: int
    marker_ids: tuple[int, ...]


def _observation_camera_id(row: dict[str, str]) -> str:
    return str(
        row.get("camera_name")
        or row.get("observer_id")
        or ""
    ).strip()


def _read_observation_rows(path: Path) -> list[dict[str, str]]:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))
    except OSError as exc:
        raise ObservationQualityError(
            f"Could not read observation evidence: {path}"
        ) from exc


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
        "marker_inventory.csv",
        "marker_inventory.json",
    ):
        path = source / name
        if path.is_file():
            shutil.copy2(path, destination / name)


def _write_ap02_graph_diagnosis(
    job_root: Path,
    diagnosis: AP02GraphDiagnosis,
) -> None:
    _write_json(
        job_root / "AP02_COMBINED_GRAPH.json",
        diagnosis.model_dump(),
    )
    component_lines = [
        (
            f"{component.component_id}: cameras="
            f"{','.join(component.static_cameras) or '-'}; "
            f"markers={','.join(map(str, component.marker_ids))}; "
            f"moving_frames={len(component.moving_frames)}; "
            "connecting_moving_frames="
            f"{len(component.connecting_moving_frames)}; "
            f"calibratable={'yes' if component.calibratable else 'no'}"
        )
        for component in diagnosis.components
    ]
    (job_root / "AP02_COMBINED_GRAPH.txt").write_text(
        "\n".join(
            [
                "AP02 COMBINED GRAPH DIAGNOSIS",
                "=" * 72,
                "",
                f"Reference marker: {diagnosis.reference_marker_id}",
                (
                    "Primary coverage: "
                    f"{len(diagnosis.reached_static_cameras)}/"
                    f"{len(diagnosis.expected_static_cameras)} static cameras"
                ),
                f"Connected components: {len(diagnosis.components)}",
                (
                    "Missing from primary component: "
                    + (
                        ", ".join(diagnosis.missing_static_cameras)
                        if diagnosis.missing_static_cameras
                        else "none"
                    )
                ),
                "Cause: " + ", ".join(diagnosis.cause_codes),
                diagnosis.explanation,
                "",
                *component_lines,
                "",
            ]
        ),
        encoding="utf-8",
    )


def build_queue_camera_coverage(
    config: RigConfig,
    raw_rows: list[dict[str, str]],
) -> tuple[tuple[CameraObservationCoverage, ...], tuple[str, ...]]:
    """Summarize raw evidence once for every method-specific preflight."""

    required_by_camera = {
        camera.id: bool(camera.required) for camera in config.static_cameras
    }
    raw_counts: dict[str, int] = defaultdict(int)
    raw_marker_ids: dict[str, set[int]] = defaultdict(set)
    for row in raw_rows:
        camera_id = _observation_camera_id(row)
        if not camera_id:
            continue
        raw_counts[camera_id] += 1
        try:
            raw_marker_ids[camera_id].add(int(float(row["marker_id"])))
        except (KeyError, TypeError, ValueError):
            continue
    coverage = tuple(
        CameraObservationCoverage(
            camera_id=camera_id,
            required=required,
            raw_detection_count=raw_counts.get(camera_id, 0),
            accepted_observation_count=0,
            marker_ids=tuple(sorted(raw_marker_ids.get(camera_id, set()))),
        )
        for camera_id, required in required_by_camera.items()
    )
    missing_required = tuple(
        item.camera_id
        for item in coverage
        if item.required and item.raw_detection_count == 0
    )
    return coverage, missing_required
