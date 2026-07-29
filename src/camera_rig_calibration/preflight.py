from __future__ import annotations

import csv
import json
import shutil
from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable

from .ap02_graph import (
    AP02GraphDiagnosis,
    diagnose_ap02_graph,
    graph_components,
)
from .components import register_builtin_components
from .config.models import RigConfig, effective_observation_quality
from .contracts import RunContext
from .observation_quality import (
    ObservationFilterResult,
    ObservationQualityError,
    filter_observations,
)
from .observations import (
    ResolvedSelections,
    resolve_selections,
    write_selection_candidates_csv,
)
from .registry import calibration_methods
from .methods.ap02.frame_selection import (
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
    job_list = list(jobs)
    if not job_list:
        raise ValueError("Queue preflight requires at least one method job")
    raw_rows = _read_observation_rows(raw_observations_csv)
    first_config = job_list[0].config
    required_by_camera = {
        camera.id: bool(camera.required)
        for camera in first_config.static_cameras
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
    queue_camera_coverage = tuple(
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
        for item in queue_camera_coverage
        if item.required and item.raw_detection_count == 0
    )
    for job in job_list:
        job_root = destination / "jobs" / job.job_id
        errors: list[str] = []
        warnings: list[str] = []
        details: list[str] = []
        filtered = None
        selections = None
        ap02_graph_diagnosis = None
        ap02_frame_selection_summary = None
        effective_quality = None
        quality_sources = None
        job_camera_coverage = tuple(
            CameraObservationCoverage(
                camera_id=item.camera_id,
                required=item.required,
                raw_detection_count=item.raw_detection_count,
                accepted_observation_count=0,
                marker_ids=item.marker_ids,
            )
            for item in queue_camera_coverage
        )
        try:
            method_id = job.config.methods.enabled[0]
            effective_quality, quality_sources = (
                effective_observation_quality(job.config, method_id)
            )
            filtered = filter_observations(
                raw_observations_csv,
                job_root,
                job_id=job.job_id,
                marker_settings=job.config.markers,
                quality=effective_quality,
            )
            if filtered.accepted_count == 0:
                errors.append("Every raw observation was rejected")
            else:
                accepted_rows = _read_observation_rows(
                    filtered.accepted_path
                )
                accepted_counts: dict[str, int] = defaultdict(int)
                for row in accepted_rows:
                    camera_id = _observation_camera_id(row)
                    if camera_id:
                        accepted_counts[camera_id] += 1
                if method_id == "ap02":
                    expected_camera_ids = tuple(
                        camera.id for camera in job.config.static_cameras
                    )
                    preliminary_components = graph_components(
                        accepted_rows, expected_camera_ids
                    )
                    configured_reference = (
                        job.config.methods.ap02.reference_marker_id
                    )
                    if isinstance(configured_reference, int):
                        preliminary_reference = configured_reference
                    else:
                        preferred_component = next(
                            (
                                component
                                for component in preliminary_components
                                if component.calibratable
                            ),
                            (
                                preliminary_components[0]
                                if preliminary_components
                                else None
                            ),
                        )
                        preliminary_reference = (
                            preferred_component.anchor_marker_id
                            if preferred_component is not None
                            else -1
                        )
                    ap02_graph_diagnosis = diagnose_ap02_graph(
                        raw_rows=raw_rows,
                        accepted_rows=accepted_rows,
                        rejected_rows=_read_observation_rows(
                            filtered.rejected_path
                        ),
                        static_camera_ids=expected_camera_ids,
                        reference_marker_id=preliminary_reference,
                    )
                    _write_ap02_graph_diagnosis(
                        job_root, ap02_graph_diagnosis
                    )
                job_camera_coverage = tuple(
                    CameraObservationCoverage(
                        camera_id=item.camera_id,
                        required=item.required,
                        raw_detection_count=item.raw_detection_count,
                        accepted_observation_count=accepted_counts.get(
                            item.camera_id, 0
                        ),
                        marker_ids=item.marker_ids,
                    )
                    for item in queue_camera_coverage
                )
                details.append(
                    "Observation quality: "
                    f"{filtered.accepted_count} accepted, "
                    f"{filtered.rejected_count} rejected; configured limits "
                    "were not altered"
                )
                details.append(
                    "Effective quality sources: "
                    + ", ".join(
                        f"{name}={source}"
                        for name, source in sorted(quality_sources.items())
                    )
                )
                selections = resolve_selections(
                    job.config, filtered.filtered_observations_root
                )
                if method_id == "ap02":
                    settings = job.config.methods.ap02
                    frame_selection = select_ap02_frames(
                        accepted_rows,
                        camera_ids=tuple(
                            camera.id
                            for camera in job.config.static_cameras
                        ),
                        reference_marker_id=(
                            selections.ap02_reference_marker_id
                        ),
                        reference_marker_maximum_frames=(
                            settings.reference_marker_maximum_frames
                        ),
                        top_per_marker=settings.top_per_marker,
                        top_per_marker_pair=settings.top_per_marker_pair,
                        maximum_total_frames=(
                            settings.maximum_total_frames
                        ),
                    )
                    write_ap02_frame_selection(
                        frame_selection, job_root
                    )
                    ap02_frame_selection_summary = (
                        frame_selection.summary
                    )
                    details.append(
                        "AP02 moving-frame selection: "
                        f"{len(frame_selection.selected_frame_ids)}/"
                        f"{frame_selection.summary['input_moving_frames']} "
                        "frames; minimum graph-preserving set "
                        f"{frame_selection.summary['minimum_graph_preserving_frames']}"
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
                static_ids = {
                    camera.id for camera in job.config.static_cameras
                }
                accepted_static = sorted(
                    item.camera_id
                    for item in job_camera_coverage
                    if item.camera_id in static_ids
                    and item.accepted_observation_count > 0
                )
                missing_static = sorted(static_ids - set(accepted_static))
                moving_accepted_count = sum(
                    1
                    for row in accepted_rows
                    if str(row.get("observer_type", "")).strip()
                    == "moving"
                )
                details.extend(
                    [
                        (
                            "Accepted static-camera observation coverage: "
                            f"{len(accepted_static)}/{len(static_ids)}"
                        ),
                        (
                            "Static cameras without accepted observations: "
                            + (", ".join(missing_static) if missing_static else "none")
                        ),
                        (
                            "Moving-camera accepted observations: "
                            f"{moving_accepted_count}"
                        ),
                        f"ArUco detection mode: {job.config.markers.detection_mode}",
                    ]
                )
                requirement = calibration_methods.get(method_id).requirements(
                    context
                )
                if not requirement.compatible:
                    errors.extend(requirement.reasons)
                if method_id == "ap01":
                    root_candidate = next(
                        (
                            item
                            for item in selections.payload[
                                "ap01_root_camera"
                            ]["candidates"]
                            if str(item["id"]) == selections.root_camera
                        ),
                        {},
                    )
                    reachable = list(
                        root_candidate.get("reachable_cameras", [])
                    )
                    unreachable = list(
                        root_candidate.get("unreachable_cameras", [])
                    )
                    details.extend(
                        [
                            (
                                "AP01 observation graph coverage: "
                                f"{len(reachable)}/"
                                f"{len(job.config.static_cameras)} cameras"
                            ),
                            (
                                "AP01 unreachable cameras: "
                                + (
                                    ", ".join(map(str, unreachable))
                                    if unreachable
                                    else "none"
                                )
                            ),
                        ]
                    )
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
                    rejected_rows = _read_observation_rows(
                        filtered.rejected_path
                    )
                    ap02_graph_diagnosis = diagnose_ap02_graph(
                        raw_rows=raw_rows,
                        accepted_rows=accepted_rows,
                        rejected_rows=rejected_rows,
                        static_camera_ids=(
                            camera.id for camera in job.config.static_cameras
                        ),
                        reference_marker_id=(
                            selections.ap02_reference_marker_id
                        ),
                    )
                    primary_component = next(
                        (
                            component
                            for component
                            in ap02_graph_diagnosis.components
                            if component.component_id
                            == ap02_graph_diagnosis.reference_component_id
                        ),
                        None,
                    )
                    if (
                        primary_component is None
                        or not primary_component.calibratable
                    ):
                        errors.append(
                            "The selected AP02 reference component is not "
                            "calibratable as Combined BA: it requires at "
                            "least two static cameras and one moving frame"
                        )
                    _write_ap02_graph_diagnosis(
                        job_root, ap02_graph_diagnosis
                    )
                    details.extend(
                        [
                            (
                                "AP02 Combined graph: "
                                f"{combined}/{expected} cameras; "
                                f"reference marker "
                                f"{selections.ap02_reference_marker_id}; "
                                f"{len(ap02_graph_diagnosis.components)} "
                                "connected components"
                            ),
                            (
                                "Missing from AP02 primary component: "
                                + ", ".join(combined_missing)
                                if combined_missing
                                else "Missing from AP02 primary component: none"
                            ),
                            (
                                "AP02 connectivity cause: "
                                + ", ".join(
                                    ap02_graph_diagnosis.cause_codes
                                )
                                + "; "
                                + ap02_graph_diagnosis.explanation
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
                    selected_markers = set(
                        selections.ap03_multi_marker_ids
                    )
                    supported_static = sorted(
                        {
                            str(camera)
                            for marker_id, item in candidates.items()
                            if marker_id in selected_markers
                            for camera in item.get("static_cameras", [])
                        }
                    )
                    missing_scale_support = sorted(
                        set(camera.id for camera in job.config.static_cameras)
                        - set(supported_static)
                    )
                    details.extend(
                        [
                            (
                                "AP03 ArUco scale-support coverage: "
                                f"{len(supported_static)}/"
                                f"{len(job.config.static_cameras)} cameras"
                            ),
                            (
                                "AP03 cameras without direct selected-marker "
                                "support: "
                                + (
                                    ", ".join(missing_scale_support)
                                    if missing_scale_support
                                    else "none"
                                )
                            ),
                            (
                                "AP03 COLMAP registration graph coverage is "
                                "reported during reconstruction; preflight "
                                "does not predict feature matching"
                            ),
                        ]
                    )
        except (
            AP02FrameSelectionError,
            ObservationQualityError,
            RuntimeError,
            ValueError,
        ) as exc:
            errors.append(str(exc))

        partial = False
        if (
            not errors
            and selections is not None
            and job.config.methods.enabled[0] == "ap02"
        ):
            partial = bool(
                ap02_graph_diagnosis is not None
                and not ap02_graph_diagnosis.complete
            )
            if partial:
                details.append(
                    "AP02 requires observation review before its primary "
                    "component and additional calibratable components may run"
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
            "observation_quality": {
                "effective": (
                    effective_quality.model_dump(mode="json")
                    if effective_quality is not None
                    else None
                ),
                "sources": quality_sources,
            },
            "detection_mode": job.config.markers.detection_mode,
            "ap02_combined_graph": (
                ap02_graph_diagnosis.model_dump()
                if ap02_graph_diagnosis is not None
                else None
            ),
            "ap02_frame_selection": ap02_frame_selection_summary,
            "camera_coverage": [
                {
                    "camera_id": item.camera_id,
                    "required": item.required,
                    "raw_detection_count": item.raw_detection_count,
                    "accepted_observation_count": (
                        item.accepted_observation_count
                    ),
                    "marker_ids": list(item.marker_ids),
                }
                for item in job_camera_coverage
            ],
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
                    "evaluation_anchor_marker_id": (
                        selections.evaluation_anchor_marker_id
                    ),
                }
                if selections is not None
                else None
            ),
            "automatic_recommendations": (
                selections.payload.get("automatic_recommendations")
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
                camera_coverage=job_camera_coverage,
                ap02_graph_diagnosis=ap02_graph_diagnosis,
            )
        )

    common_evaluation_anchor: int | None = None
    evaluation_rows = [
        (index, job, result)
        for index, (job, result) in enumerate(
            zip(job_list, results, strict=True)
        )
        if job.config.evaluation.enabled
        and result.runnable
        and result.selections is not None
    ]
    if evaluation_rows:
        explicit = {
            int(job.config.evaluation.anchor_marker_id)
            for _, job, _ in evaluation_rows
            if isinstance(job.config.evaluation.anchor_marker_id, int)
        }
        candidate_sets = [
            set(
                int(value)
                for value in result.selections.payload[
                    "evaluation_anchor"
                ][
                    (
                        "automatic_observation_candidates"
                        if job.config.evaluation.anchor_marker_id == "auto"
                        else "observation_candidates"
                    )
                ]
            )
            for _, job, result in evaluation_rows
        ]
        common_candidates = set.intersection(*candidate_sets)
        anchor_error: str | None = None
        if len(explicit) > 1:
            anchor_error = (
                "Enabled queue jobs request conflicting explicit evaluation "
                f"anchors: {sorted(explicit)}"
            )
        elif explicit:
            requested = next(iter(explicit))
            if requested not in common_candidates:
                anchor_error = (
                    f"Evaluation anchor {requested} is not compatible with "
                    "every enabled method after its effective quality filter."
                )
            else:
                common_evaluation_anchor = requested
        elif not common_candidates:
            anchor_error = (
                "Evaluation is enabled, but no repeat-supported marker is "
                "compatible with every runnable method after their effective "
                "quality filters. Adjust filters/whitelist or disable "
                "evaluation explicitly."
            )
        else:
            aggregate: dict[int, tuple[float, int]] = {}
            for marker_id in common_candidates:
                scores: list[float] = []
                support = 0
                for _, _, result in evaluation_rows:
                    candidates = {
                        int(item["id"]): item
                        for item in result.selections.payload[
                            "ap03_single_scale_marker"
                        ]["candidates"]
                    }
                    details = candidates[marker_id]
                    scores.append(
                        float(details.get("median_selection_score") or 0.0)
                    )
                    support += int(details.get("accepted_observations", 0))
                aggregate[marker_id] = (min(scores), support)
            common_evaluation_anchor = max(
                common_candidates,
                key=lambda marker_id: (
                    aggregate[marker_id][0],
                    aggregate[marker_id][1],
                    -marker_id,
                ),
            )

        if anchor_error is not None:
            for index, _, result in evaluation_rows:
                results[index] = replace(
                    result,
                    status="FAILED_PREFLIGHT",
                    errors=(*result.errors, anchor_error),
                    details=(
                        *result.details,
                        "Common evaluation anchor: unavailable",
                    ),
                )
        else:
            for index, _, result in evaluation_rows:
                assert result.selections is not None
                payload = json.loads(
                    json.dumps(result.selections.payload)
                )
                payload["evaluation_anchor"]["selected"] = (
                    common_evaluation_anchor
                )
                payload["evaluation_anchor"]["reason"] = (
                    "one deterministic anchor frozen across all runnable "
                    "queue methods before calibration"
                )
                payload["automatic_recommendations"][
                    "evaluation_anchor_marker_id"
                ] = common_evaluation_anchor
                selections = replace(
                    result.selections,
                    evaluation_anchor_marker_id=common_evaluation_anchor,
                    payload=payload,
                )
                if result.filter_result is not None:
                    for name in (
                        "SELECTION_CANDIDATES.json",
                        "REFERENCE_SELECTIONS.json",
                    ):
                        _write_json(
                            result.filter_result.filtered_observations_root
                            / name,
                            payload,
                        )
                    write_selection_candidates_csv(
                        result.filter_result.filtered_observations_root,
                        payload,
                    )
                results[index] = replace(
                    result,
                    selections=selections,
                    details=(
                        *result.details,
                        "Common evaluation anchor frozen before methods: "
                        f"marker {common_evaluation_anchor}",
                    ),
                )
                summary_path = (
                    result.output_directory / "preflight_summary.json"
                )
                updated_summary = json.loads(
                    summary_path.read_text(encoding="utf-8")
                )
                updated_summary["resolved_selections"] = {
                    **(
                        updated_summary.get("resolved_selections")
                        or {}
                    ),
                    "evaluation_anchor_marker_id": (
                        common_evaluation_anchor
                    ),
                }
                updated_summary[
                    "common_evaluation_anchor_marker_id"
                ] = common_evaluation_anchor
                _write_json(summary_path, updated_summary)

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
