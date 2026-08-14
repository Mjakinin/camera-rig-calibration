"""Method-specific queue preflight phase."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from ..contracts import RunContext
from ..methods.ap02.frame_selection import AP02FrameSelectionError
from ..observation_quality import ObservationQualityError
from ..registry import calibration_methods
from .bindings import PreflightDependencies
from .core import (
    CameraObservationCoverage,
    PreflightJob,
    PreflightJobResult,
)


def run_job_preflight(
    job: PreflightJob,
    *,
    raw_observations_csv: Path,
    raw_rows: list[dict[str, str]],
    dataset_root: Path,
    destination: Path,
    repository_root: Path,
    queue_camera_coverage: tuple[CameraObservationCoverage, ...],
    dependencies: PreflightDependencies,
) -> PreflightJobResult:
    """Validate and materialize one method job without running calibration."""
    effective_observation_quality = dependencies.effective_observation_quality
    filter_observations = dependencies.filter_observations
    _read_observation_rows = dependencies.read_observation_rows
    _observation_camera_id = dependencies.observation_camera_id
    graph_components = dependencies.graph_components
    diagnose_ap02_graph = dependencies.diagnose_ap02_graph
    _write_ap02_graph_diagnosis = dependencies.write_ap02_graph_diagnosis
    resolve_selections = dependencies.resolve_selections
    select_ap02_frames = dependencies.select_ap02_frames
    write_ap02_frame_selection = dependencies.write_ap02_frame_selection
    _write_json = dependencies.write_json
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
    return PreflightJobResult(
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
