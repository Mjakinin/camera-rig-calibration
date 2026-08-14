from __future__ import annotations

import copy
import csv
import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable

import yaml


_INSTALLED = False
_EVALUATION_ONLY_PREFIXES = (
    "Real Vehicle canonical marker 0 was observed, but it is not export-compatible",
    "Evaluation is enabled, but shared detection found no marker",
    "Enabled queue jobs request conflicting explicit evaluation anchors",
)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _category(config: Any) -> str:
    value = getattr(config.dataset, "category", "real_vehicle")
    return str(getattr(value, "value", value))


def _evaluation_only_error(value: object) -> bool:
    text = str(value)
    if text.startswith(_EVALUATION_ONLY_PREFIXES):
        return True
    return text.startswith("Evaluation anchor ") and (
        "not compatible with every enabled method" in text
        or "not compatible with every enabled method after" in text
    )


def _calibration_readiness(config: Any, report: Any, errors: tuple[str, ...]) -> str:
    if errors:
        return "FAILED_PREFLIGHT"
    method_id = str(config.methods.enabled[0])
    if (
        method_id == "ap02"
        and report.ap02_graph_diagnosis is not None
        and not report.ap02_graph_diagnosis.complete
    ):
        return "READY_PARTIAL"
    if report.warnings:
        return "READY_WITH_WARNINGS"
    return "READY"


def _selection_with_real_anchor(report: Any, marker_id: int = 0) -> Any:
    """Keep marker 0 requested without falsifying compatibility evidence."""

    selections = report.selections
    if selections is None or marker_id not in set(selections.marker_ids):
        return report
    payload = copy.deepcopy(selections.payload)
    anchor = payload.setdefault("evaluation_anchor", {})
    anchor.update(
        {
            "configured": marker_id,
            "selected": marker_id,
            "preferred_marker_id": marker_id,
            "selection_mode": "real_vehicle_canonical_nonblocking_v1",
            "resolution_stage": "preflight",
            "reason": (
                "Real Vehicle canonical marker 0 remains the requested common "
                "evaluation/export anchor. Calibration readiness is independent "
                "from common-anchor export observability; methods that cannot "
                "express a partial result in marker 0 remain runnable and are "
                "reported as evaluation unavailable/not observable. Existing "
                "observation/automatic compatibility candidate sets are retained "
                "unchanged."
            ),
        }
    )
    payload["real_vehicle_marker_zero_policy"] = {
        "canonical_marker_id": marker_id,
        "marker_zero_observed": True,
        "selected": marker_id,
        "calibration_gating": False,
        "common_evaluation_requested": True,
        "compatibility_evidence_overridden": False,
        "unobservable_method_results_are_reported_not_failed": True,
        "ground_truth_used": False,
    }
    updated_selections = replace(
        selections,
        evaluation_anchor_marker_id=marker_id,
        payload=payload,
    )
    updated = replace(report, selections=updated_selections)
    if report.filter_result is not None:
        from .. import observations

        root = report.filter_result.filtered_observations_root
        for name in ("SELECTION_CANDIDATES.json", "REFERENCE_SELECTIONS.json"):
            _write_json(root / name, payload)
        observations.write_selection_candidates_csv(root, payload)
    return updated


def _rewrite_preflight_summary(report: Any) -> None:
    summary_path = report.output_directory / "preflight_summary.json"
    summary = _read_json(summary_path)
    if not summary:
        return
    summary["status"] = report.status
    summary["errors"] = list(report.errors)
    summary["warnings"] = list(report.warnings)
    summary["details"] = list(report.details)
    if report.selections is not None:
        resolved = dict(summary.get("resolved_selections") or {})
        resolved["evaluation_anchor_marker_id"] = (
            report.selections.evaluation_anchor_marker_id
        )
        summary["resolved_selections"] = resolved
        summary["common_evaluation_anchor_marker_id"] = (
            report.selections.evaluation_anchor_marker_id
        )
    summary["evaluation_readiness_policy"] = {
        "enabled": True,
        "calibration_gating": False,
        "compatibility_evidence_overridden": False,
        "unobservable_evaluation_is_reported": True,
    }
    _write_json(summary_path, summary)


def _queue_status(jobs: tuple[Any, ...], review_reasons: tuple[str, ...]) -> str:
    if review_reasons:
        return "REVIEW_REQUIRED"
    runnable = [job for job in jobs if job.runnable]
    if not runnable:
        return "FAILED_PREFLIGHT"
    failed = [job for job in jobs if not job.runnable]
    if failed or any(job.status == "READY_PARTIAL" for job in runnable):
        return "READY_PARTIAL"
    if any(job.status == "READY_WITH_WARNINGS" for job in runnable):
        return "READY_WITH_WARNINGS"
    return "READY"


def _install_nonblocking_preflight() -> None:
    from .. import preflight, queueing

    original = preflight.run_queue_preflight
    if getattr(original, "_rigcal_real_partial_evaluation_nonblocking", False):
        return

    def run_queue_preflight(
        jobs: Iterable[Any],
        *,
        raw_observations_csv: Path,
        dataset_root: Path,
        output_directory: Path,
        repository_root: Path,
    ):
        requested = list(jobs)
        result = original(
            requested,
            raw_observations_csv=raw_observations_csv,
            dataset_root=dataset_root,
            output_directory=output_directory,
            repository_root=repository_root,
        )
        paired = list(zip(requested, result.jobs, strict=True))
        real_active = [
            (job, report)
            for job, report in paired
            if _category(job.config) == "real_vehicle"
            and bool(job.config.evaluation.enabled)
        ]
        if not real_active:
            return result

        marker_zero_observed = any(
            report.selections is not None
            and 0 in set(report.selections.marker_ids)
            for _, report in real_active
        )
        updated_jobs: list[Any] = []
        for requested_job, report in paired:
            config = requested_job.config
            if _category(config) != "real_vehicle" or not config.evaluation.enabled:
                updated_jobs.append(report)
                continue

            calibration_errors = tuple(
                str(error)
                for error in report.errors
                if not _evaluation_only_error(error)
            )
            removed_evaluation_errors = tuple(
                str(error)
                for error in report.errors
                if _evaluation_only_error(error)
            )
            warnings = list(report.warnings)
            details = list(report.details)
            if removed_evaluation_errors:
                warning = (
                    "Common evaluation/export observability is incomplete, but "
                    "evaluation remains enabled and does not gate calibration. "
                    "Unavailable method/anchor combinations will be reported "
                    "after calibration instead of failing method preflight."
                )
                if warning not in warnings:
                    warnings.append(warning)
                details.append(
                    "Non-blocking evaluation preflight removed evaluation-only "
                    "readiness errors: " + " | ".join(removed_evaluation_errors)
                )
            status = _calibration_readiness(
                config,
                report,
                calibration_errors,
            )
            updated = replace(
                report,
                status=status,
                errors=calibration_errors,
                warnings=tuple(warnings),
                details=tuple(details),
            )
            if marker_zero_observed:
                updated = _selection_with_real_anchor(updated, 0)
            _rewrite_preflight_summary(updated)
            updated_jobs.append(updated)

        jobs_tuple = tuple(updated_jobs)
        common_anchor = (
            0 if marker_zero_observed else result.common_evaluation_anchor_marker_id
        )
        status = _queue_status(jobs_tuple, result.review_reasons)
        updated_result = replace(
            result,
            status=status,
            jobs=jobs_tuple,
            common_evaluation_anchor_marker_id=common_anchor,
        )
        queue_summary = updated_result.output_directory / "queue_preflight_summary.json"
        summary = _read_json(queue_summary)
        if summary:
            runnable = [job for job in jobs_tuple if job.runnable]
            summary.update(
                {
                    "status": status,
                    "methods_may_start": bool(runnable),
                    "common_evaluation_anchor_marker_id": common_anchor,
                    "runnable_jobs": [job.job_id for job in runnable],
                    "skipped_jobs": [
                        job.job_id for job in jobs_tuple if not job.runnable
                    ],
                    "real_vehicle_evaluation_policy": {
                        "enabled": True,
                        "canonical_marker_id": 0,
                        "marker_zero_observed": marker_zero_observed,
                        "calibration_gating": False,
                        "compatibility_evidence_overridden": False,
                        "unobservable_evaluation_is_reported": True,
                        "ground_truth_used": False,
                    },
                }
            )
            summary["jobs"] = [
                {
                    "job_id": job.job_id,
                    "status": job.status,
                    "errors": list(job.errors),
                    "warnings": list(job.warnings),
                    "details": list(job.details),
                    "preflight_summary": str(
                        job.output_directory / "preflight_summary.json"
                    ),
                }
                for job in jobs_tuple
            ]
            _write_json(queue_summary, summary)
        return updated_result

    run_queue_preflight._rigcal_real_partial_evaluation_nonblocking = True  # type: ignore[attr-defined]
    preflight.run_queue_preflight = run_queue_preflight
    queueing.run_queue_preflight = run_queue_preflight


def _install_real_wizard_evaluation_contract() -> None:
    from .. import wizard
    from .product_policy import _DATASET_CONTEXT

    original_new_method_job = wizard._new_method_job
    if not getattr(original_new_method_job, "_rigcal_real_evaluation_required", False):
        def new_method_job(*args, **kwargs):
            job = original_new_method_job(*args, **kwargs)
            if _DATASET_CONTEXT.get() == "real_vehicle":
                job.evaluation = job.evaluation.model_copy(update={"enabled": True})
            return job

        new_method_job._rigcal_real_evaluation_required = True  # type: ignore[attr-defined]
        wizard._new_method_job = new_method_job

    original_setting_rows = wizard._setting_rows
    if not getattr(original_setting_rows, "_rigcal_real_evaluation_required", False):
        def setting_rows(job, groups=None):
            rows = original_setting_rows(job, groups)
            if _DATASET_CONTEXT.get() != "real_vehicle":
                return rows
            rendered = []
            for key, group, label, current, baseline, description in rows:
                if key == "evaluation_enabled":
                    # Real Vehicle evaluation is part of the final reporting
                    # contract. It is attempted even when a partial method cannot
                    # participate in the common full-rig comparison.
                    continue
                if key == "evaluation_anchor":
                    description = (
                        "Real Vehicle common evaluation is always enabled. Marker "
                        "0 remains canonical when observed. If a partial method "
                        "cannot express its result in that frame, calibration "
                        "continues and RESULT reports evaluation as unavailable/"
                        "not observable instead of failing the method."
                    )
                rendered.append(
                    (key, group, label, current, baseline, description)
                )
            return rendered

        setting_rows._rigcal_real_evaluation_required = True  # type: ignore[attr-defined]
        wizard._setting_rows = setting_rows

    original_edit_method_job = wizard._edit_method_job
    if not getattr(original_edit_method_job, "_rigcal_real_evaluation_required", False):
        def edit_method_job(*args, **kwargs):
            if len(args) >= 2:
                job = args[1]
            else:
                job = kwargs.get("job")
            if job is not None and _DATASET_CONTEXT.get() == "real_vehicle":
                job.evaluation = job.evaluation.model_copy(update={"enabled": True})
            updated = original_edit_method_job(*args, **kwargs)
            if _DATASET_CONTEXT.get() == "real_vehicle":
                updated.evaluation = updated.evaluation.model_copy(
                    update={"enabled": True}
                )
            return updated

        edit_method_job._rigcal_real_evaluation_required = True  # type: ignore[attr-defined]
        wizard._edit_method_job = edit_method_job


def _component_summary_text(payload: dict[str, Any]) -> str:
    if str(payload.get("method")) != "ap02":
        return ""
    metrics = payload.get("metrics", {})
    summary = metrics.get("ap02_component_results", {}) if isinstance(metrics, dict) else {}
    if not isinstance(summary, dict) or not summary:
        return ""
    components = [
        item for item in summary.get("components", []) if isinstance(item, dict)
    ]
    primary = str(summary.get("primary_component_id") or "-")
    rows = []
    for item in components:
        component_id = str(item.get("component_id", "-"))
        role = "primary" if component_id == primary else "diagnostic"
        local_reference = item.get(
            "local_reference_marker_id", item.get("anchor_marker_id", "-")
        )
        rows.append(
            [
                component_id,
                role,
                str(item.get("execution_status", "-")),
                str(local_reference),
                ",".join(map(str, item.get("static_cameras", []))) or "-",
                ",".join(map(str, item.get("marker_ids", []))) or "-",
                str(item.get("moving_frame_count", "-")),
                str(item.get("quality_status", "-")),
                str(item.get("result_path", "-")),
            ]
        )

    def table(headers: list[str], values: list[list[str]]) -> str:
        rendered = [[str(cell) for cell in row] for row in values]
        widths = [
            max([len(header), *(len(row[i]) for row in rendered)])
            for i, header in enumerate(headers)
        ] if rendered else [len(header) for header in headers]
        lines = [
            " | ".join(header.ljust(widths[i]) for i, header in enumerate(headers)),
            "-+-".join("-" * width for width in widths),
        ]
        lines.extend(
            " | ".join(cell.ljust(widths[i]) for i, cell in enumerate(row))
            for row in rendered
        )
        return "\n".join(lines)

    pair_rows = [
        row for row in summary.get("camera_pair_observability", [])
        if isinstance(row, dict)
    ]
    within = sum(row.get("status") == "within_component" for row in pair_rows)
    unobservable = sum(row.get("status") == "not_observable" for row in pair_rows)
    return "\n".join(
        [
            "AP02 DISCONNECTED / PARTIAL COMPONENT RESULTS",
            "-" * 118,
            f"Overall component status: {summary.get('status', '-')}",
            f"Primary component: {primary}",
            (
                "Cross-component extrinsics: "
                f"{summary.get('cross_component_extrinsics', 'not_observable')}"
            ),
            (
                "Camera-pair observability: "
                f"within-component={within}, cross-component/unobservable={unobservable}"
            ),
            "Each component keeps its own local marker frame. Components are never artificially aligned.",
            table(
                [
                    "Component",
                    "Role",
                    "Execution",
                    "Local ref",
                    "Cameras",
                    "Markers",
                    "Moving",
                    "Quality",
                    "Artifact",
                ],
                rows,
            ),
            "",
        ]
    )


def _read_csv(path: Path) -> list[dict[str, str]]:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))
    except OSError:
        return []


def _fmt_pose(row: dict[str, str]) -> str:
    camera = str(row.get("entity_id") or row.get("camera_id") or "-")
    xyz = ", ".join(
        f"{axis}={float(row[axis]):.6f}m"
        for axis in ("x_m", "y_m", "z_m")
        if str(row.get(axis, "")).strip()
    )
    if all(str(row.get(key, "")).strip() for key in ("roll_deg", "pitch_deg", "yaw_deg")):
        orientation = ", ".join(
            f"{key.removesuffix('_deg')}={float(row[key]):.6f}deg"
            for key in ("roll_deg", "pitch_deg", "yaw_deg")
        )
    elif all(str(row.get(key, "")).strip() for key in ("rvec_x", "rvec_y", "rvec_z")):
        orientation = ", ".join(
            f"{key}={float(row[key]):.9f}"
            for key in ("rvec_x", "rvec_y", "rvec_z")
        )
    else:
        orientation = "orientation fields unavailable"
    return f"    {camera}: {xyz}; {orientation}"


def _component_pose_detail(result_root: Path) -> str:
    summary = _read_json(
        result_root
        / "diagnostics"
        / "method"
        / "component_diagnostics"
        / "AP02_COMPONENT_RESULTS.json"
    )
    if not summary:
        return ""
    primary = str(summary.get("primary_component_id") or "")
    lines = [
        "AP02 LOCAL COMPONENT CAMERA POSES",
        "-" * 118,
        "These are valid only inside the listed component/local marker frame.",
        "Cross-component transforms are NOT observable and are not synthesized.",
        "",
    ]
    for item in summary.get("components", []):
        if not isinstance(item, dict):
            continue
        component_id = str(item.get("component_id", "-"))
        execution = str(item.get("execution_status", "-"))
        local_reference = item.get(
            "local_reference_marker_id", item.get("anchor_marker_id", "-")
        )
        lines.append(
            f"{component_id} | execution={execution} | local frame=marker_{local_reference}"
        )
        camera_ids = set(map(str, item.get("static_cameras", [])))
        if component_id == primary:
            pose_path = result_root / "camera_extrinsics.csv"
        elif execution == "available":
            pose_path = (
                result_root
                / "diagnostics"
                / "method"
                / "component_diagnostics"
                / component_id
                / "camera_extrinsics.csv"
            )
        else:
            pose_path = Path("__missing__")
        poses = [
            row
            for row in _read_csv(pose_path)
            if str(row.get("entity_id") or row.get("camera_id") or "") in camera_ids
        ] if pose_path.is_file() else []
        if poses:
            lines.extend(_fmt_pose(row) for row in poses)
            try:
                artifact = pose_path.relative_to(result_root).as_posix()
            except ValueError:
                artifact = str(pose_path)
            lines.append(f"    artifact: {artifact}")
        else:
            reason = item.get("reason") or item.get("error") or "no local pose result"
            lines.append(f"    poses: unavailable ({reason})")
        lines.append("")
    return "\n".join(lines)


def _install_partial_result_reporting() -> None:
    from ..evaluation import reporting

    original_method_text = reporting._method_report_text
    if not getattr(original_method_text, "_rigcal_ap02_partial_components", False):
        def method_report_text(payload, poses, pairs, anchor_cameras=None):
            text = original_method_text(payload, poses, pairs, anchor_cameras)
            section = _component_summary_text(payload)
            if section and "AP02 DISCONNECTED / PARTIAL COMPONENT RESULTS" not in text:
                text = text.rstrip() + "\n\n" + section
            return text

        method_report_text._rigcal_ap02_partial_components = True  # type: ignore[attr-defined]
        reporting._method_report_text = method_report_text

    original_refresh = reporting.refresh_method_reports
    if not getattr(original_refresh, "_rigcal_ap02_partial_components", False):
        def refresh_method_reports(experiment_root: Path):
            payloads = original_refresh(experiment_root)
            root = Path(experiment_root)
            for result_path in sorted((root / "methods" / "ap02").glob("*/RESULT.json")):
                result_root = result_path.parent
                detail = _component_pose_detail(result_root)
                if not detail:
                    continue
                payload = _read_json(result_path)
                summary = payload.get("metrics", {}).get(
                    "ap02_component_results", {}
                )
                partial = isinstance(summary, dict) and (
                    summary.get("status") == "partial_coverage"
                    or summary.get("cross_component_extrinsics") == "not_observable"
                )
                if partial:
                    config_path = result_root / "provenance" / "resolved_config.yaml"
                    evaluation_requested = True
                    try:
                        config = yaml.safe_load(
                            config_path.read_text(encoding="utf-8")
                        ) or {}
                        evaluation_requested = bool(
                            config.get("evaluation", {}).get("enabled", True)
                        )
                    except (OSError, TypeError, ValueError, yaml.YAMLError):
                        evaluation_requested = True
                    payload["evaluation_requested"] = evaluation_requested
                    if evaluation_requested and payload.get("evaluation_status") in {
                        None,
                        "not_run",
                    }:
                        payload["evaluation_status"] = (
                            "unavailable_partial_cross_component"
                        )
                    _write_json(result_path, payload)
                text_path = result_root / "RESULT.txt"
                try:
                    text = text_path.read_text(encoding="utf-8")
                except OSError:
                    continue
                if partial:
                    text = text.replace(
                        "Evaluation status: not_run",
                        "Evaluation status: unavailable_partial_cross_component",
                        1,
                    )
                if "AP02 LOCAL COMPONENT CAMERA POSES" not in text:
                    text = text.rstrip() + "\n\n" + detail + "\n"
                text_path.write_text(text, encoding="utf-8")
                for index, item in enumerate(payloads):
                    if (
                        str(item.get("method")) == "ap02"
                        and str(item.get("label")) == result_root.name
                    ):
                        payloads[index] = payload
            return payloads

        refresh_method_reports._rigcal_ap02_partial_components = True  # type: ignore[attr-defined]
        reporting.refresh_method_reports = refresh_method_reports

    original_real_text = reporting._real_results_text
    if not getattr(original_real_text, "_rigcal_ap02_partial_components", False):
        def real_results_text(experiment_root, method_payloads, dataset_root=None):
            text, payload = original_real_text(
                experiment_root, method_payloads, dataset_root
            )
            sections = [
                _component_summary_text(item)
                for item in method_payloads
                if str(item.get("method")) == "ap02"
            ]
            sections = [section for section in sections if section]
            if sections and "AP02 DISCONNECTED / PARTIAL COMPONENT RESULTS" not in text:
                text = text.rstrip() + "\n\n" + "\n".join(sections)
            return text, payload

        real_results_text._rigcal_ap02_partial_components = True  # type: ignore[attr-defined]
        reporting._real_results_text = real_results_text


def install_real_partial_evaluation_policy() -> None:
    """Keep Real Vehicle evaluation enabled without making it a method gate.

    Real Vehicle marker 0 remains the canonical requested evaluation/export
    anchor whenever observed. Calibration methods are judged only by their own
    mathematical observability. Partial AP02 components therefore remain
    runnable, cross-component relations remain explicitly not observable, and
    common evaluation failures are published as unavailable rather than being
    converted into calibration preflight failures.
    """
    global _INSTALLED
    if _INSTALLED:
        return
    _install_nonblocking_preflight()
    _install_real_wizard_evaluation_contract()
    _install_partial_result_reporting()
    _INSTALLED = True


__all__ = [
    "install_real_partial_evaluation_policy",
]
