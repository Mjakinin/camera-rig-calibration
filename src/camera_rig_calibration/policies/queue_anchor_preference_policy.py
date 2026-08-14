from __future__ import annotations

import copy
import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable


_INSTALLED = False


def _category_name(config: Any) -> str:
    value = getattr(config.dataset, "category", "real_vehicle")
    return str(getattr(value, "value", value))


def _preferred_marker(config: Any) -> int:
    return 14 if _category_name(config) == "simulation" else 0


def _is_category_preference(config: Any) -> bool:
    anchor = config.evaluation.anchor_marker_id
    return (
        config.evaluation.anchor_selection_mode == "auto"
        and isinstance(anchor, int)
        and int(anchor) == _preferred_marker(config)
    )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _automatic_anchor_candidates(result: Any) -> set[int]:
    if result.selections is None:
        return set()
    values = result.selections.payload.get("evaluation_anchor", {}).get(
        "automatic_observation_candidates", []
    )
    output: set[int] = set()
    for value in values:
        try:
            output.add(int(value))
        except (TypeError, ValueError):
            continue
    return output


def _install_queue_preflight_policy() -> None:
    from .. import observations, preflight, queueing
    from .common_anchor_authority_policy import _preferred_export_compatible

    original = preflight.run_queue_preflight
    if getattr(original, "_rigcal_queue_anchor_preference", False):
        return

    def run_queue_preflight(
        jobs: Iterable[Any],
        *,
        raw_observations_csv: Path,
        dataset_root: Path,
        output_directory: Path,
        repository_root: Path,
    ):
        requested_jobs = list(jobs)
        preferred_by_job: dict[str, int] = {}
        effective_jobs = []
        for job in requested_jobs:
            config = job.config
            if not _is_category_preference(config):
                effective_jobs.append(job)
                continue
            preferred_by_job[job.job_id] = _preferred_marker(config)
            # Let the core compute its ordinary common-candidate evidence first.
            # The policy below then freezes the category rule deliberately.
            evaluation = config.evaluation.model_copy(
                update={"anchor_marker_id": "auto", "anchor_selection_mode": "auto"}
            )
            effective_jobs.append(
                preflight.PreflightJob(
                    job.job_id,
                    config.model_copy(update={"evaluation": evaluation}, deep=True),
                )
            )

        result = original(
            effective_jobs,
            raw_observations_csv=raw_observations_csv,
            dataset_root=dataset_root,
            output_directory=output_directory,
            repository_root=repository_root,
        )
        if not preferred_by_job:
            return result

        paired = list(zip(requested_jobs, result.jobs, strict=True))
        active = [
            (job, job_result)
            for job, job_result in paired
            if job.config.evaluation.enabled and job_result.selections is not None
        ]
        if not active:
            return result

        real_rows = [
            (job, job_result)
            for job, job_result in active
            if _category_name(job.config) == "real_vehicle"
            and preferred_by_job.get(job.job_id) == 0
        ]

        # REAL VEHICLE CONTRACT:
        # marker 0 is the canonical AP02/AP03/evaluation reference whenever it is
        # observed by any enabled method after filtering. We do not silently switch
        # to marker 2/3/... because it ranks better. If marker 0 is present but is
        # not export-compatible for every enabled method, preflight fails. Only a
        # genuine absence of marker 0 permits deterministic automatic fallback.
        real_zero_observed = any(
            0 in set(job_result.selections.marker_ids)
            for _, job_result in real_rows
            if job_result.selections is not None
        )
        real_zero_failures: list[str] = []
        if real_rows and real_zero_observed:
            for requested, job_result in real_rows:
                if job_result.selections is None:
                    continue
                if not _preferred_export_compatible(
                    requested.config, job_result.selections, 0
                ):
                    real_zero_failures.append(requested.job_id)

        if real_zero_failures:
            error = (
                "Real Vehicle canonical marker 0 was observed, but it is not "
                "export-compatible for every enabled method after filtering. "
                "Automatic fallback is prohibited while marker 0 is observed; "
                "fix marker-0 support/quality or explicitly change the scientific "
                "reference. Affected jobs: " + ", ".join(sorted(real_zero_failures))
            )
            failed_jobs = []
            for requested, job_result in paired:
                if _category_name(requested.config) == "real_vehicle" and requested.config.evaluation.enabled:
                    failed_jobs.append(
                        replace(
                            job_result,
                            status="FAILED_PREFLIGHT",
                            errors=(*job_result.errors, error),
                            details=(*job_result.details, "Common evaluation/export anchor marker 0 required; fallback blocked"),
                        )
                    )
                else:
                    failed_jobs.append(job_result)
            result = replace(
                result,
                status="FAILED_PREFLIGHT",
                jobs=tuple(failed_jobs),
                common_evaluation_anchor_marker_id=None,
            )
            summary = result.output_directory / "queue_preflight_summary.json"
            try:
                payload = json.loads(summary.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                payload = {}
            if payload:
                payload["status"] = "FAILED_PREFLIGHT"
                payload["common_evaluation_anchor_marker_id"] = None
                payload["real_vehicle_marker_zero_policy"] = {
                    "canonical_marker_id": 0,
                    "marker_zero_observed": True,
                    "fallback_allowed": False,
                    "status": "failed_marker_zero_not_export_compatible",
                    "affected_jobs": sorted(real_zero_failures),
                    "ground_truth_used": False,
                }
                _write_json(summary, payload)
            return result

        # If marker 0 is present and valid, freeze 0. If it is genuinely absent,
        # the core automatic common anchor is allowed to stand. Simulation keeps
        # the established marker-14 preference-with-fallback semantics.
        chosen = result.common_evaluation_anchor_marker_id
        if real_rows and real_zero_observed:
            chosen = 0
        elif not real_rows:
            preferences = {
                preferred_by_job[job.job_id]
                for job, _ in active
                if job.job_id in preferred_by_job
            }
            preferred = next(iter(preferences)) if len(preferences) == 1 else None
            if preferred is not None and all(
                preferred in _automatic_anchor_candidates(job_result)
                for _, job_result in active
            ):
                chosen = preferred

        if chosen is None:
            return result

        updated_jobs = []
        for requested, job_result in paired:
            if not requested.config.evaluation.enabled or job_result.selections is None:
                updated_jobs.append(job_result)
                continue

            payload = copy.deepcopy(job_result.selections.payload)
            anchor_payload = payload.setdefault("evaluation_anchor", {})
            preferred = preferred_by_job.get(requested.job_id)
            category = _category_name(requested.config)
            if preferred is not None:
                if category == "real_vehicle" and preferred == 0:
                    fallback_used = not real_zero_observed
                    mode = "required_if_observed_else_auto_fallback"
                    reason = (
                        "Real Vehicle canonical evaluation/export anchor marker 0 was observed and frozen"
                        if not fallback_used
                        else f"marker 0 has zero accepted observations; deterministic common-anchor fallback selected marker {chosen}"
                    )
                else:
                    fallback_used = int(chosen) != int(preferred)
                    mode = "preferred_with_auto_fallback"
                    reason = (
                        f"preferred common anchor {preferred} is compatible with every runnable method; preference frozen"
                        if not fallback_used
                        else f"preferred common anchor {preferred} is unavailable; deterministic common-anchor fallback selected marker {chosen}"
                    )
                anchor_payload.update(
                    {
                        "configured": int(preferred),
                        "selection_mode": mode,
                        "preferred_marker_id": int(preferred),
                        "fallback_used": fallback_used,
                        "selected": int(chosen),
                        "reason": reason,
                    }
                )
                category_payload = payload.setdefault("category_marker_preference", {})
                category_payload.update(
                    {
                        "dataset_category": category,
                        "category_default_marker_id": int(preferred),
                        "ground_truth_used": False,
                    }
                )
                category_payload["evaluation_anchor"] = {
                    "preferred": int(preferred),
                    "selected": int(chosen),
                    "fallback_used": fallback_used,
                }
                if category == "real_vehicle":
                    payload["real_vehicle_marker_zero_policy"] = {
                        "canonical_marker_id": 0,
                        "marker_zero_observed": real_zero_observed,
                        "fallback_allowed": not real_zero_observed,
                        "rule": "marker_0_required_if_observed_else_auto_fallback",
                        "ground_truth_used": False,
                    }
            else:
                anchor_payload["selected"] = int(chosen)

            payload.setdefault("automatic_recommendations", {})[
                "evaluation_anchor_marker_id"
            ] = int(chosen)
            selections = replace(
                job_result.selections,
                evaluation_anchor_marker_id=int(chosen),
                payload=payload,
            )
            details = tuple(
                detail
                for detail in job_result.details
                if not str(detail).startswith("Common evaluation anchor frozen before methods:")
            ) + (
                f"Common evaluation anchor frozen before methods: marker {chosen}",
            )
            updated = replace(job_result, selections=selections, details=details)
            updated_jobs.append(updated)

            if job_result.filter_result is not None:
                root = job_result.filter_result.filtered_observations_root
                for name in ("SELECTION_CANDIDATES.json", "REFERENCE_SELECTIONS.json"):
                    _write_json(root / name, payload)
                observations.write_selection_candidates_csv(root, payload)

            summary_path = job_result.output_directory / "preflight_summary.json"
            try:
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                summary = {}
            if summary:
                selected = dict(summary.get("resolved_selections") or {})
                selected["evaluation_anchor_marker_id"] = int(chosen)
                summary["resolved_selections"] = selected
                summary["common_evaluation_anchor_marker_id"] = int(chosen)
                _write_json(summary_path, summary)

        result = replace(
            result,
            jobs=tuple(updated_jobs),
            common_evaluation_anchor_marker_id=int(chosen),
        )
        queue_summary = result.output_directory / "queue_preflight_summary.json"
        try:
            summary_payload = json.loads(queue_summary.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            summary_payload = {}
        if summary_payload:
            summary_payload["common_evaluation_anchor_marker_id"] = int(chosen)
            if real_rows:
                summary_payload["real_vehicle_marker_zero_policy"] = {
                    "canonical_marker_id": 0,
                    "marker_zero_observed": real_zero_observed,
                    "fallback_allowed": not real_zero_observed,
                    "selected": int(chosen),
                    "ground_truth_used": False,
                }
            _write_json(queue_summary, summary_payload)
        return result

    run_queue_preflight._rigcal_queue_anchor_preference = True  # type: ignore[attr-defined]
    preflight.run_queue_preflight = run_queue_preflight
    queueing.run_queue_preflight = run_queue_preflight


def _install_readiness_error_display() -> None:
    from .. import queueing

    original = queueing._method_preflight_coverage
    if getattr(original, "_rigcal_readiness_error_display", False):
        return

    def method_preflight_coverage(config, report):
        coverage, reason = original(config, report)
        if report.errors:
            error_text = "; ".join(str(value) for value in report.errors)
            reason = f"ERROR: {error_text}; coverage: {reason}" if reason else f"ERROR: {error_text}"
        return coverage, reason

    method_preflight_coverage._rigcal_readiness_error_display = True  # type: ignore[attr-defined]
    queueing._method_preflight_coverage = method_preflight_coverage


def install_queue_anchor_preference_policy() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    _install_queue_preflight_policy()
    _install_readiness_error_display()
    _INSTALLED = True
