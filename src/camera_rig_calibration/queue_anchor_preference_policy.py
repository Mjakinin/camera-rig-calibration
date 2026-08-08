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
    result_ids: set[int] = set()
    for value in values:
        try:
            result_ids.add(int(value))
        except (TypeError, ValueError):
            continue
    return result_ids


def _install_queue_preflight_policy() -> None:
    from . import observations, preflight, queueing

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
            evaluation = config.evaluation.model_copy(
                update={
                    # The integer is a preference, not an explicit request. The
                    # core queue aggregator must therefore compute its normal
                    # automatic common-candidate intersection first.
                    "anchor_marker_id": "auto",
                    "anchor_selection_mode": "auto",
                }
            )
            effective_config = config.model_copy(
                update={"evaluation": evaluation}, deep=True
            )
            effective_jobs.append(
                preflight.PreflightJob(job.job_id, effective_config)
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

        active_rows = [
            (job, job_result)
            for job, job_result in zip(
                requested_jobs, result.jobs, strict=True
            )
            if job.config.evaluation.enabled
            and job_result.runnable
            and job_result.selections is not None
        ]
        if not active_rows:
            return result

        preferences = {
            preferred_by_job[job.job_id]
            for job, _ in active_rows
            if job.job_id in preferred_by_job
        }
        category_preferred = (
            next(iter(preferences)) if len(preferences) == 1 else None
        )
        preferred_supported = bool(
            category_preferred is not None
            and all(
                category_preferred in _automatic_anchor_candidates(job_result)
                for _, job_result in active_rows
            )
        )
        chosen = (
            category_preferred
            if preferred_supported
            else result.common_evaluation_anchor_marker_id
        )
        if chosen is None:
            # Keep the core no-common-anchor result unchanged. This policy only
            # changes preference semantics; it never invents an unsupported
            # anchor.
            return result

        updated_jobs = []
        for requested, job_result in zip(
            requested_jobs, result.jobs, strict=True
        ):
            if (
                not requested.config.evaluation.enabled
                or not job_result.runnable
                or job_result.selections is None
            ):
                updated_jobs.append(job_result)
                continue

            payload = copy.deepcopy(job_result.selections.payload)
            anchor_payload = payload.setdefault("evaluation_anchor", {})
            preferred = preferred_by_job.get(requested.job_id)
            if preferred is not None:
                fallback_used = int(chosen) != int(preferred)
                anchor_payload.update(
                    {
                        "configured": int(preferred),
                        "selection_mode": "preferred_with_auto_fallback",
                        "preferred_marker_id": int(preferred),
                        "fallback_used": fallback_used,
                        "selected": int(chosen),
                        "reason": (
                            f"preferred common anchor {preferred} is compatible "
                            "with every runnable method; preference frozen"
                            if not fallback_used
                            else f"preferred common anchor {preferred} is not "
                            "compatible with every runnable method; deterministic "
                            f"common-anchor fallback selected marker {chosen}"
                        ),
                    }
                )
                category_payload = payload.setdefault(
                    "category_marker_preference",
                    {
                        "dataset_category": _category_name(requested.config),
                        "category_default_marker_id": int(preferred),
                        "ground_truth_used": False,
                    },
                )
                category_payload["evaluation_anchor"] = {
                    "preferred": int(preferred),
                    "selected": int(chosen),
                    "fallback_used": fallback_used,
                }
                category_payload["ground_truth_used"] = False
            else:
                anchor_payload["selected"] = int(chosen)

            automatic = payload.setdefault("automatic_recommendations", {})
            automatic["evaluation_anchor_marker_id"] = int(chosen)
            selections = replace(
                job_result.selections,
                evaluation_anchor_marker_id=int(chosen),
                payload=payload,
            )
            details = tuple(
                detail
                for detail in job_result.details
                if not str(detail).startswith(
                    "Common evaluation anchor frozen before methods:"
                )
            ) + (
                "Common evaluation anchor frozen before methods: "
                f"marker {chosen}"
                + (
                    f" (preferred {preferred})"
                    if preferred is not None and int(chosen) == int(preferred)
                    else f" (automatic fallback from preferred {preferred})"
                    if preferred is not None
                    else ""
                ),
            )
            updated = replace(
                job_result,
                selections=selections,
                details=details,
            )
            updated_jobs.append(updated)

            if job_result.filter_result is not None:
                root = job_result.filter_result.filtered_observations_root
                for name in (
                    "SELECTION_CANDIDATES.json",
                    "REFERENCE_SELECTIONS.json",
                ):
                    _write_json(root / name, payload)
                observations.write_selection_candidates_csv(root, payload)

            summary_path = job_result.output_directory / "preflight_summary.json"
            try:
                summary = json.loads(
                    summary_path.read_text(encoding="utf-8")
                )
            except (OSError, json.JSONDecodeError):
                summary = {}
            if summary:
                resolved = dict(summary.get("resolved_selections") or {})
                resolved["evaluation_anchor_marker_id"] = int(chosen)
                summary["resolved_selections"] = resolved
                summary["common_evaluation_anchor_marker_id"] = int(chosen)
                summary["category_anchor_preference"] = {
                    "preferred": preferred,
                    "selected": int(chosen),
                    "fallback_used": (
                        int(chosen) != int(preferred)
                        if preferred is not None
                        else None
                    ),
                }
                _write_json(summary_path, summary)

        result = replace(
            result,
            jobs=tuple(updated_jobs),
            common_evaluation_anchor_marker_id=int(chosen),
        )
        queue_summary_path = result.output_directory / "queue_preflight_summary.json"
        try:
            queue_summary = json.loads(
                queue_summary_path.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError):
            queue_summary = {}
        if queue_summary:
            queue_summary["common_evaluation_anchor_marker_id"] = int(chosen)
            queue_summary["category_anchor_preference"] = {
                "preferred": category_preferred,
                "selected": int(chosen),
                "fallback_used": (
                    int(chosen) != int(category_preferred)
                    if category_preferred is not None
                    else None
                ),
            }
            _write_json(queue_summary_path, queue_summary)
        return result

    run_queue_preflight._rigcal_queue_anchor_preference = True  # type: ignore[attr-defined]
    preflight.run_queue_preflight = run_queue_preflight
    # queueing imports the function by name, so update the already-bound
    # reference as well. This keeps Wizard and non-interactive queue execution
    # on the same semantics.
    queueing.run_queue_preflight = run_queue_preflight


def _install_readiness_error_display() -> None:
    from . import queueing

    original = queueing._method_preflight_coverage
    if getattr(original, "_rigcal_readiness_error_display", False):
        return

    def method_preflight_coverage(config, report):
        coverage, reason = original(config, report)
        if report.errors:
            error_text = "; ".join(str(value) for value in report.errors)
            reason = (
                f"ERROR: {error_text}; coverage: {reason}"
                if reason
                else f"ERROR: {error_text}"
            )
        return coverage, reason

    method_preflight_coverage._rigcal_readiness_error_display = True  # type: ignore[attr-defined]
    queueing._method_preflight_coverage = method_preflight_coverage


def install_queue_anchor_preference_policy() -> None:
    """Treat category marker IDs as preferences, never implicit explicit anchors."""
    global _INSTALLED
    if _INSTALLED:
        return
    _install_queue_preflight_policy()
    _install_readiness_error_display()
    _INSTALLED = True
