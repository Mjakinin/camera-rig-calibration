"""Common evaluation-anchor resolution across runnable queue jobs."""
from __future__ import annotations

import json
from dataclasses import replace

from .bindings import PreflightDependencies
from .core import PreflightJob, PreflightJobResult


def resolve_common_evaluation_anchor(
    job_list: list[PreflightJob],
    results: list[PreflightJobResult],
    dependencies: PreflightDependencies,
) -> tuple[list[PreflightJobResult], int | None]:
    """Choose or validate the single queue-wide evaluation anchor."""
    _write_json = dependencies.write_json
    write_selection_candidates_csv = (
        dependencies.write_selection_candidates_csv
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
        manual_review = any(
            job.config.evaluation.anchor_selection_mode == "review_once"
            for _, job, _ in evaluation_rows
        )
        explicit = {
            int(job.config.evaluation.anchor_marker_id)
            for _, job, _ in evaluation_rows
            if isinstance(job.config.evaluation.anchor_marker_id, int)
        }
        strict_candidate_sets = [
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
        strict_common_candidates = set.intersection(*strict_candidate_sets)
        raw_candidate_sets = [
            {
                int(item["id"])
                for item in result.selections.payload.get(
                    "raw_marker_inventory", []
                )
            }
            for _, _, result in evaluation_rows
        ]
        raw_common_candidates = (
            set.intersection(*raw_candidate_sets)
            if raw_candidate_sets
            else set()
        )
        common_candidates = (
            raw_common_candidates if manual_review else strict_common_candidates
        )
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
                "Evaluation is enabled, but shared detection found no marker "
                "that can be selected as the common anchor."
            )
        elif manual_review:
            if strict_common_candidates:
                aggregate: dict[int, tuple[float, int]] = {}
                for marker_id in strict_common_candidates:
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
                            float(
                                details.get("median_selection_score") or 0.0
                            )
                        )
                        support += int(
                            details.get("accepted_observations", 0)
                        )
                    aggregate[marker_id] = (min(scores), support)
                common_evaluation_anchor = max(
                    strict_common_candidates,
                    key=lambda marker_id: (
                        aggregate[marker_id][0],
                        aggregate[marker_id][1],
                        -marker_id,
                    ),
                )
            else:
                common_evaluation_anchor = None
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
                    (
                        "recommended common anchor awaiting one manual "
                        "post-preflight decision"
                        if manual_review
                        else "one deterministic anchor frozen across all "
                        "runnable queue methods before calibration"
                    )
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
                        + (
                            f"marker {common_evaluation_anchor}"
                            if common_evaluation_anchor is not None
                            else "manual selection pending"
                        ),
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
    return results, common_evaluation_anchor
