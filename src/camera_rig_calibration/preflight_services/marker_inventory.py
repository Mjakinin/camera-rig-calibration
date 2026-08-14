"""Raw marker inventory enrichment for preflight selections."""
from __future__ import annotations

import json
from dataclasses import replace
from typing import Any

from .bindings import PreflightDependencies
from .core import PreflightJobResult


def enrich_raw_marker_inventory(
    results: list[PreflightJobResult],
    raw_rows: list[dict[str, str]],
    dependencies: PreflightDependencies,
) -> list[PreflightJobResult]:
    """Attach every detected marker to each job-specific selection report."""
    _write_json = dependencies.write_json
    _observation_camera_id = dependencies.observation_camera_id
    raw_inventory: dict[int, dict[str, Any]] = {}
    for row in raw_rows:
        try:
            marker_id = int(float(row.get("marker_id", "")))
        except (TypeError, ValueError):
            continue
        item = raw_inventory.setdefault(
            marker_id,
            {
                "id": marker_id,
                "raw_observations": 0,
                "static_cameras": set(),
                "moving_frames": set(),
            },
        )
        item["raw_observations"] += 1
        observer_type = str(row.get("observer_type", "")).strip()
        if observer_type == "static":
            camera_id = _observation_camera_id(row)
            if camera_id:
                item["static_cameras"].add(camera_id)
        elif observer_type == "moving":
            frame_id = str(row.get("frame_id", "")).strip()
            if frame_id:
                item["moving_frames"].add(frame_id)

    # Keep the complete raw marker inventory beside every job's filtered
    # candidates. Manual common-anchor review may therefore select a genuinely
    # detected marker even when that job's quality filter rejected it.
    for result_index, result in enumerate(results):
        if result.selections is None:
            continue
        payload = json.loads(json.dumps(result.selections.payload))
        compatible = set(
            int(value)
            for value in payload["evaluation_anchor"].get(
                "observation_candidates", []
            )
        )
        automatic = set(
            int(value)
            for value in payload["evaluation_anchor"].get(
                "automatic_observation_candidates", []
            )
        )
        candidate_details = {
            int(item["id"]): item
            for item in payload["ap03_single_scale_marker"]["candidates"]
        }
        inventory_rows: list[dict[str, Any]] = []
        for marker_id, raw in sorted(raw_inventory.items()):
            details = candidate_details.get(marker_id, {})
            issues: list[str] = []
            if marker_id not in candidate_details:
                issues.append("rejected by this job's quality/whitelist filter")
            if marker_id not in compatible:
                issues.append("not reconstructable by every enabled stage")
            if marker_id not in automatic:
                issues.append("insufficient repeated support for auto")
            inventory_rows.append(
                {
                    "id": marker_id,
                    "raw_observations": raw["raw_observations"],
                    "static_cameras": sorted(raw["static_cameras"]),
                    "static_camera_count": len(raw["static_cameras"]),
                    "moving_frames": len(raw["moving_frames"]),
                    "accepted_observations": int(
                        details.get("accepted_observations", 0)
                    ),
                    "median_selection_score": details.get(
                        "median_selection_score"
                    ),
                    "median_pnp_reprojection_rmse_px": details.get(
                        "median_pnp_reprojection_rmse_px"
                    ),
                    "median_marker_area_ratio": details.get(
                        "median_marker_area_ratio"
                    ),
                    "compatible": marker_id in compatible,
                    "automatic_candidate": marker_id in automatic,
                    "issues": issues,
                }
            )
        payload["raw_marker_inventory"] = inventory_rows
        selections = replace(result.selections, payload=payload)
        results[result_index] = replace(result, selections=selections)
        if result.filter_result is not None:
            for name in (
                "SELECTION_CANDIDATES.json",
                "REFERENCE_SELECTIONS.json",
            ):
                _write_json(
                    result.filter_result.filtered_observations_root / name,
                    payload,
                )

    return results
