from __future__ import annotations

import csv
import json
import math
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config.models import DatasetCategory, RigConfig



from .observation_candidates import (
    _marker_candidates,
    _marker_choice,
    _root_candidates,
)
from .observation_core import (
    ResolvedSelections,
    _ap02_rank,
    _best_candidate,
    _is_simulation_dataset,
    _marker_id,
    _observer_id,
    _read_observations,
    _root_rank,
    ap03_candidate_rank,
    write_selection_candidates_csv,
)
def resolve_selections(
    config: RigConfig, observations_root: Path
) -> ResolvedSelections:
    rows = _read_observations(observations_root)
    camera_ids = tuple(camera.id for camera in config.static_cameras)
    declared = set(camera_ids)
    observed = {
        _observer_id(row)
        for row in rows
        if row.get("observer_type") == "static"
    }
    unknown = observed - declared
    if unknown:
        raise RuntimeError(
            f"Observations contain undeclared static camera IDs: {sorted(unknown)}"
        )

    roots = _root_candidates(rows, camera_ids)
    markers = _marker_candidates(rows, camera_ids)
    enabled = set(config.methods.enabled)
    configured_root = config.methods.ap01.root_camera
    recommended_root = str(_best_candidate(roots, _root_rank))
    root = (
        recommended_root
        if configured_root == "auto"
        else configured_root
    )
    if root not in roots:
        raise RuntimeError(f"Configured AP01 root camera is not in the rig: {root}")
    if "ap01" in enabled and not roots[root]["compatible"]:
        raise RuntimeError(
            f"Configured AP01 root camera '{root}' has no successful observations"
        )

    try:
        recommended_ap02_reference: int | None = _marker_choice(
            "auto",
            markers,
            compatibility_key="ap02_compatible",
            purpose="AP02 reference",
            rank=_ap02_rank,
            require_compatibility="ap02" in enabled,
            expected_camera_ids=camera_ids,
        )
    except RuntimeError:
        if "ap02" in enabled:
            raise
        recommended_ap02_reference = None
    ap02_selection_mode = (
        config.methods.ap02.reference_marker_selection_mode
    )
    if (
        ap02_selection_mode == "baseline"
        and not _is_simulation_dataset(config)
    ):
        raise RuntimeError(
            "AP02 baseline reference-marker selection is available only "
            "for simulation datasets"
        )
    configured_ap02_reference: int | str
    if ap02_selection_mode == "baseline":
        configured_ap02_reference = 14
    elif config.methods.ap02.reference_marker_id == "auto":
        configured_ap02_reference = (
            recommended_ap02_reference
            if recommended_ap02_reference is not None
            else int(
                _best_candidate(
                    markers,
                    _ap02_rank,
                    compatibility_key="_all",
                )
            )
        )
    else:
        configured_ap02_reference = (
            config.methods.ap02.reference_marker_id
        )
    ap02_reference = _marker_choice(
        configured_ap02_reference,
        markers,
        compatibility_key="ap02_compatible",
        purpose="AP02 reference",
        rank=_ap02_rank,
        require_compatibility=(
            "ap02" in enabled and ap02_selection_mode != "manual"
        ),
        expected_camera_ids=camera_ids,
    )
    try:
        recommended_single_marker: int | None = _marker_choice(
            "auto",
            markers,
            compatibility_key="ap03_compatible",
            purpose="AP03 Single scale",
            rank=ap03_candidate_rank,
            require_compatibility="ap03" in enabled,
        )
    except RuntimeError:
        if "ap03" in enabled:
            raise
        recommended_single_marker = None
    configured_single_marker = (
        int(
            _best_candidate(
                markers,
                ap03_candidate_rank,
                compatibility_key="_all",
            )
        )
        if (
            config.methods.ap03_single.scale_marker_id == "auto"
            and recommended_single_marker is None
        )
        else config.methods.ap03_single.scale_marker_id
    )
    single_marker = _marker_choice(
        configured_single_marker,
        markers,
        compatibility_key="ap03_compatible",
        purpose="AP03 Single scale",
        rank=ap03_candidate_rank,
        require_compatibility="ap03" in enabled,
    )

    compatible_multi = tuple(
        sorted(
            marker
            for marker, details in markers.items()
            if details["ap03_compatible"]
        )
    )
    configured_multi = config.methods.ap03_multi.marker_ids
    multi_markers = (
        compatible_multi
        if configured_multi == "auto"
        else tuple(sorted(dict.fromkeys(int(value) for value in configured_multi)))
    )
    if "ap03" not in enabled and not multi_markers:
        multi_markers = tuple(sorted(markers))
    if "ap03" in enabled and not multi_markers:
        raise RuntimeError("AP03 Multi has no compatible moving-camera markers")
    incompatible_multi = [
        marker
        for marker in multi_markers
        if marker not in markers or not markers[marker]["ap03_compatible"]
    ]
    if "ap03" in enabled and incompatible_multi:
        raise RuntimeError(
            f"AP03 Multi markers are not compatible: {incompatible_multi}"
        )

    moving_markers = {
        _marker_id(row) for row in rows if row.get("observer_type") == "moving"
    }
    root_markers = {
        _marker_id(row)
        for row in rows
        if row.get("observer_type") == "static" and _observer_id(row) == root
    }
    evaluation_candidates = {
        marker: details
        for marker, details in markers.items()
        if marker in moving_markers
        and bool(details["static_cameras"])
        and (
            "ap01" not in enabled
            or marker in root_markers
        )
    }
    if "ap02" in enabled:
        ap02_component_markers = set(
            markers[ap02_reference][
                "combined_graph_reachable_marker_ids"
            ]
        )
        evaluation_candidates = {
            marker: details
            for marker, details in evaluation_candidates.items()
            if marker in ap02_component_markers
        }
    if "ap03" in enabled:
        evaluation_candidates = {
            marker: details
            for marker, details in evaluation_candidates.items()
            if details["ap03_compatible"]
        }
    configured_evaluation = config.evaluation.anchor_marker_id
    evaluation_anchor: int | None
    automatic_evaluation_candidates = {
        marker: details
        for marker, details in evaluation_candidates.items()
        if details.get("automatic_candidate", False)
    }
    recommended_evaluation_anchor = (
        int(
            _best_candidate(
                automatic_evaluation_candidates,
                ap03_candidate_rank,
                compatibility_key="_all",
            )
        )
        if automatic_evaluation_candidates
        else None
    )
    if not config.evaluation.enabled:
        evaluation_anchor = None
    elif configured_evaluation == "auto":
        if recommended_evaluation_anchor is None:
            if config.evaluation.anchor_selection_mode == "review_once":
                evaluation_anchor = None
            else:
                raise RuntimeError(
                    "Evaluation is enabled, but preflight found no common marker "
                    "with repeated accepted static/moving support for every "
                    "enabled method. Adjust quality filters/whitelist or disable "
                    "evaluation explicitly."
                )
        else:
            evaluation_anchor = recommended_evaluation_anchor
    else:
        evaluation_anchor = int(configured_evaluation)

    if config.selection.mode == "explicit":
        unresolved: list[str] = []
        if "ap01" in enabled and configured_root == "auto":
            unresolved.append("methods.ap01.root_camera")
        if (
            "ap02" in enabled
            and ap02_selection_mode in {"auto", "manual"}
            and config.methods.ap02.reference_marker_id == "auto"
        ):
            unresolved.append("methods.ap02.reference_marker_id")
        if (
            "ap03" in enabled
            and config.methods.ap03_single.scale_marker_id == "auto"
        ):
            unresolved.append("methods.ap03.single.scale_marker_id")
        if (
            "ap03" in enabled
            and config.methods.ap03_multi.marker_ids == "auto"
        ):
            unresolved.append("methods.ap03.multi.marker_ids")
        if unresolved:
            raise RuntimeError(
                "selection.mode=explicit requires values for: "
                + ", ".join(unresolved)
            )

    marker_ids = tuple(sorted(markers))
    root_payload = [
        {**details, "recommended": camera == recommended_root}
        for camera, details in sorted(roots.items())
    ]
    ap02_payload = [
        {
            **details,
            "compatible": (
                details["ap02_compatible"]
                or details["ap02_partial_compatible"]
            ),
            "diagnostic_partial": (
                not details["ap02_compatible"]
                and details["ap02_partial_compatible"]
            ),
            "recommended": marker == recommended_ap02_reference,
        }
        for marker, details in sorted(markers.items())
    ]
    ap03_payload = [
        {
            **details,
            "compatible": details["ap03_compatible"],
            "recommended": marker == recommended_single_marker,
        }
        for marker, details in sorted(markers.items())
    ]
    payload: dict[str, Any] = {
        "schema_version": 5,
        "selection_mode": config.selection.mode,
        "ap01_root_camera": {
            "configured": configured_root,
            "selected": root,
            "candidates": root_payload,
            "reason": (
                "explicit user configuration"
                if configured_root != "auto"
                else "lexicographic AP01 reachability, direct links, moving bridges, observation quality and stable camera ID"
            ),
        },
        "ap02_reference_marker": {
            "configured": config.methods.ap02.reference_marker_id,
            "selection_mode": ap02_selection_mode,
            "selected": ap02_reference,
            "candidates": ap02_payload,
            "reason": (
                "Route-2 simulation baseline contract: marker 14"
                if ap02_selection_mode == "baseline"
                else "manual post-preflight selection"
                if ap02_selection_mode == "manual"
                else "explicit compatibility configuration"
                if ap02_selection_mode == "explicit"
                else (
                    "deterministic recommendation from static-only reachability, "
                    "direct static coverage, moving-frame coverage, observation "
                    "count, median PnP RMSE and median marker area"
                )
            ),
            "evidence": (
                next(
                    (
                        item
                        for item in ap02_payload
                        if int(item["id"]) == int(ap02_reference)
                    ),
                    None,
                )
            ),
        },
        "ap03_single_scale_marker": {
            "configured": config.methods.ap03_single.scale_marker_id,
            "selected": single_marker,
            "candidates": ap03_payload,
            "reason": (
                "explicit user configuration"
                if config.methods.ap03_single.scale_marker_id != "auto"
                else (
                    "deterministic recommendation from moving-frame coverage, "
                    "direct static coverage, median moving PnP RMSE and median "
                    "moving marker area"
                )
            ),
        },
        "ap03_multi_marker_set": {
            "configured": configured_multi,
            "selected": list(multi_markers),
            "candidates": ap03_payload,
            "reason": (
                "all compatible detected moving-camera markers"
                if configured_multi == "auto"
                else "explicit user configuration"
            ),
        },
        "evaluation_anchor": {
            "configured": configured_evaluation,
            "selected": evaluation_anchor,
            "selection_mode": config.evaluation.anchor_selection_mode,
            "resolution_stage": "disabled" if not config.evaluation.enabled else "preflight",
            "observation_candidates": sorted(evaluation_candidates),
            "automatic_observation_candidates": sorted(
                marker
                for marker, details in evaluation_candidates.items()
                if details.get("automatic_candidate", False)
            ),
            "reason": (
                "evaluation disabled explicitly"
                if not config.evaluation.enabled
                else (
                    "deterministic common preflight recommendation from repeated "
                    "support, selection score, PnP RMSE, marker area ratio and "
                    "stable marker ID"
                    if configured_evaluation == "auto"
                    and config.evaluation.anchor_selection_mode == "auto"
                    else (
                        "manual selection requested after shared detection"
                        if config.evaluation.anchor_selection_mode == "review_once"
                        else "explicit user configuration"
                    )
                )
            ),
        },
        "automatic_recommendations": {
            "ap01_root_camera": recommended_root,
            "ap02_reference_marker_id": recommended_ap02_reference,
            "ap03_single_scale_marker_id": recommended_single_marker,
            "ap03_multi_marker_ids": list(compatible_multi),
            "evaluation_anchor_marker_id": (
                recommended_evaluation_anchor
            ),
        },
        "detected_marker_ids": list(marker_ids),
    }
    observations_root.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2) + "\n"
    (observations_root / "SELECTION_CANDIDATES.json").write_text(
        text, encoding="utf-8"
    )
    # Compatibility alias for existing result readers.
    (observations_root / "REFERENCE_SELECTIONS.json").write_text(
        text, encoding="utf-8"
    )
    (observations_root / "REFERENCE_MARKER_ID.txt").write_text(
        f"{ap02_reference}\n", encoding="utf-8"
    )
    write_selection_candidates_csv(observations_root, payload)
    return ResolvedSelections(
        root,
        ap02_reference,
        single_marker,
        multi_markers,
        evaluation_anchor,
        marker_ids,
        payload,
    )


