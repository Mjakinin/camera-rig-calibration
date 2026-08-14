from __future__ import annotations

import copy
import csv
import json
import math
import statistics
from dataclasses import replace
from pathlib import Path
from typing import Any

from ..config.models import RigConfig


_INSTALLED = False


def _finite(row: dict[str, str], *keys: str, default: float = 0.0) -> float:
    for key in keys:
        raw = row.get(key)
        if raw in {None, ""}:
            continue
        try:
            value = float(raw)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            return value
    return default


def _camera_id(row: dict[str, str]) -> str:
    return str(row.get("observer_id") or row.get("camera_name") or "").strip()


def _row_rank(row: dict[str, str]) -> tuple[float, float, float]:
    """Rank one already-filtered static observation without any GT input."""

    score = _finite(row, "selection_score", default=0.0)
    rmse = _finite(
        row,
        "pnp_reprojection_rmse_px",
        "reprojection_rmse_px",
        "reprojection_error_px",
        default=float("inf"),
    )
    area = _finite(
        row,
        "marker_area_ratio",
        "area_px2",
        default=0.0,
    )
    return (-score, rmse, -area)


def _automatic_ap01_direct_target(
    config: RigConfig,
    observations_root: Path,
    root_camera: str,
) -> tuple[str | None, list[dict[str, Any]]]:
    """Choose AP01's one Direct target from filtered static overlap evidence.

    The AP01 baseline has exactly one Direct branch and uses Relay for
    every other static camera.  The product must therefore determine that branch
    from the data instead of asking the operator for a camera name.  Selection is
    GT-free and happens after the common observation-quality filter: at least two
    independent shared markers are required, then larger shared-marker support,
    lower PnP reprojection RMSE and higher observation quality win.  If no target
    has enough direct support, AP01 runs Relay-only.
    """

    path = observations_root / "shared_all_aruco_observations.csv"
    if not path.is_file():
        raise RuntimeError(
            "AP01 automatic Direct-target selection requires the filtered "
            f"observation table: {path}"
        )
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    camera_ids = tuple(camera.id for camera in config.static_cameras)
    best: dict[tuple[str, int], dict[str, str]] = {}
    for row in rows:
        if str(row.get("observer_type", "")).strip().lower() != "static":
            continue
        camera = _camera_id(row)
        if camera not in camera_ids:
            continue
        try:
            marker = int(float(row["marker_id"]))
        except (KeyError, TypeError, ValueError):
            continue
        key = (camera, marker)
        if key not in best or _row_rank(row) < _row_rank(best[key]):
            best[key] = row

    root_markers = {
        marker for camera, marker in best if camera == root_camera
    }
    candidates: list[dict[str, Any]] = []
    for target in camera_ids:
        if target == root_camera:
            continue
        target_markers = {
            marker for camera, marker in best if camera == target
        }
        shared = sorted(root_markers & target_markers)
        pair_rmse: list[float] = []
        pair_score: list[float] = []
        pair_area: list[float] = []
        for marker in shared:
            first = best[(root_camera, marker)]
            second = best[(target, marker)]
            pair_rmse.append(
                max(
                    _finite(
                        first,
                        "pnp_reprojection_rmse_px",
                        "reprojection_rmse_px",
                        "reprojection_error_px",
                        default=float("inf"),
                    ),
                    _finite(
                        second,
                        "pnp_reprojection_rmse_px",
                        "reprojection_rmse_px",
                        "reprojection_error_px",
                        default=float("inf"),
                    ),
                )
            )
            pair_score.append(
                min(
                    _finite(first, "selection_score", default=0.0),
                    _finite(second, "selection_score", default=0.0),
                )
            )
            pair_area.append(
                min(
                    _finite(
                        first,
                        "marker_area_ratio",
                        "area_px2",
                        default=0.0,
                    ),
                    _finite(
                        second,
                        "marker_area_ratio",
                        "area_px2",
                        default=0.0,
                    ),
                )
            )
        median_rmse = (
            float(statistics.median(pair_rmse))
            if pair_rmse
            else float("inf")
        )
        median_score = (
            float(statistics.median(pair_score)) if pair_score else 0.0
        )
        median_area = (
            float(statistics.median(pair_area)) if pair_area else 0.0
        )
        compatible = len(shared) >= 2 and math.isfinite(median_rmse)
        candidates.append(
            {
                "id": target,
                "compatible": compatible,
                "shared_marker_ids": shared,
                "independent_shared_markers": len(shared),
                "median_pair_pnp_reprojection_rmse_px": (
                    median_rmse if math.isfinite(median_rmse) else None
                ),
                "median_pair_selection_score": median_score,
                "median_pair_marker_area": median_area,
            }
        )

    compatible = [item for item in candidates if item["compatible"]]
    compatible.sort(
        key=lambda item: (
            -int(item["independent_shared_markers"]),
            float(item["median_pair_pnp_reprojection_rmse_px"]),
            -float(item["median_pair_selection_score"]),
            -float(item["median_pair_marker_area"]),
            str(item["id"]),
        )
    )
    selected = str(compatible[0]["id"]) if compatible else None
    for item in candidates:
        item["recommended"] = item["id"] == selected
    return selected, candidates


def _install_ap01_direct_target_policy() -> None:
    from .. import observations

    original_resolve = observations.resolve_selections
    original_freeze = observations.freeze_selections
    if getattr(original_resolve, "_rigcal_submission_policy", False):
        return

    def resolve_selections(*args, **kwargs):
        resolved = original_resolve(*args, **kwargs)
        config = kwargs.get("config")
        if config is None and args:
            config = args[0]
        observations_root = kwargs.get("observations_root")
        if observations_root is None and len(args) >= 2:
            observations_root = args[1]
        if (
            config is None
            or observations_root is None
            or "ap01" not in set(config.methods.enabled)
            or config.methods.ap01.direct_target_camera != "auto"
        ):
            return resolved

        selected, candidates = _automatic_ap01_direct_target(
            config,
            Path(observations_root),
            resolved.root_camera,
        )
        payload = copy.deepcopy(resolved.payload)
        payload["ap01_direct_target_camera"] = {
            "configured": "auto",
            "selected": selected,
            "candidates": candidates,
            "reason": (
                "deterministic filtered static-overlap selection: require at "
                "least two independent shared markers, then maximize shared "
                "support, minimize PnP reprojection RMSE, maximize observation "
                "quality, and use stable camera ID as the final tie-breaker; "
                "no compatible target means Relay-only"
            ),
        }
        payload.setdefault("automatic_recommendations", {})[
            "ap01_direct_target_camera"
        ] = selected
        updated = replace(resolved, payload=payload)

        root = Path(observations_root)
        text = json.dumps(payload, indent=2) + "\n"
        for name in ("SELECTION_CANDIDATES.json", "REFERENCE_SELECTIONS.json"):
            destination = root / name
            if destination.parent.is_dir():
                destination.write_text(text, encoding="utf-8")
        (root / "AP01_DIRECT_TARGET_SELECTION.json").write_text(
            json.dumps(payload["ap01_direct_target_camera"], indent=2) + "\n",
            encoding="utf-8",
        )
        return updated

    def freeze_selections(config, resolved, overrides=None):
        frozen = original_freeze(config, resolved, overrides)
        if (
            "ap01" not in set(config.methods.enabled)
            or config.methods.ap01.direct_target_camera != "auto"
        ):
            return frozen
        selected = (
            resolved.payload.get("ap01_direct_target_camera", {})
            .get("selected")
        )
        direct_target = str(selected) if selected else "relay_only"
        ap01 = frozen.methods.ap01.model_copy(
            update={"direct_target_camera": direct_target}
        )
        methods = frozen.methods.model_copy(
            update={"ap01": ap01}, deep=True
        )
        return RigConfig.model_validate(
            frozen.model_copy(update={"methods": methods}, deep=True).model_dump(
                mode="python"
            )
        )

    resolve_selections._rigcal_submission_policy = True  # type: ignore[attr-defined]
    freeze_selections._rigcal_submission_policy = True  # type: ignore[attr-defined]
    observations.resolve_selections = resolve_selections
    observations.freeze_selections = freeze_selections


def _install_wizard_submission_policy() -> None:
    from .. import wizard

    if getattr(wizard, "_SUBMISSION_POLICY_INSTALLED", False):
        return

    wizard._PUBLIC_POLICY_NAMES.update(
        {
            "smart_v1": "baseline: BA-boundary frame limits",
            "wizard_graph_preserving_v1": "advanced: graph-preserving preselection",
            "maximum_frontier_v1": "baseline: maximum-frontier tree",
            "wizard_maximum_bottleneck_v2": "advanced: path-aware bottleneck tree",
            "unweighted_bfs_diagnostic": "diagnostic: unweighted BFS tree",
            "geometric_observation_quality_v1": "baseline: geometric observation quality",
            "wizard_selection_score_v2": "advanced: shared selection score",
            "pinhole_v1": "baseline: pinhole reprojection",
            "distortion_aware_v1": "advanced: distortion-aware reprojection",
            "colmap_defaults_v1": "baseline: COLMAP native feature limits",
            "wizard_explicit_limits_v1": "advanced: explicit SIFT limits",
            "registered_image_redetection_v1": "baseline: all registered-image detections",
            "wizard_filtered_observations_v1": "advanced: preflight-filtered detections",
        }
    )

    original_setting_rows = wizard._setting_rows

    def setting_rows(job, groups=None):
        rows = original_setting_rows(job, groups)
        rendered = []
        for key, group, label, current, baseline, description in rows:
            if key == "ap01_direct_target":
                # Direct-vs-Relay is a method decision from filtered evidence,
                # not an operator-selected camera parameter.
                continue
            if key == "ap02_frame_strategy":
                label = "Algorithm variant — moving-frame selection"
                description = (
                    "Baseline applies the explicit per-marker and marker-pair "
                    "limits at the BA boundary. The alternative selects a "
                    "graph-preserving frame subset before initialization and is "
                    "intended only for an explicit ablation."
                )
            elif key == "ap02_initialization_strategy":
                label = "Algorithm variant — graph initialization"
                description = (
                    "Baseline uses the deterministic maximum-frontier tree. "
                    "The path-aware tree and unweighted BFS change the actual "
                    "initialization algorithm and are ablation/diagnostic options."
                )
            elif key == "ap02_edge_weight_strategy":
                label = "Algorithm variant — graph edge score"
                description = (
                    "Baseline weights graph edges by geometric observation "
                    "quality. The shared selection score is a different ranking "
                    "rule for an explicit ablation."
                )
            elif key == "ap02_reprojection_model":
                label = "Algorithm variant — BA reprojection model"
                description = (
                    "Baseline uses zero-distortion pinhole reprojection. The "
                    "distortion-aware alternative changes the BA residual model."
                )
            elif key == "ap03_feature_limit_policy":
                label = "Algorithm variant — COLMAP feature limits"
                description = (
                    "Baseline leaves COLMAP SIFT limits unset. The explicit "
                    "alternative sends the AP03 image-size/feature limits to "
                    "COLMAP and therefore changes reconstruction behavior."
                )
            elif key == "ap03_scale_input_policy":
                label = "Algorithm variant — metric-scale observations"
                description = (
                    "Baseline re-detects scale markers in every registered image. "
                    "The filtered alternative intersects those detections with "
                    "accepted preflight observations and optionally caps views "
                    "per marker."
                )
            rendered.append(
                (key, group, label, current, baseline, description)
            )
        return rendered

    wizard._setting_rows = setting_rows

    original_method_job_label = wizard._method_job_label

    def method_job_label(job, context_key=None):
        if job.method_id != "ap01":
            return original_method_job_label(job, context_key)
        normalized = copy.deepcopy(job)
        normalized.methods = normalized.methods.model_copy(
            update={
                "ap01": normalized.methods.ap01.model_copy(
                    update={"direct_target_camera": "cam_edge_1"}
                )
            },
            deep=True,
        )
        for key, methods in tuple(normalized.context_methods.items()):
            normalized.context_methods[key] = methods.model_copy(
                update={
                    "ap01": methods.ap01.model_copy(
                        update={"direct_target_camera": "cam_edge_1"}
                    )
                },
                deep=True,
            )
        return original_method_job_label(normalized, context_key)

    wizard._method_job_label = method_job_label

    original_method_job_summary = wizard._method_job_summary

    def method_job_summary(job):
        text = original_method_job_summary(job)
        if job.method_id == "ap01" and job.methods.ap01.direct_target_camera == "auto":
            text = text.replace(
                "direct=auto",
                "direct=automatic from filtered static overlap",
            )
        return text

    wizard._method_job_summary = method_job_summary

    original_new_method_job = wizard._new_method_job

    def new_method_job(*args, **kwargs):
        job = original_new_method_job(*args, **kwargs)
        if job.method_id == "ap01":
            ap01 = job.methods.ap01.model_copy(
                update={"direct_target_camera": "auto"}
            )
            job.methods = job.methods.model_copy(
                update={"ap01": ap01}, deep=True
            )
            wizard._refresh_method_job_label(job)
        return job

    wizard._new_method_job = new_method_job
    wizard._SUBMISSION_POLICY_INSTALLED = True


def install_submission_policy() -> None:
    """Install submission-facing AP01 automation and explicit UI semantics."""

    global _INSTALLED
    if _INSTALLED:
        return
    _install_ap01_direct_target_policy()
    _install_wizard_submission_policy()
    _INSTALLED = True
