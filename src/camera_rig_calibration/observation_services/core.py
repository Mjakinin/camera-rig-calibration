from __future__ import annotations

import csv
import json
import math
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..config.models import DatasetCategory, RigConfig


def _success(row: dict[str, str]) -> bool:
    return str(row.get("pnp_success", "")).strip().lower() in {"true", "1", "yes"}


def _number(row: dict[str, str], *keys: str) -> float:
    for key in keys:
        try:
            value = row.get(key)
            if value not in {None, ""}:
                return float(value)
        except (TypeError, ValueError):
            continue
    return 0.0


def _marker_id(row: dict[str, str]) -> int:
    return int(float(row["marker_id"]))


def _observer_id(row: dict[str, str]) -> str:
    return str(row.get("observer_id") or row.get("camera_name") or "").strip()


def _median_value(
    rows: list[dict[str, str]], *keys: str
) -> float | None:
    values: list[float] = []
    for row in rows:
        for key in keys:
            raw = row.get(key)
            if raw in {None, ""}:
                continue
            try:
                value = float(raw)
            except (TypeError, ValueError):
                continue
            if math.isfinite(value):
                values.append(value)
                break
    return float(statistics.median(values)) if values else None


def _is_simulation_dataset(config: RigConfig) -> bool:
    """Resolve simulation identity from config or canonical prepared metadata.

    Selection previews may intentionally rebuild a lightweight prepared-data
    config.  The canonical dataset manifest remains authoritative when that
    preview omits the original category/scene metadata.
    """

    if config.dataset.category == DatasetCategory.SIMULATION:
        return True
    root = config.dataset.prepared_root
    if root is None:
        return False
    metadata = Path(root) / "dataset.json"
    try:
        payload = json.loads(metadata.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(payload, dict):
        return False
    return (
        str(payload.get("category", "")).strip().lower() == "simulation"
        or str(payload.get("scene_type", "")).strip().lower() == "simulation"
    )


@dataclass(frozen=True)
class ResolvedSelections:
    root_camera: str
    ap02_reference_marker_id: int
    ap03_single_scale_marker_id: int
    ap03_multi_marker_ids: tuple[int, ...]
    evaluation_anchor_marker_id: int | None
    marker_ids: tuple[int, ...]
    payload: dict[str, Any]


def write_selection_candidates_csv(
    observations_root: Path,
    payload: dict[str, Any],
) -> Path:
    """Write the auditable score table for every preflight selection."""

    root_candidates = payload["ap01_root_camera"]["candidates"]
    ap02_candidates = payload["ap02_reference_marker"]["candidates"]
    ap03_candidates = payload["ap03_single_scale_marker"]["candidates"]
    evaluation = payload["evaluation_anchor"]
    evaluation_ids = {
        int(value) for value in evaluation["observation_candidates"]
    }
    automatic_evaluation_ids = {
        int(value) for value in evaluation["automatic_observation_candidates"]
    }
    evaluation_candidates = [
        {
            **candidate,
            "compatible": int(candidate["id"]) in evaluation_ids,
            "automatic_candidate": (
                int(candidate["id"]) in automatic_evaluation_ids
            ),
            "recommended": (
                evaluation.get("selected") is not None
                and int(candidate["id"]) == int(evaluation["selected"])
            ),
        }
        for candidate in ap03_candidates
    ]
    selection_rows: list[dict[str, Any]] = []
    for selection_name, candidates in (
        ("ap01_root_camera", root_candidates),
        ("ap02_reference_marker", ap02_candidates),
        ("ap03_scale_marker", ap03_candidates),
        ("evaluation_anchor", evaluation_candidates),
    ):
        for candidate in candidates:
            selection_rows.append(
                {
                    "selection": selection_name,
                    "candidate_id": candidate["id"],
                    "compatible": candidate.get("compatible", False),
                    "automatic_candidate": candidate.get(
                        "automatic_candidate", True
                    ),
                    "recommended": candidate.get("recommended", False),
                    "selection_score": candidate.get(
                        "median_selection_score"
                    ),
                    "score_area": candidate.get("median_score_area"),
                    "score_reprojection": candidate.get(
                        "median_score_reprojection"
                    ),
                    "score_border": candidate.get("median_score_border"),
                    "score_distance": candidate.get(
                        "median_score_distance"
                    ),
                    "tie_breaker": (
                        "stable ascending camera ID"
                        if selection_name == "ap01_root_camera"
                        else "stable ascending marker ID"
                    ),
                }
            )
    destination = observations_root / "SELECTION_CANDIDATES.csv"
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                list(selection_rows[0])
                if selection_rows
                else [
                    "selection",
                    "candidate_id",
                    "compatible",
                    "automatic_candidate",
                    "recommended",
                    "selection_score",
                    "score_area",
                    "score_reprojection",
                    "score_border",
                    "score_distance",
                    "tie_breaker",
                ]
            ),
        )
        writer.writeheader()
        writer.writerows(selection_rows)
    return destination


def _read_observations(root: Path) -> list[dict[str, str]]:
    path = root / "shared_all_aruco_observations.csv"
    if not path.is_file():
        raise RuntimeError(f"Observation table is missing: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    successful = [row for row in rows if _success(row)]
    if not successful:
        raise RuntimeError(f"No successful marker observations in {path}")
    return successful


def _best_candidate(
    candidates: dict[str | int, dict[str, Any]],
    rank,
    *,
    compatibility_key: str = "compatible",
) -> str | int:
    pool = {
        key: value
        for key, value in candidates.items()
        if bool(value.get(compatibility_key, True))
    }
    if not pool:
        raise RuntimeError("No compatible selection candidates are available")
    best_rank = max(rank(value) for value in pool.values())
    # The stable ascending ID is deliberately the final tie-breaker.
    return sorted(
        (key for key, value in pool.items() if rank(value) == best_rank),
        key=lambda value: (str(type(value)), value),
    )[0]


def _lower_is_better(value: float | None) -> float:
    return -value if value is not None else float("-inf")


def _higher_is_better(value: float | None) -> float:
    return value if value is not None else float("-inf")


def _root_rank(candidate: dict[str, Any]) -> tuple[Any, ...]:
    return (
        len(candidate["reachable_cameras"]),
        len(candidate["direct_connections"]),
        len(candidate["moving_bridges"]),
        candidate["distinct_markers"],
        candidate["observations"],
        _higher_is_better(candidate["median_selection_score"]),
        _lower_is_better(candidate["median_pnp_reprojection_rmse_px"]),
        _higher_is_better(candidate["median_marker_area_ratio"]),
    )


def _ap02_rank(candidate: dict[str, Any]) -> tuple[Any, ...]:
    return (
        candidate["static_graph_reachable_count"],
        candidate["static_camera_count"],
        candidate["moving_frames"],
        candidate["accepted_observations"],
        _higher_is_better(candidate["median_selection_score"]),
        _lower_is_better(candidate["median_pnp_reprojection_rmse_px"]),
        _higher_is_better(candidate["median_marker_area_ratio"]),
    )


def ap03_candidate_rank(candidate: dict[str, Any]) -> tuple[Any, ...]:
    return (
        candidate["moving_frames"],
        candidate["static_camera_count"],
        _higher_is_better(candidate["moving_median_selection_score"]),
        _lower_is_better(candidate["moving_median_pnp_reprojection_rmse_px"]),
        _higher_is_better(candidate["moving_median_marker_area_ratio"]),
    )
