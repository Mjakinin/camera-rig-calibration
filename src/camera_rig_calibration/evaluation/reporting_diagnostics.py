"""Focused scientific reporting responsibility."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml

from ..anchor_export import ensure_experiment_anchor_exports
from ..anchor_export.geometry import rotation_to_quaternion
from ..visualization.scene import ensure_visualization_artifacts
from .ap03_derived import ensure_ap03_derived_results
from .simulation_ground_truth import (
    ensure_simulation_ground_truth,
    resolve_simulation_ground_truth,
)

from ..methods.common.geometry import (
    R_to_rpy_deg,
    R_to_rvec,
    invT,
    make_T,
    rot_error_deg,
    rpy_to_R,
    rvec_to_R,
)

from .reporting_core import (
    _fmt,
    _read_json,
    _text_table,
)
from .reporting_bindings import current_reporting_bindings

def _method_diagnostics(
    result_root: Path, method: str
) -> tuple[dict[str, Any], list[str]]:
    _read_json = current_reporting_bindings().read_json
    method_root = result_root / "diagnostics" / "method"
    diagnostics: dict[str, Any] = {}
    paths: list[str] = []
    candidates: tuple[tuple[str, Path], ...]
    if method == "ap01":
        candidates = (
            (
                "ap01_scale",
                method_root / "metric_scale" / "SCALE_DIAGNOSTICS.json",
            ),
            (
                "ap01_relay_selection",
                method_root / "candidates" / "AP01_RELAY_SELECTION.json",
            ),
        )
    elif method == "ap02":
        candidates = (
            (
                "ap02_frame_selection",
                method_root
                / "aruco_observations"
                / "ap02_frame_selection.json",
            ),
            (
                "ap02_combined_optimization",
                method_root
                / "graph_ba"
                / "with_moving"
                / "ap02_optimization_summary.json",
            ),
        )
    elif method == "ap03":
        candidates = (
            (
                "ap03_scale",
                method_root
                / "scale_multi"
                / "AP03_MARKER_SIZE_SCALE_ONLY_METADATA.json",
            ),
        )
    elif method in {"ap03_single", "ap03_multi"}:
        provenance = _read_json(
            result_root / "provenance" / "derived_result.json"
        )
        experiment_root = result_root.parents[2]
        metadata = provenance.get("scale_metadata")
        candidates = (
            (
                "ap03_scale",
                experiment_root / str(metadata)
                if metadata
                else Path("__missing__"),
            ),
        )
        if provenance:
            diagnostics["shared_colmap"] = provenance
            paths.append("provenance/derived_result.json")
            shared = provenance.get("shared_colmap_container")
            if shared:
                reconstruction = (
                    experiment_root
                    / str(shared)
                    / "diagnostics"
                    / "method"
                    / "colmap"
                    / "inspection"
                    / "AP03_RECONSTRUCTION_DIAGNOSTICS.json"
                )
                if reconstruction.is_file():
                    diagnostics["ap03_reconstruction"] = _read_json(
                        reconstruction
                    )
                    paths.append(
                        str(
                            reconstruction.relative_to(experiment_root)
                        )
                    )
    else:
        candidates = ()
    for key, path in candidates:
        if not path.is_file():
            continue
        value = json.loads(path.read_text(encoding="utf-8"))
        diagnostics[key] = value
        try:
            relative = path.relative_to(result_root)
        except ValueError:
            relative = (
                Path("../../..")
                / path.relative_to(result_root.parents[2])
            )
        paths.append(relative.as_posix())
    for path in (
        method_root.rglob("*optimization_history.csv")
        if method == "ap02"
        else ()
    ):
        paths.append(str(path.relative_to(result_root)))
    return diagnostics, sorted(set(paths))


def _scale_comparison_rows(
    method_payloads: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Summarize method scale mechanisms without treating unlike values alike."""
    rows: list[dict[str, Any]] = []
    for payload in method_payloads:
        method = str(payload.get("method", ""))
        metrics = payload.get("metrics", {})
        if method == "ap01":
            scale = metrics.get("ap01_scale", {})
            rows.append(
                {
                    "method": method,
                    "label": payload.get("label", "-"),
                    "mechanism": "marker-motion pair scale",
                    "scale_m_per_colmap_unit": scale.get(
                        "scale_m_per_colmap_unit"
                    ),
                    "used": scale.get("used_pairs"),
                    "total": scale.get("raw_pairs"),
                    "relative_std": scale.get("used_relative_std"),
                }
            )
        elif method == "ap02":
            rows.append(
                {
                    "method": method,
                    "label": payload.get("label", "-"),
                    "mechanism": "metric marker-graph BA",
                    "scale_m_per_colmap_unit": None,
                    "used": None,
                    "total": None,
                    "relative_std": None,
                }
            )
        elif method in {"ap03", "ap03_single", "ap03_multi"}:
            scale = metrics.get("ap03_scale", {})
            rows.append(
                {
                    "method": method,
                    "label": payload.get("label", "-"),
                    "mechanism": "marker-corner RANSAC scale",
                    "scale_m_per_colmap_unit": scale.get(
                        "scale_m_per_colmap_unit"
                    ),
                    "used": scale.get("num_scale_observations_used"),
                    "total": scale.get("num_scale_observations_total"),
                    "relative_std": scale.get("used_rel_std_scale"),
                }
            )
    return rows


def _scale_comparison_text(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "No successful method scale result is available."
    return _text_table(
        [
            "Method",
            "Variant",
            "Scale mechanism",
            "Scale [m/COLMAP unit]",
            "Used/total",
            "Relative std",
        ],
        [
            [
                row["method"],
                row["label"],
                row["mechanism"],
                (
                    _fmt(row["scale_m_per_colmap_unit"], 6)
                    if row["scale_m_per_colmap_unit"] is not None
                    else "n/a (already metric)"
                ),
                (
                    f"{row['used']}/{row['total']}"
                    if row["used"] is not None
                    and row["total"] is not None
                    else "-"
                ),
                (
                    _fmt(row["relative_std"], 6)
                    if row["relative_std"] is not None
                    else "-"
                ),
            ]
            for row in rows
        ],
    )



__all__ = [
    '_method_diagnostics',
    '_scale_comparison_rows',
    '_scale_comparison_text',
]
