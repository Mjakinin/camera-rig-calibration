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

from .reporting_configuration import (
    _config_text,
)
from .reporting_core import (
    _fmt,
    _read_json,
    _text_table,
)
from .reporting_diagnostics import (
    _scale_comparison_rows,
    _scale_comparison_text,
)
from .reporting_simulation_geometry import (
    _latest_marker_report,
    _real_variant_disagreement,
)
from .reporting_bindings import current_reporting_bindings

def _real_results_text(
    experiment_root: Path,
    method_payloads: list[dict[str, Any]],
    dataset_root: Path | None = None,
) -> tuple[str, dict[str, Any]]:
    _read_json = current_reporting_bindings().read_json
    width = 138
    dataset_root = dataset_root or experiment_root
    dataset = _read_json(dataset_root / "dataset.json")
    selection = _read_json(
        dataset_root / "observations" / "SELECTION_CANDIDATES.json"
    )
    evaluation_anchor = selection.get("evaluation_anchor", {})
    lines = [
        "REAL-VEHICLE CAMERA-RIG CALIBRATION RESULTS",
        "=" * width,
        "",
        f"Experiment: {experiment_root.name}",
        f"Dataset: {dataset.get('id', dataset_root.name)}",
        (
            "Common evaluation anchor: marker "
            f"{evaluation_anchor.get('selected', '-')} "
            f"(configured {evaluation_anchor.get('configured', '-')}; "
            "frozen during preflight)"
        ),
        f"Anchor reason: {evaluation_anchor.get('reason', '-')}",
        "",
    ]
    marker_text, marker_path = _latest_marker_report(experiment_root)
    lines.extend(
        [
            marker_text,
            "",
            "METHOD / VARIANT OVERVIEW",
            "-" * width,
        ]
    )
    overview: list[list[str]] = []
    all_pairs: list[dict[str, Any]] = []
    for payload in method_payloads:
        root = (
            experiment_root
            / "methods"
            / str(payload["method"])
            / str(payload["label"])
        )
        pairs_path = root / "pairwise_camera_extrinsics.csv"
        if pairs_path.is_file():
            with pairs_path.open(newline="", encoding="utf-8") as handle:
                all_pairs.extend(list(csv.DictReader(handle)))
        quality_text = str(payload.get("quality_status", "-"))
        graph = payload.get("metrics", {}).get(
            "ap02_combined_graph", {}
        )
        if payload.get("method") == "ap02" and graph and not graph.get(
            "complete", True
        ):
            quality_text = (
                "partial — primary "
                f"{graph.get('reached_static_camera_count', 0)}/"
                f"{graph.get('expected_static_camera_count', 0)}, "
                f"{graph.get('component_count', 0)} graph components"
            )
        overview.append(
            [
                payload["method"],
                payload["label"],
                payload.get("artifact_status", "available"),
                quality_text,
                payload.get("static_camera_count", 0),
                (
                    _fmt(payload.get("runtime_seconds"), 1) + " s"
                    if payload.get("runtime_seconds") is not None
                    else "-"
                ),
                _config_text(payload.get("config_summary", {})),
            ]
        )
    lines.extend(
        [
            _text_table(
                [
                    "Method",
                    "Variant",
                    "Artifact",
                    "Quality",
                    "Cameras",
                    "Runtime",
                    "Key configuration",
                ],
                overview,
            ),
            "",
            "SCALE COMPARISON",
            "-" * width,
            _scale_comparison_text(
                _scale_comparison_rows(method_payloads)
            ),
            "",
        ]
    )
    for payload in method_payloads:
        root = (
            experiment_root
            / "methods"
            / str(payload["method"])
            / str(payload["label"])
        )
        lines.extend(
            [
                f"{payload['method']} / {payload['label']}",
                "-" * width,
                (root / "RESULT.txt").read_text(encoding="utf-8"),
                "",
            ]
        )
    disagreement_summaries, disagreement_rows = (
        _real_variant_disagreement(all_pairs)
    )
    lines.extend(
        [
            "DIRECT VARIANT-TO-VARIANT DISAGREEMENT",
            "-" * width,
            _text_table(
                [
                    "First",
                    "Second",
                    "Pairs",
                    "mean t delta [cm]",
                    "max t delta [cm]",
                    "mean r delta [deg]",
                    "max r delta [deg]",
                    "mean baseline delta [cm]",
                ],
                [
                    [
                        f"{row['first_method']}/{row['first_label']}",
                        f"{row['second_method']}/{row['second_label']}",
                        row["pair_count"],
                        _fmt(row["mean_translation_delta_cm"]),
                        _fmt(row["max_translation_delta_cm"]),
                        _fmt(row["mean_rotation_delta_deg"]),
                        _fmt(row["max_rotation_delta_deg"]),
                        _fmt(row["mean_baseline_delta_cm"]),
                    ]
                    for row in disagreement_summaries
                ],
            ),
            "",
        ]
    )
    payload = {
        "category": "real_vehicle",
        "experiment": experiment_root.name,
        "methods": method_payloads,
        "pairwise_camera_extrinsics": all_pairs,
        "variant_disagreement_summary": disagreement_summaries,
        "variant_disagreement_rows": disagreement_rows,
        "marker_consistency_path": (
            str(marker_path.relative_to(experiment_root))
            if marker_path is not None
            else None
        ),
        "evaluation_anchor": evaluation_anchor,
        "scale_comparison": _scale_comparison_rows(method_payloads),
    }
    return "\n".join(lines), payload



__all__ = [
    '_real_results_text',
]
