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



from .core import ResolvedSelections
def freeze_selections(
    config: RigConfig,
    resolved: ResolvedSelections,
    overrides: dict[str, Any] | None = None,
) -> RigConfig:
    """Return a prompt-free config with every pre-method decision explicit."""
    values = dict(overrides or {})
    root = str(values.get("root_camera", resolved.root_camera))
    ap02_marker = int(
        values.get(
            "ap02_reference_marker_id", resolved.ap02_reference_marker_id
        )
    )
    single_marker = int(
        values.get(
            "ap03_single_scale_marker_id",
            resolved.ap03_single_scale_marker_id,
        )
    )
    multi_value = values.get(
        "ap03_multi_marker_ids", resolved.ap03_multi_marker_ids
    )
    multi_markers = tuple(sorted(dict.fromkeys(int(item) for item in multi_value)))
    evaluation_anchor = values.get(
        "evaluation_anchor_marker_id",
        resolved.evaluation_anchor_marker_id,
    )

    available_roots = {
        str(item["id"])
        for item in resolved.payload["ap01_root_camera"]["candidates"]
        if item.get("compatible", True)
    }
    if root not in available_roots:
        raise ValueError(f"AP01 root camera is not compatible: {root}")
    marker_details = {
        int(item["id"]): item
        for item in resolved.payload["ap03_single_scale_marker"]["candidates"]
    }
    if (
        "ap02" in config.methods.enabled
        and (
            ap02_marker not in marker_details
            or not (
                marker_details[ap02_marker].get("ap02_compatible", False)
                or marker_details[ap02_marker].get(
                    "ap02_partial_compatible", False
                )
                or (
                    config.methods.ap02.reference_marker_selection_mode
                    == "manual"
                    and ap02_marker in marker_details
                )
            )
        )
    ):
        raise ValueError(f"AP02 reference marker is not compatible: {ap02_marker}")
    if (
        "ap03" in config.methods.enabled
        and single_marker not in marker_details
    ):
        raise ValueError(
            f"AP03 Single scale marker was not detected: {single_marker}"
        )
    invalid_multi = [
        marker
        for marker in multi_markers
        if marker not in marker_details
        or not marker_details[marker].get("ap03_compatible", False)
    ]
    if "ap03" in config.methods.enabled and invalid_multi:
        raise ValueError(f"AP03 Multi markers are not compatible: {invalid_multi}")
    raw_anchor_ids = {
        int(item["id"])
        for item in resolved.payload.get("raw_marker_inventory", [])
        if isinstance(item, dict) and item.get("id") is not None
    }
    if not raw_anchor_ids:
        raw_anchor_ids = {
            int(value)
            for value in resolved.payload.get("detected_marker_ids", [])
        }
    if (
        config.evaluation.enabled
        and evaluation_anchor is not None
        and int(evaluation_anchor) not in raw_anchor_ids
    ):
        raise ValueError(
            "Common evaluation/export anchor was not detected in the shared "
            f"preflight: marker {evaluation_anchor}"
        )

    ap03 = config.methods.ap03.model_copy(
        update={
            "single": config.methods.ap03.single.model_copy(
                update={"scale_marker_id": single_marker}
            ),
            "multi": config.methods.ap03.multi.model_copy(
                update={"marker_ids": list(multi_markers)}
            ),
        },
        deep=True,
    )
    methods = config.methods.model_copy(
        update={
            "ap01": config.methods.ap01.model_copy(
                update={"root_camera": root}
            ),
            "ap02": config.methods.ap02.model_copy(
                update={
                    "reference_marker_id": ap02_marker,
                    "reference_marker_selection_mode": (
                        config.methods.ap02.reference_marker_selection_mode
                    ),
                }
            ),
            "ap03": ap03,
        },
        deep=True,
    )
    evaluation = config.evaluation.model_copy(
        update={
            "anchor_marker_id": (
                int(evaluation_anchor)
                if evaluation_anchor is not None
                else config.evaluation.anchor_marker_id
            ),
            "anchor_selection_mode": (
                "explicit"
                if evaluation_anchor is not None
                else config.evaluation.anchor_selection_mode
            ),
        }
    )
    return RigConfig.model_validate(
        config.model_copy(
            update={
                "selection": config.selection.model_copy(
                    update={"mode": "explicit"}
                ),
                "methods": methods,
                "evaluation": evaluation,
            },
            deep=True,
        ).model_dump(mode="python")
    )
