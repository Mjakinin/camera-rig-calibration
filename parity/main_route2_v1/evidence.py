"""Writers for explicit unavailable comparison evidence."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


JSON_RESERVATIONS = (
    "OBSERVATION_PARITY.json",
    "AP01_CANDIDATE_PARITY.json",
    "AP01_SELECTION_PARITY.json",
    "AP02_FRAME_SELECTION_PARITY.json",
    "AP02_GRAPH_PARITY.json",
    "AP02_PARAMETER_PARITY.json",
    "AP02_INITIAL_RESIDUAL_PARITY.json",
    "AP03_COLMAP_CONFIG_PARITY.json",
)

CSV_RESERVATIONS = (
    "OBSERVATION_ROW_DIFF.csv",
    "AP01_POSE_PARITY.csv",
    "AP02_INITIAL_POSE_PARITY.csv",
    "AP02_FINAL_POSE_PARITY.csv",
    "AP03_REGISTERED_IMAGES_PARITY.csv",
)


def unavailable(reason: str, *, mode: str | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": 1,
        "status": "unavailable",
        "reason": reason,
        "ground_truth_used": False,
    }
    if mode is not None:
        payload["mode"] = mode
    return payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in rows)


def reserve_unavailable_artifacts(output: Path, reason: str) -> None:
    for name in JSON_RESERVATIONS:
        path = output / name
        if not path.exists():
            write_json(path, unavailable(reason))
    for name in CSV_RESERVATIONS:
        path = output / name
        if not path.exists():
            write_csv(
                path,
                [{"status": "unavailable", "reason": reason}],
                ["status", "reason"],
            )

