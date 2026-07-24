from __future__ import annotations

import csv
from pathlib import Path

from camera_rig_calibration.pipeline.artifacts import require_file


def read_observations(path: Path) -> list[dict[str, str]]:
    source = require_file(path, label="accepted observation table")
    with source.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict], fields: list[str] | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return path
