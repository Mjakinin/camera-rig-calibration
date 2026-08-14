"""AP01 scientific core.

The functions in this module preserve the established marker-direct and
moving-COLMAP-relay mathematics.  The v4 stage modules import these functions
directly; no path mutation or simulated command-line invocation is required.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
import subprocess
import sys
import time
import traceback
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import cv2
import numpy as np

from .contracts import AP01MethodContract, resolve_ap01_method_contract


CAMERAS = [
    "cam_edge_0",
    "cam_edge_1",
    "cam_edge_3",
    "cam_edge_5",
]
ROOT_CAMERA = "cam_edge_3"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run AP01 marker-direct / COLMAP-relay on real data without GT."
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--observations-root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--matcher", choices=["exhaustive", "sequential"], default="exhaustive")
    parser.add_argument("--use-gpu", type=int, choices=[0, 1], default=0)
    parser.add_argument("--max-image-size", type=int, default=2400)
    parser.add_argument("--reuse-colmap", action="store_true")
    parser.add_argument("--marker-length-m", type=float, default=0.17)
    parser.add_argument("--cameras", default=",".join(CAMERAS))
    parser.add_argument("--root-camera", default=ROOT_CAMERA)
    parser.add_argument("--moving-camera-id", default="moving_calib_camera")
    parser.add_argument("--colmap-executable", default="colmap")
    parser.add_argument("--max-features", type=int, default=8192)
    parser.add_argument("--sequential-overlap", type=int, default=20)
    parser.add_argument("--loop-detection", type=int, choices=[0, 1], default=1)
    parser.add_argument("--mapper-min-matches", type=int, default=8)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise RuntimeError(f"Missing CSV: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = []
        for row in rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def status_path(out: Path) -> Path:
    return out / "METHOD_STATUS.json"


def write_status(out: Path, payload: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    status_path(out).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def is_success(row: dict[str, str]) -> bool:
    return str(row.get("pnp_success", "")).strip().lower() in {"true", "1", "yes"}


def safe_float(row: dict[str, str], key: str, default: float = float("nan")) -> float:
    try:
        value = float(row.get(key, ""))
        return value if math.isfinite(value) else default
    except Exception:
        return default


def frame_number(row: dict[str, str]) -> int:
    for key in ("frame_id", "observer_id", "image_path"):
        value = str(row.get(key, ""))
        matches = re.findall(r"(\d+)", value)
        if matches:
            return int(matches[-1])
    raise RuntimeError(f"Cannot infer moving-frame number from row: {row}")


def load_camera_info(path: Path) -> dict:
    data = json.loads(path.read_text())
    flat = data.get("K", data.get("k"))
    if flat is None:
        flat = [
            float(data["fx"]), 0.0, float(data["cx"]),
            0.0, float(data.get("fy", data["fx"])), float(data["cy"]),
            0.0, 0.0, 1.0,
        ]
    K = np.asarray(flat, dtype=np.float64).reshape(3, 3)
    D = np.asarray(data.get("D", data.get("d", [])), dtype=np.float64).reshape(-1)
    return {
        "K": K,
        "D": D,
        "width": int(data.get("width", data.get("image_width", 0)) or 0),
        "height": int(data.get("height", data.get("image_height", 0)) or 0),
        "distortion_model": str(data.get("distortion_model", "plumb_bob")),
        "source": str(path),
    }
