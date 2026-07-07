#!/usr/bin/env python3
"""Resolution-aware entry point for AP01 direct static-camera estimation."""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
LEGACY_PATH = HERE / "13_eval_direct_static_cam3_cam1_multimarker.py"
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

spec = importlib.util.spec_from_file_location("ap01_direct_legacy", LEGACY_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Cannot load AP01 direct implementation: {LEGACY_PATH}")
legacy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(legacy)


def center_norm(row, width=None, height=None):
    if width is None:
        width = legacy.safe_float(row, "image_width", 1280.0)
    if height is None:
        height = legacy.safe_float(row, "image_height", 720.0)
    cu = legacy.safe_float(row, "center_u")
    cv = legacy.safe_float(row, "center_v")
    if not all(math.isfinite(value) for value in [width, height, cu, cv]):
        return float("nan")
    if width <= 0.0 or height <= 0.0:
        return float("nan")
    cx = width / 2.0
    cy = height / 2.0
    half_diag = math.hypot(cx, cy)
    return math.hypot(cu - cx, cv - cy) / max(half_diag, 1e-12)


legacy.center_norm = center_norm


if __name__ == "__main__":
    legacy.main()
