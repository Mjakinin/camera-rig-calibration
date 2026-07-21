#!/usr/bin/env python3
"""Resolution-aware entry point for AP01 COLMAP scale estimation."""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
LEGACY_PATH = HERE / "12_estimate_colmap_scale_from_aruco.py"
CAMERA_INFO = Path(
    "results/bus_real_data/00_shared_baseline/"
    "bus_real_data_ref_marker_v1/raw_images/camera_info/"
    "moving_calib_camera.json"
)

if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
spec = importlib.util.spec_from_file_location("ap01_scale_legacy", LEGACY_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Cannot load AP01 scale implementation: {LEGACY_PATH}")
legacy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(legacy)


def load_dimensions() -> tuple[float, float]:
    if not CAMERA_INFO.is_file():
        raise RuntimeError(f"Missing moving camera_info: {CAMERA_INFO}")
    data = json.loads(CAMERA_INFO.read_text())
    width = float(data.get("width", data.get("image_width", 0)) or 0)
    height = float(data.get("height", data.get("image_height", 0)) or 0)
    if width <= 0 or height <= 0:
        raise RuntimeError(f"Invalid image dimensions in {CAMERA_INFO}")
    return width, height


def main() -> None:
    width, height = load_dimensions()
    legacy.W = width
    legacy.H = height
    legacy.CX = width / 2.0
    legacy.CY = height / 2.0
    legacy.HALF_DIAG = math.hypot(legacy.CX, legacy.CY)
    print(f"[OK] AP01 quality normalization: {int(width)}x{int(height)}")
    legacy.main()


if __name__ == "__main__":
    main()
