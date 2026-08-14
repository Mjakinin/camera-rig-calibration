#!/usr/bin/env python3
"""Calibrate a managed camera-intrinsics profile from video or images."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import time
from pathlib import Path

import cv2
import numpy as np

try:
    from camera_rig_calibration.input.video_geometry import open_oriented_video
except ImportError:  # pragma: no cover - direct script fallback
    from video_geometry import open_oriented_video


def write_contact_sheet(
    debug_paths: list[Path],
    destination: Path,
) -> None:
    if not debug_paths:
        return

    sample_count = min(20, len(debug_paths))

    positions = np.linspace(
        0,
        len(debug_paths) - 1,
        sample_count,
        dtype=int,
    )

    thumbs = []

    for position in positions:
        image = cv2.imread(str(debug_paths[position]))

        if image is None:
            continue

        source_height, source_width = image.shape[:2]
        scale = min(480.0 / source_width, 480.0 / source_height)
        target_width = max(1, int(round(source_width * scale)))
        target_height = max(1, int(round(source_height * scale)))
        resized = cv2.resize(
            image,
            (target_width, target_height),
            interpolation=cv2.INTER_AREA,
        )
        thumbnail = np.zeros((480, 480, 3), dtype=np.uint8)
        x0 = (480 - target_width) // 2
        y0 = (480 - target_height) // 2
        thumbnail[y0:y0 + target_height, x0:x0 + target_width] = resized
        thumbs.append(thumbnail)

    if not thumbs:
        return

    columns = 4
    rows = math.ceil(len(thumbs) / columns)

    blank = np.zeros_like(thumbs[0])

    while len(thumbs) < rows * columns:
        thumbs.append(blank.copy())

    row_images = []

    for row in range(rows):
        start = row * columns
        row_images.append(
            np.hstack(
                thumbs[start:start + columns]
            )
        )

    sheet = np.vstack(row_images)
    cv2.imwrite(str(destination), sheet)


