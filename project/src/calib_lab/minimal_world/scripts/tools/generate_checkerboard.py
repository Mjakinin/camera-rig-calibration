#!/usr/bin/env python3
# AUTO_IMPORT_COMMON_START
from pathlib import Path as _CalibLabPath
import sys as _CalibLabSys
for _p in _CalibLabPath(__file__).resolve().parents:
    if _p.name == "calib_lab":
        if str(_p) not in _CalibLabSys.path:
            _CalibLabSys.path.insert(0, str(_p))
        _common_scripts = _p / "common" / "scripts"
        if _common_scripts.exists() and str(_common_scripts) not in _CalibLabSys.path:
            _CalibLabSys.path.insert(0, str(_common_scripts))
        break
# AUTO_IMPORT_COMMON_END

import cv2
import numpy as np
from pathlib import Path


# This script is located at:
# project/src/calib_lab/minimal_world/scripts/tools/generate_checkerboard.py
#
# package_dir becomes:
# project/src/calib_lab
package_dir = Path(__file__).resolve().parents[1]

# 10 x 7 squares => OpenCV inner corners: 9 x 6
squares_x = 10
squares_y = 7
square_px = 100

# White margin is important for robust OpenCV checkerboard detection.
margin_px = 100

board_w = squares_x * square_px
board_h = squares_y * square_px

img_w = board_w + 2 * margin_px
img_h = board_h + 2 * margin_px

img = np.ones((img_h, img_w), dtype=np.uint8) * 255

for y in range(squares_y):
    for x in range(squares_x):
        if (x + y) % 2 == 0:
            x0 = margin_px + x * square_px
            y0 = margin_px + y * square_px
            x1 = margin_px + (x + 1) * square_px
            y1 = margin_px + (y + 1) * square_px
            cv2.rectangle(img, (x0, y0), (x1, y1), 0, -1)

out = package_dir / "models/checkerboard_target/materials/textures/checkerboard_10x7.png"
out.parent.mkdir(parents=True, exist_ok=True)

ok = cv2.imwrite(str(out), img)
if not ok:
    raise RuntimeError(f"Failed to write checkerboard image to {out}")

print(f"Saved checkerboard to: {out}")
print(f"Image size: {img_w} x {img_h}")
print("Board: 10 x 7 squares")
print("OpenCV inner corners: 9 x 6")
