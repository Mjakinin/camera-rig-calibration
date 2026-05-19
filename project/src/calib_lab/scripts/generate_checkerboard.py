#!/usr/bin/env python3
import cv2
import numpy as np
from pathlib import Path

# 10 x 7 squares => OpenCV inner corners: 9 x 6
squares_x = 10
squares_y = 7
square_px = 100
margin_px = 100   # important for OpenCV detection

board_w = squares_x * square_px
board_h = squares_y * square_px

img_w = board_w + 2 * margin_px
img_h = board_h + 2 * margin_px

# white background including margin
img = np.ones((img_h, img_w), dtype=np.uint8) * 255

for y in range(squares_y):
    for x in range(squares_x):
        if (x + y) % 2 == 0:
            x0 = margin_px + x * square_px
            y0 = margin_px + y * square_px
            x1 = margin_px + (x + 1) * square_px
            y1 = margin_px + (y + 1) * square_px
            cv2.rectangle(img, (x0, y0), (x1, y1), 0, -1)

out = Path("/workspaces/project/src/calib_lab/models/checkerboard_target/materials/textures/checkerboard_10x7.png")
out.parent.mkdir(parents=True, exist_ok=True)
cv2.imwrite(str(out), img)

print(f"Saved checkerboard to: {out}")
print(f"Image size: {img_w} x {img_h}")
print("OpenCV inner corners: 9 x 6")
