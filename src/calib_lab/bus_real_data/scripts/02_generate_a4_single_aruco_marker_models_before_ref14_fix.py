#!/usr/bin/env python3

from pathlib import Path
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"

DICT_NAME = "DICT_4X4_50"
MARKER_IDS = list(range(14))

A4_WIDTH_M = 0.210
A4_HEIGHT_M = 0.297
MARKER_SIZE_M = 0.170

# OpenCV 4x4 ArUco has 4 inner bits + 1 black border cell on each side = 6x6 cells.
TOTAL_CELLS = 6
CELL_M = MARKER_SIZE_M / TOTAL_CELLS

def get_dict():
    aruco = cv2.aruco
    return aruco.getPredefinedDictionary(getattr(aruco, DICT_NAME))

def generate_marker_image(marker_id: int, size_px: int = 600):
    aruco = cv2.aruco
    dictionary = get_dict()

    if hasattr(aruco, "generateImageMarker"):
        img = aruco.generateImageMarker(dictionary, marker_id, size_px)
    else:
        img = np.zeros((size_px, size_px), dtype=np.uint8)
        aruco.drawMarker(dictionary, marker_id, size_px, img, 1)

    return img

def sample_cells(marker_img):
    h, w = marker_img.shape[:2]
    cells = []
    for r in range(TOTAL_CELLS):
        row = []
        for c in range(TOTAL_CELLS):
            y = int((r + 0.5) * h / TOTAL_CELLS)
            x = int((c + 0.5) * w / TOTAL_CELLS)
            row.append(1 if marker_img[y, x] < 128 else 0)  # 1 = black
        cells.append(row)
    return cells

def black_cell_visual(name, row, col):
    # Board lies in local XZ plane, thin dimension is Y.
    # Front face is toward local -Y.
    x = -MARKER_SIZE_M / 2 + (col + 0.5) * CELL_M
    z =  MARKER_SIZE_M / 2 - (row + 0.5) * CELL_M
    y = -0.004

    size = CELL_M * 1.04

    return f"""
        <visual name="{name}">
          <cast_shadows>false</cast_shadows>
          <pose>{x:.6f} {y:.6f} {z:.6f} 0 0 0</pose>
          <geometry>
            <box>
              <size>{size:.6f} 0.003 {size:.6f}</size>
            </box>
          </geometry>
          <material>
            <ambient>0 0 0 1</ambient>
            <diffuse>0 0 0 1</diffuse>
            <specular>0 0 0 1</specular>
          </material>
        </visual>
"""

def write_model(marker_id: int):
    model_name = f"a4_aruco_marker_{marker_id:03d}"
    model_dir = MODELS_DIR / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    img = generate_marker_image(marker_id)
    cells = sample_cells(img)

    black_visuals = []
    for r in range(TOTAL_CELLS):
        for c in range(TOTAL_CELLS):
            if cells[r][c] == 1:
                black_visuals.append(black_cell_visual(f"black_cell_{r}_{c}", r, c))

    sdf = f"""<?xml version="1.0" ?>
<sdf version="1.9">
  <model name="{model_name}">
    <static>true</static>
    <link name="board_link">

      <visual name="a4_white_sheet">
          <cast_shadows>false</cast_shadows>
        <pose>0 0 0 0 0 0</pose>
        <geometry>
          <box>
            <size>{A4_WIDTH_M:.6f} 0.004 {A4_HEIGHT_M:.6f}</size>
          </box>
        </geometry>
        <material>
          <ambient>1 1 1 1</ambient>
          <diffuse>1 1 1 1</diffuse>
            <specular>0 0 0 1</specular>
        </material>
      </visual>

{''.join(black_visuals)}

    </link>
  </model>
</sdf>
"""

    config = f"""<?xml version="1.0" ?>
<model>
  <name>{model_name}</name>
  <version>1.0</version>
  <sdf version="1.9">model.sdf</sdf>
  <author>
    <name>bus_real_data</name>
  </author>
  <description>A4 sheet with one geometry-based ArUco marker, ID {marker_id}, marker size 0.17 m.</description>
</model>
"""

    (model_dir / "model.sdf").write_text(sdf)
    (model_dir / "model.config").write_text(config)

    print(f"[OK] wrote {model_dir}")

def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    for marker_id in MARKER_IDS:
        write_model(marker_id)

if __name__ == "__main__":
    main()
