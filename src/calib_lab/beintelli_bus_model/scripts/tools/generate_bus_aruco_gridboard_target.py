#!/usr/bin/env python3

from pathlib import Path
import cv2
import numpy as np

MODEL_NAME = "aruco_gridboard_target"
MODEL_DIR = Path("src/calib_lab/beintelli_bus_model/models") / MODEL_NAME
TEXTURE_DIR = MODEL_DIR / "materials" / "textures"

# Physical board size in Gazebo.
# The model is a vertical board: thickness in X, width in Y, height in Z.
BOARD_WIDTH_M = 1.20
BOARD_HEIGHT_M = 0.80
BOARD_THICKNESS_M = 0.02

# Texture size with same aspect ratio as board.
TEXTURE_W = 1200
TEXTURE_H = 800

# ArUco layout.
DICT_NAME = "DICT_4X4_50"
COLS = 4
ROWS = 3
MARKER_PX = 160
GAP_PX = 50

# IDs used on this board.
FIRST_ID = 0


def get_dictionary(name: str):
    if not hasattr(cv2, "aruco"):
        raise RuntimeError("cv2.aruco not available.")
    return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, name))


def generate_marker(dictionary, marker_id: int, size_px: int):
    if hasattr(cv2.aruco, "generateImageMarker"):
        marker = cv2.aruco.generateImageMarker(dictionary, marker_id, size_px)
    else:
        marker = np.zeros((size_px, size_px), dtype=np.uint8)
        cv2.aruco.drawMarker(dictionary, marker_id, size_px, marker, 1)
    return marker


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    TEXTURE_DIR.mkdir(parents=True, exist_ok=True)

    dictionary = get_dictionary(DICT_NAME)

    canvas = np.full((TEXTURE_H, TEXTURE_W), 255, dtype=np.uint8)

    grid_w = COLS * MARKER_PX + (COLS - 1) * GAP_PX
    grid_h = ROWS * MARKER_PX + (ROWS - 1) * GAP_PX

    x0 = (TEXTURE_W - grid_w) // 2
    y0 = (TEXTURE_H - grid_h) // 2

    ids = []
    for r in range(ROWS):
        for c in range(COLS):
            marker_id = FIRST_ID + r * COLS + c
            ids.append(marker_id)

            marker = generate_marker(dictionary, marker_id, MARKER_PX)
            x = x0 + c * (MARKER_PX + GAP_PX)
            y = y0 + r * (MARKER_PX + GAP_PX)
            canvas[y:y + MARKER_PX, x:x + MARKER_PX] = marker

    # RGB PNG because Gazebo/Ogre2 is safer with RGB textures.
    rgb = cv2.cvtColor(canvas, cv2.COLOR_GRAY2RGB)
    texture_path = TEXTURE_DIR / "aruco_gridboard.png"
    cv2.imwrite(str(texture_path), rgb)

    (MODEL_DIR / "model.config").write_text(f"""<?xml version="1.0"?>
<model>
  <name>{MODEL_NAME}</name>
  <version>1.0</version>
  <sdf version="1.8">model.sdf</sdf>
  <author>
    <name>Camera Rig Calibration Lab</name>
  </author>
  <description>Single textured ArUco GridBoard target for BeIntelli bus experiments.</description>
</model>
""")

    # Same robust style as minimal_world targets: thin box + albedo_map.
    (MODEL_DIR / "model.sdf").write_text(f"""<?xml version="1.0"?>
<sdf version="1.8">
  <model name="{MODEL_NAME}">
    <static>true</static>
    <link name="target_link">
      <visual name="target_visual">
        <cast_shadows>false</cast_shadows>
        <geometry>
          <box>
            <size>{BOARD_THICKNESS_M:.6f} {BOARD_WIDTH_M:.6f} {BOARD_HEIGHT_M:.6f}</size>
          </box>
        </geometry>

        <material>
          <ambient>1 1 1 1</ambient>
          <diffuse>1 1 1 1</diffuse>
          <specular>0.1 0.1 0.1 1</specular>
          <emissive>1 1 1 1</emissive>
          <pbr>
            <metal>
              <albedo_map>model://{MODEL_NAME}/materials/textures/aruco_gridboard.png</albedo_map>
              <roughness>0.8</roughness>
              <metalness>0.0</metalness>
            </metal>
          </pbr>
        </material>
      </visual>

      <collision name="target_collision">
        <geometry>
          <box>
            <size>{BOARD_THICKNESS_M:.6f} {BOARD_WIDTH_M:.6f} {BOARD_HEIGHT_M:.6f}</size>
          </box>
        </geometry>
      </collision>
    </link>
  </model>
</sdf>
""")

    # Also write metadata for later pose estimation.
    meta_path = MODEL_DIR / "board_layout.txt"
    marker_size_m = MARKER_PX / TEXTURE_W * BOARD_WIDTH_M
    gap_m = GAP_PX / TEXTURE_W * BOARD_WIDTH_M
    meta_path.write_text(
        "ARUCO GRIDBOARD TARGET METADATA\n"
        "===============================\n"
        f"dictionary: {DICT_NAME}\n"
        f"ids: {ids}\n"
        f"cols: {COLS}\n"
        f"rows: {ROWS}\n"
        f"board_width_m: {BOARD_WIDTH_M}\n"
        f"board_height_m: {BOARD_HEIGHT_M}\n"
        f"marker_size_m: {marker_size_m:.6f}\n"
        f"gap_m: {gap_m:.6f}\n"
        f"texture_width_px: {TEXTURE_W}\n"
        f"texture_height_px: {TEXTURE_H}\n"
        f"marker_px: {MARKER_PX}\n"
        f"gap_px: {GAP_PX}\n"
    )

    print("[OK] Generated ArUco GridBoard target")
    print(f"     model:   {MODEL_DIR}")
    print(f"     texture: {texture_path}")
    print(f"     ids:     {ids}")
    print(f"     marker_size_m: {marker_size_m:.6f}")
    print(f"     gap_m:         {gap_m:.6f}")


if __name__ == "__main__":
    main()
