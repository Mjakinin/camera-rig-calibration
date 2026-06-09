#!/usr/bin/env python3
from pathlib import Path
import cv2
import numpy as np

BASE_DIR = Path("src/calib_lab/beintelli_bus_model/models")

# Need 0..85 IDs, therefore DICT_4X4_100 instead of DICT_4X4_50.
DICT_NAME = "DICT_4X4_100"

BOARD_WIDTH_M = 0.80
BOARD_HEIGHT_M = 0.60
BOARD_THICKNESS_M = 0.02

TEXTURE_W = 1440
TEXTURE_H = 1080

COLS = 3
ROWS = 2
MARKER_PX = 300
GAP_PX = 80

BOARDS = [
    ("aruco_station_F3_ids_00_05", 0,  "F3_front_near_left_seat"),
    ("aruco_station_F4_ids_10_15", 10, "F4_front_right_table_or_box"),
    ("aruco_station_R3_ids_20_25", 20, "R3_rear_right_seat_angled"),
    ("aruco_station_R2_ids_30_35", 30, "R2_rear_table_flat"),
    ("aruco_station_R1_ids_40_45", 40, "R1_rear_left_seat_leaned_occluded"),

    ("aruco_station_F1_ids_50_55", 50, "F1_front_mid_far_seat_leaned"),
    ("aruco_station_F2_ids_60_65", 60, "F2_front_mid_high_left_seat"),
    ("aruco_station_G1_ids_70_75", 70, "G1_floor_mid_yaw_pi"),
    ("aruco_station_G2_ids_80_85", 80, "G2_floor_mid_yaw_0"),
]

def get_dictionary(name: str):
    if not hasattr(cv2, "aruco"):
        raise RuntimeError("cv2.aruco not available. Install OpenCV contrib.")
    if not hasattr(cv2.aruco, name):
        raise RuntimeError(f"Unknown ArUco dictionary: {name}")
    return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, name))

def generate_marker(dictionary, marker_id: int, size_px: int):
    if hasattr(cv2.aruco, "generateImageMarker"):
        return cv2.aruco.generateImageMarker(dictionary, marker_id, size_px)
    marker = np.zeros((size_px, size_px), dtype=np.uint8)
    cv2.aruco.drawMarker(dictionary, marker_id, size_px, marker, 1)
    return marker

def write_board(model_name: str, first_id: int, station_name: str):
    model_dir = BASE_DIR / model_name
    texture_dir = model_dir / "materials" / "textures"
    texture_dir.mkdir(parents=True, exist_ok=True)

    dictionary = get_dictionary(DICT_NAME)
    canvas = np.full((TEXTURE_H, TEXTURE_W), 255, dtype=np.uint8)

    grid_w = COLS * MARKER_PX + (COLS - 1) * GAP_PX
    grid_h = ROWS * MARKER_PX + (ROWS - 1) * GAP_PX

    x0 = (TEXTURE_W - grid_w) // 2
    y0 = (TEXTURE_H - grid_h) // 2

    ids = []
    for r in range(ROWS):
        for c in range(COLS):
            marker_id = first_id + r * COLS + c
            ids.append(marker_id)

            marker = generate_marker(dictionary, marker_id, MARKER_PX)
            x = x0 + c * (MARKER_PX + GAP_PX)
            y = y0 + r * (MARKER_PX + GAP_PX)
            canvas[y:y + MARKER_PX, x:x + MARKER_PX] = marker

    rgb = cv2.cvtColor(canvas, cv2.COLOR_GRAY2RGB)
    texture_path = texture_dir / "aruco_gridboard.png"
    cv2.imwrite(str(texture_path), rgb)

    (model_dir / "model.config").write_text(f"""<?xml version="1.0"?>
<model>
  <name>{model_name}</name>
  <version>1.0</version>
  <sdf version="1.8">model.sdf</sdf>
  <author>
    <name>Camera Rig Calibration Lab</name>
  </author>
  <description>Unique ArUco GridBoard target for station {station_name}.</description>
</model>
""")

    (model_dir / "model.sdf").write_text(f"""<?xml version="1.0"?>
<sdf version="1.8">
  <model name="{model_name}">
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
              <albedo_map>model://{model_name}/materials/textures/aruco_gridboard.png</albedo_map>
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

    marker_size_m = MARKER_PX / TEXTURE_W * BOARD_WIDTH_M
    gap_m = GAP_PX / TEXTURE_W * BOARD_WIDTH_M

    (model_dir / "board_layout.txt").write_text(
        "UNIQUE ARUCO GRIDBOARD TARGET METADATA\n"
        "======================================\n"
        f"station: {station_name}\n"
        f"model_name: {model_name}\n"
        f"dictionary: {DICT_NAME}\n"
        f"ids: {ids}\n"
        f"first_id: {first_id}\n"
        f"cols: {COLS}\n"
        f"rows: {ROWS}\n"
        f"board_width_m: {BOARD_WIDTH_M}\n"
        f"board_height_m: {BOARD_HEIGHT_M}\n"
        f"texture_width_px: {TEXTURE_W}\n"
        f"texture_height_px: {TEXTURE_H}\n"
        f"marker_px: {MARKER_PX}\n"
        f"gap_px: {GAP_PX}\n"
        f"marker_size_m: {marker_size_m:.6f}\n"
        f"gap_m: {gap_m:.6f}\n"
    )

    print(f"[OK] {model_name}: dictionary={DICT_NAME}, ids={ids}, texture={texture_path}")

def main():
    for model_name, first_id, station_name in BOARDS:
        write_board(model_name, first_id, station_name)

if __name__ == "__main__":
    main()
