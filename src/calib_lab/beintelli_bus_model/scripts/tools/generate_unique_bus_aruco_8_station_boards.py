#!/usr/bin/env python3
from pathlib import Path
import cv2
import numpy as np

BASE_DIR = Path("src/calib_lab/beintelli_bus_model/models")

DICT_NAME = "DICT_4X4_50"

BOARD_WIDTH_M = 0.80
BOARD_HEIGHT_M = 0.60
BOARD_THICKNESS_M = 0.02

TEXTURE_W = 1440
TEXTURE_H = 1080

COLS = 3
ROWS = 2
MARKER_PX = 300
GAP_PX = 80

# 8 stations total:
# 4 front: F1/F2/F3/F4
# 3 rear:  R1/R2/R3
# 1 floor/general: G
BOARDS = [
    ("aruco_station_F3_ids_00_05", 0,  "F3_front_near_left_seat"),
    ("aruco_station_F4_ids_06_11", 6,  "F4_front_right_table_or_box"),
    ("aruco_station_R3_ids_12_17", 12, "R3_rear_right_seat_angled"),
    ("aruco_station_R2_ids_18_23", 18, "R2_rear_table_flat"),
    ("aruco_station_R1_ids_24_29", 24, "R1_rear_left_seat_leaned_occluded"),
    ("aruco_station_F1_ids_30_35", 30, "F1_front_mid_far_seat_leaned"),
    ("aruco_station_F2_ids_36_41", 36, "F2_front_mid_high_left_seat"),
    ("aruco_station_G_ids_42_47",  42, "G_floor_mid"),
]

def get_dictionary(name: str):
    if not hasattr(cv2, "aruco"):
        raise RuntimeError("cv2.aruco not available. Install OpenCV contrib.")
    return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, name))

def generate_marker(dictionary, marker_id: int, size_px: int):
    if hasattr(cv2.aruco, "generateImageMarker"):
        return cv2.aruco.generateImageMarker(dictionary, marker_id, size_px)
    marker = np.zeros((size_px, size_px), dtype=np.uint8)
    cv2.aruco.drawMarker(dictionary, marker_id, size_px, marker, 1)
    return marker

def make_texture(first_id: int):
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
    return rgb, ids

def write_model(model_name: str, first_id: int, station_name: str):
    model_dir = BASE_DIR / model_name
    texture_dir = model_dir / "materials" / "textures"

    model_dir.mkdir(parents=True, exist_ok=True)
    texture_dir.mkdir(parents=True, exist_ok=True)

    texture, ids = make_texture(first_id)

    texture_name = f"{model_name}.png"
    texture_path = texture_dir / texture_name
    cv2.imwrite(str(texture_path), texture)

    texture_uri = texture_path.resolve().as_uri()

    (model_dir / "model.config").write_text(f"""<?xml version="1.0"?>
<model>
  <name>{model_name}</name>
  <version>1.0</version>
  <sdf version="1.8">model.sdf</sdf>
  <author>
    <name>Camera Rig Calibration Lab</name>
  </author>
  <description>Unique ArUco station board: {station_name}</description>
</model>
""")

    # Direct model with absolute file URI for albedo_map to avoid model:// material caching.
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
              <albedo_map>{texture_uri}</albedo_map>
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
        "UNIQUE ARUCO 8-STATION BOARD METADATA\n"
        "=====================================\n"
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
        f"texture_uri: {texture_uri}\n"
    )

    black_ratio = float(np.mean(texture[:, :, 0] < 128))
    print(f"[OK] {model_name}: ids={ids}, black_ratio={black_ratio:.3f}, texture_uri={texture_uri}")

def main():
    for model_name, first_id, station_name in BOARDS:
        write_model(model_name, first_id, station_name)

if __name__ == "__main__":
    main()
