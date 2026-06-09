#!/usr/bin/env python3
from pathlib import Path
import cv2
import numpy as np
import html

BASE_DIR = Path("src/calib_lab/beintelli_bus_model/models")

DICT_NAME = "DICT_4X4_50"
DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

BOARD_WIDTH_M = 0.80
BOARD_HEIGHT_M = 0.60
BOARD_THICKNESS_M = 0.02

TEXTURE_W = 1440
TEXTURE_H = 1080

COLS = 3
ROWS = 2
MARKER_PX = 300
GAP_PX = 80

# 4x4 dictionary marker = 4 code bits + black border = 6x6 visible cells.
MARKER_CELLS = 6
CELL_SAMPLE = 30
MARKER_SAMPLE_PX = MARKER_CELLS * CELL_SAMPLE

BLACK_CELL_THICKNESS_M = 0.004
SURFACE_OFFSET_M = 0.001

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

def generate_marker(marker_id: int):
    if hasattr(cv2.aruco, "generateImageMarker"):
        marker = cv2.aruco.generateImageMarker(DICT, marker_id, MARKER_SAMPLE_PX)
    else:
        marker = np.zeros((MARKER_SAMPLE_PX, MARKER_SAMPLE_PX), dtype=np.uint8)
        cv2.aruco.drawMarker(DICT, marker_id, MARKER_SAMPLE_PX, marker, 1)
    return marker

def px_to_board(u_px: float, v_px: float):
    y_m = (u_px / TEXTURE_W - 0.5) * BOARD_WIDTH_M
    z_m = (0.5 - v_px / TEXTURE_H) * BOARD_HEIGHT_M
    return y_m, z_m

def marker_cells(marker_id: int):
    marker = generate_marker(marker_id)
    cells = np.zeros((MARKER_CELLS, MARKER_CELLS), dtype=np.uint8)
    for r in range(MARKER_CELLS):
        for c in range(MARKER_CELLS):
            patch = marker[
                r * CELL_SAMPLE:(r + 1) * CELL_SAMPLE,
                c * CELL_SAMPLE:(c + 1) * CELL_SAMPLE,
            ]
            cells[r, c] = 1 if float(np.mean(patch)) < 128.0 else 0
    return cells

def make_visual_box(name, pose, size, color):
    x, y, z, roll, pitch, yaw = pose
    sx, sy, sz = size
    r, g, b, a = color
    return f"""
      <visual name="{html.escape(name)}">
        <cast_shadows>false</cast_shadows>
        <pose>{x:.8f} {y:.8f} {z:.8f} {roll:.8f} {pitch:.8f} {yaw:.8f}</pose>
        <geometry>
          <box>
            <size>{sx:.8f} {sy:.8f} {sz:.8f}</size>
          </box>
        </geometry>
        <material>
          <ambient>{r} {g} {b} {a}</ambient>
          <diffuse>{r} {g} {b} {a}</diffuse>
          <specular>0 0 0 1</specular>
          <emissive>{r} {g} {b} {a}</emissive>
        </material>
      </visual>
"""

def build_black_cell_visuals(first_id: int):
    visuals = []
    visual_count = 0

    grid_w_px = COLS * MARKER_PX + (COLS - 1) * GAP_PX
    grid_h_px = ROWS * MARKER_PX + (ROWS - 1) * GAP_PX

    x0_px = (TEXTURE_W - grid_w_px) / 2.0
    y0_px = (TEXTURE_H - grid_h_px) / 2.0

    cell_px = MARKER_PX / MARKER_CELLS

    # Put valid marker patterns on both physical sides.
    # +X face: normal pattern.
    # -X face: mirrored in local Y, so from the back side the camera still sees a non-mirrored ArUco marker.
    faces = [
        ("front_plus_x", +1.0, False),
        ("back_minus_x", -1.0, True),
    ]

    for board_r in range(ROWS):
        for board_c in range(COLS):
            marker_id = first_id + board_r * COLS + board_c
            cells = marker_cells(marker_id)

            marker_u0 = x0_px + board_c * (MARKER_PX + GAP_PX)
            marker_v0 = y0_px + board_r * (MARKER_PX + GAP_PX)

            for face_name, face_sign, mirror_y in faces:
                x_center = face_sign * (BOARD_THICKNESS_M / 2.0 + BLACK_CELL_THICKNESS_M / 2.0 + SURFACE_OFFSET_M)

                for cell_r in range(MARKER_CELLS):
                    cell_c = 0
                    while cell_c < MARKER_CELLS:
                        if cells[cell_r, cell_c] == 0:
                            cell_c += 1
                            continue

                        run_start = cell_c
                        while cell_c < MARKER_CELLS and cells[cell_r, cell_c] == 1:
                            cell_c += 1
                        run_end = cell_c

                        u_left = marker_u0 + run_start * cell_px
                        u_right = marker_u0 + run_end * cell_px
                        v_top = marker_v0 + cell_r * cell_px
                        v_bottom = marker_v0 + (cell_r + 1) * cell_px

                        y_left, z_top = px_to_board(u_left, v_top)
                        y_right, z_bottom = px_to_board(u_right, v_bottom)

                        y_center = (y_left + y_right) / 2.0
                        z_center = (z_top + z_bottom) / 2.0

                        if mirror_y:
                            y_center = -y_center

                        size_y = abs(y_right - y_left)
                        size_z = abs(z_top - z_bottom)

                        visual_name = f"black_cell_{face_name}_id{marker_id}_r{cell_r}_c{run_start}_{run_end}"
                        visuals.append(
                            make_visual_box(
                                visual_name,
                                (x_center, y_center, z_center, 0.0, 0.0, 0.0),
                                (BLACK_CELL_THICKNESS_M, size_y, size_z),
                                (0, 0, 0, 1),
                            )
                        )
                        visual_count += 1

    return "\n".join(visuals), visual_count

def write_model(model_name: str, first_id: int, station_name: str):
    model_dir = BASE_DIR / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    ids = list(range(first_id, first_id + COLS * ROWS))
    black_visuals, visual_count = build_black_cell_visuals(first_id)

    (model_dir / "model.config").write_text(f"""<?xml version="1.0"?>
<model>
  <name>{model_name}</name>
  <version>1.0</version>
  <sdf version="1.8">model.sdf</sdf>
  <author>
    <name>Camera Rig Calibration Lab</name>
  </author>
  <description>Geometry-based ArUco station board: {station_name}</description>
</model>
""")

    white_board_visual = make_visual_box(
        "white_board_base",
        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        (BOARD_THICKNESS_M, BOARD_WIDTH_M, BOARD_HEIGHT_M),
        (1, 1, 1, 1),
    )

    (model_dir / "model.sdf").write_text(f"""<?xml version="1.0"?>
<sdf version="1.8">
  <model name="{model_name}">
    <static>true</static>
    <link name="target_link">
{white_board_visual}
{black_visuals}
      <collision name="target_collision">
        <geometry>
          <box>
            <size>{BOARD_THICKNESS_M:.8f} {BOARD_WIDTH_M:.8f} {BOARD_HEIGHT_M:.8f}</size>
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
        "GEOMETRY-BASED UNIQUE ARUCO 8-STATION BOARD METADATA\n"
        "====================================================\n"
        f"station: {station_name}\n"
        f"model_name: {model_name}\n"
        f"dictionary: {DICT_NAME}\n"
        f"ids: {ids}\n"
        f"first_id: {first_id}\n"
        f"cols: {COLS}\n"
        f"rows: {ROWS}\n"
        f"board_width_m: {BOARD_WIDTH_M}\n"
        f"board_height_m: {BOARD_HEIGHT_M}\n"
        f"texture_width_px_equivalent: {TEXTURE_W}\n"
        f"texture_height_px_equivalent: {TEXTURE_H}\n"
        f"marker_px_equivalent: {MARKER_PX}\n"
        f"gap_px_equivalent: {GAP_PX}\n"
        f"marker_size_m: {marker_size_m:.6f}\n"
        f"gap_m: {gap_m:.6f}\n"
        f"black_cell_visuals: {visual_count}\n"
    )

    print(f"[OK] {model_name}: ids={ids}, black_cell_visuals={visual_count}")

def main():
    for model_name, first_id, station_name in BOARDS:
        write_model(model_name, first_id, station_name)

if __name__ == "__main__":
    main()
