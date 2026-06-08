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

from pathlib import Path
import argparse
import cv2
import numpy as np


def get_dictionary(name: str):
    if not hasattr(cv2, "aruco"):
        raise RuntimeError("OpenCV aruco module not available.")
    if not hasattr(cv2.aruco, name):
        raise RuntimeError(f"Unknown ArUco dictionary: {name}")
    return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, name))


def draw_marker(dictionary, marker_id: int, side_px: int):
    if hasattr(cv2.aruco, "generateImageMarker"):
        return cv2.aruco.generateImageMarker(dictionary, marker_id, side_px)

    img = np.zeros((side_px, side_px), dtype=np.uint8)
    cv2.aruco.drawMarker(dictionary, marker_id, side_px, img, 1)
    return img


def write_marker_model(out_root: Path, marker_id: int, marker_size_m: float, board_margin_m: float, dictionary_name: str):
    model_name = f"aruco_marker_{marker_id:02d}"
    model_dir = out_root / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    dictionary = get_dictionary(dictionary_name)

    # DICT_4X4 marker = 4 inner bits + 1 black border each side = 6x6 cells.
    total_cells = 6
    sample_px = 600
    img = draw_marker(dictionary, marker_id, sample_px)

    cell_px = sample_px // total_cells
    white_cells = []

    for r in range(total_cells):
        for c in range(total_cells):
            patch = img[
                r * cell_px : (r + 1) * cell_px,
                c * cell_px : (c + 1) * cell_px,
            ]
            mean = float(np.mean(patch))
            if mean > 128:
                white_cells.append((r, c))

    board_size = marker_size_m + 2.0 * board_margin_m
    marker_cell = marker_size_m / total_cells

    board_thickness = 0.002
    black_thickness = 0.004
    white_thickness = 0.003

    visuals = []

    # White outer board.
    visuals.append(f"""
      <visual name="white_outer_board">
        <cast_shadows>false</cast_shadows>
        <pose>0 0 0 0 0 0</pose>
        <geometry>
          <box>
            <size>{board_size:.6f} {board_size:.6f} {board_thickness:.6f}</size>
          </box>
        </geometry>
        <material>
          <ambient>1 1 1 1</ambient>
          <diffuse>1 1 1 1</diffuse>
          <emissive>1 1 1 1</emissive>
        </material>
      </visual>
""")

    # Continuous black marker square. This removes cracks in black areas.
    black_z = board_thickness / 2.0 + black_thickness / 2.0
    visuals.append(f"""
      <visual name="black_marker_base">
        <cast_shadows>false</cast_shadows>
        <pose>0 0 {black_z:.6f} 0 0 0</pose>
        <geometry>
          <box>
            <size>{marker_size_m:.6f} {marker_size_m:.6f} {black_thickness:.6f}</size>
          </box>
        </geometry>
        <material>
          <ambient>0 0 0 1</ambient>
          <diffuse>0 0 0 1</diffuse>
          <emissive>0 0 0 1</emissive>
        </material>
      </visual>
""")

    # White cells on top of black marker square.
    white_z = board_thickness / 2.0 + black_thickness + white_thickness / 2.0

    for r, c in white_cells:
        x = -marker_size_m / 2.0 + (c + 0.5) * marker_cell
        y = marker_size_m / 2.0 - (r + 0.5) * marker_cell

        visuals.append(f"""
      <visual name="white_cell_{r}_{c}">
        <cast_shadows>false</cast_shadows>
        <pose>{x:.6f} {y:.6f} {white_z:.6f} 0 0 0</pose>
        <geometry>
          <box>
            <size>{marker_cell * 1.02:.6f} {marker_cell * 1.02:.6f} {white_thickness:.6f}</size>
          </box>
        </geometry>
        <material>
          <ambient>1 1 1 1</ambient>
          <diffuse>1 1 1 1</diffuse>
          <emissive>1 1 1 1</emissive>
        </material>
      </visual>
""")

    model_config = model_dir / "model.config"
    model_config.write_text(f"""<?xml version="1.0"?>
<model>
  <name>{model_name}</name>
  <version>1.0</version>
  <sdf version="1.7">model.sdf</sdf>
  <description>Geometry-only ArUco marker {marker_id}, dictionary {dictionary_name}</description>
</model>
""")

    model_sdf = model_dir / "model.sdf"
    model_sdf.write_text(f"""<?xml version="1.0"?>
<sdf version="1.7">
  <model name="{model_name}">
    <static>true</static>
    <link name="link">
{''.join(visuals)}
    </link>
  </model>
</sdf>
""")

    print(f"[OK] Created robust geometry-only marker: {model_name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", default="src/calib_lab/beintelli_bus_model/models/aruco_individual")
    parser.add_argument("--dictionary", default="DICT_4X4_50")
    parser.add_argument("--ids", nargs="+", type=int, default=list(range(10)))
    parser.add_argument("--marker-size-m", type=float, default=0.50)
    parser.add_argument("--board-margin-m", type=float, default=0.05)
    args = parser.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    for marker_id in args.ids:
        write_marker_model(
            out_root=out_root,
            marker_id=marker_id,
            marker_size_m=args.marker_size_m,
            board_margin_m=args.board_margin_m,
            dictionary_name=args.dictionary,
        )

    print("[DONE] Robust geometry-only ArUco marker models generated.")


if __name__ == "__main__":
    main()
