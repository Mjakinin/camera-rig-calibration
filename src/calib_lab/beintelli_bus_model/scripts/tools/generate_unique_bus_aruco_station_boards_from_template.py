#!/usr/bin/env python3
from pathlib import Path
import shutil
import re
import cv2
import numpy as np

BASE_DIR = Path("src/calib_lab/beintelli_bus_model/models")
TEMPLATE = BASE_DIR / "aruco_gridboard_target_a1_080x060"

# Need IDs up to 85, so DICT_4X4_100 is required.
DICT_NAME = "DICT_4X4_100"

BOARD_WIDTH_M = 0.80
BOARD_HEIGHT_M = 0.60
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

def get_dictionary(name):
    if not hasattr(cv2, "aruco"):
        raise RuntimeError("cv2.aruco missing. Need opencv-contrib.")
    return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, name))

def generate_marker(dictionary, marker_id, size_px):
    if hasattr(cv2.aruco, "generateImageMarker"):
        return cv2.aruco.generateImageMarker(dictionary, marker_id, size_px)
    marker = np.zeros((size_px, size_px), dtype=np.uint8)
    cv2.aruco.drawMarker(dictionary, marker_id, size_px, marker, 1)
    return marker

def make_texture(first_id):
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

def replace_text_file(path, replacements):
    try:
        text = path.read_text()
    except UnicodeDecodeError:
        return

    old = text
    for a, b in replacements.items():
        text = text.replace(a, b)

    if text != old:
        path.write_text(text)

def patch_materials_and_sdf(model_dir: Path, model_name: str, texture_name: str):
    material_name = f"{model_name}/unique_marker_material"

    # Make / overwrite a unique Gazebo material script.
    scripts_dir = model_dir / "materials" / "scripts"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    material_file = scripts_dir / f"{model_name}.material"

    material_file.write_text(f"""
material {material_name}
{{
  technique
  {{
    pass
    {{
      ambient 1 1 1 1
      diffuse 1 1 1 1
      specular 0.1 0.1 0.1 1
      emissive 1 1 1 1

      texture_unit
      {{
        texture {texture_name}
        filtering none
      }}
    }}
  }}
}}
""".strip() + "\n")

    model_sdf = model_dir / "model.sdf"
    sdf = model_sdf.read_text()

    # Replace model names / template references.
    sdf = sdf.replace("aruco_gridboard_target_a1_080x060", model_name)
    sdf = sdf.replace("aruco_gridboard_target", model_name)

    # Force material block to use unique script material.
    material_block = f"""
        <material>
          <script>
            <uri>model://{model_name}/materials/scripts</uri>
            <uri>model://{model_name}/materials/textures</uri>
            <name>{material_name}</name>
          </script>
          <ambient>1 1 1 1</ambient>
          <diffuse>1 1 1 1</diffuse>
          <specular>0.1 0.1 0.1 1</specular>
          <emissive>1 1 1 1</emissive>
          <pbr>
            <metal>
              <albedo_map>model://{model_name}/materials/textures/{texture_name}</albedo_map>
              <roughness>0.8</roughness>
              <metalness>0.0</metalness>
            </metal>
          </pbr>
        </material>""".rstrip()

    # Replace first <material>...</material> inside visual if present.
    sdf_new = re.sub(r"\s*<material>.*?</material>", "\n" + material_block, sdf, count=1, flags=re.DOTALL)

    if sdf_new == sdf:
        # Fallback: insert material after geometry close in target visual.
        sdf_new = sdf.replace("</geometry>", "</geometry>\n" + material_block, 1)

    model_sdf.write_text(sdf_new)

def write_board(model_name, first_id, station_name):
    if not TEMPLATE.exists():
        raise RuntimeError(f"Template model missing: {TEMPLATE}")
    if not (TEMPLATE / "model.config").exists():
        raise RuntimeError(f"Template model.config missing: {TEMPLATE / 'model.config'}")

    model_dir = BASE_DIR / model_name
    if model_dir.exists():
        shutil.rmtree(model_dir)

    shutil.copytree(TEMPLATE, model_dir)

    texture, ids = make_texture(first_id)

    texture_dir = model_dir / "materials" / "textures"
    texture_dir.mkdir(parents=True, exist_ok=True)

    # Delete old copied PNGs to avoid accidental reuse.
    for old_png in texture_dir.glob("*.png"):
        old_png.unlink()

    texture_name = f"{model_name}.png"
    texture_path = texture_dir / texture_name
    cv2.imwrite(str(texture_path), texture)

    replacements = {
        TEMPLATE.name: model_name,
        "aruco_gridboard_target_a1_080x060": model_name,
        "aruco_gridboard_target": model_name,
    }

    for p in model_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in [".sdf", ".config", ".material", ".txt"]:
            replace_text_file(p, replacements)

    patch_materials_and_sdf(model_dir, model_name, texture_name)

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

    black_ratio = float(np.mean(texture[:, :, 0] < 128))
    print(f"[OK] {model_name}: dict={DICT_NAME}, ids={ids}, texture={texture_name}, black_ratio={black_ratio:.3f}")

def main():
    for model_name, first_id, station_name in BOARDS:
        write_board(model_name, first_id, station_name)

if __name__ == "__main__":
    main()
