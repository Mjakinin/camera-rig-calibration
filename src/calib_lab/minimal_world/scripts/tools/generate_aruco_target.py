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
import shutil
import re
import cv2
import numpy as np
import yaml


def get_dictionary(dictionary_name: str):
    if not hasattr(cv2, "aruco"):
        raise RuntimeError("cv2.aruco is not available. Install/use OpenCV contrib.")

    if not hasattr(cv2.aruco, dictionary_name):
        raise RuntimeError(f"Unknown ArUco dictionary: {dictionary_name}")

    return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary_name))


def create_grid_board(dictionary, markers_x, markers_y, marker_length, marker_separation):
    # OpenCV API differs by version.
    if hasattr(cv2.aruco, "GridBoard_create"):
        return cv2.aruco.GridBoard_create(
            markers_x,
            markers_y,
            marker_length,
            marker_separation,
            dictionary,
        )

    return cv2.aruco.GridBoard(
        (markers_x, markers_y),
        marker_length,
        marker_separation,
        dictionary,
    )


def draw_board(board, width_px, height_px):
    if hasattr(board, "generateImage"):
        return board.generateImage(
            (width_px, height_px),
            marginSize=0,
            borderBits=1,
        )

    return cv2.aruco.drawPlanarBoard(
        board,
        (width_px, height_px),
        marginSize=0,
        borderBits=1,
    )


def patch_text_file(path: Path):
    text = path.read_text(errors="ignore")

    text = text.replace("checkerboard_target", "aruco_target")
    text = text.replace("Checkerboard", "ArUco")
    text = text.replace("checkerboard", "aruco")
    text = text.replace("model://checkerboard_target", "model://aruco_target")

    # Force texture references to the generated ArUco board texture.
    text = re.sub(
        r"model://aruco_target/materials/textures/[^<>\s'\"]+\.png",
        "model://aruco_target/materials/textures/aruco_board.png",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r"model://checkerboard_target/materials/textures/[^<>\s'\"]+\.png",
        "model://aruco_target/materials/textures/aruco_board.png",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r"(?<=<uri>)[^<]*\.png(?=</uri>)",
        "model://aruco_target/materials/textures/aruco_board.png",
        text,
        flags=re.IGNORECASE,
    )

    path.write_text(text)


def main():
    dictionary_name = "DICT_4X4_50"

    markers_x = 6
    markers_y = 4
    marker_length_m = 0.15
    marker_separation_m = 0.06

    target_width_m = 1.60
    target_height_m = 1.10

    image_width_px = 1600
    image_height_px = 1100

    source_model = Path("src/calib_lab/minimal_world/models/checkerboard_target")
    target_model = Path("src/calib_lab/minimal_world/models/aruco_target")

    if not source_model.exists():
        raise FileNotFoundError(f"Missing source model: {source_model}")

    if target_model.exists():
        shutil.rmtree(target_model)

    shutil.copytree(source_model, target_model)

    texture_dir = target_model / "materials" / "textures"
    texture_dir.mkdir(parents=True, exist_ok=True)

    dictionary = get_dictionary(dictionary_name)

    board = create_grid_board(
        dictionary=dictionary,
        markers_x=markers_x,
        markers_y=markers_y,
        marker_length=marker_length_m,
        marker_separation=marker_separation_m,
    )

    board_width_m = markers_x * marker_length_m + (markers_x - 1) * marker_separation_m
    board_height_m = markers_y * marker_length_m + (markers_y - 1) * marker_separation_m

    board_width_px = int(round(image_width_px * board_width_m / target_width_m))
    board_height_px = int(round(image_height_px * board_height_m / target_height_m))

    board_width_px = min(board_width_px, image_width_px)
    board_height_px = min(board_height_px, image_height_px)

    board_img = draw_board(board, board_width_px, board_height_px)

    if len(board_img.shape) == 3:
        board_img = cv2.cvtColor(board_img, cv2.COLOR_BGR2GRAY)

    canvas = np.full((image_height_px, image_width_px), 255, dtype=np.uint8)

    x0 = (image_width_px - board_width_px) // 2
    y0 = (image_height_px - board_height_px) // 2

    canvas[y0:y0 + board_height_px, x0:x0 + board_width_px] = board_img

    texture_path = texture_dir / "aruco_board.png"
    cv2.imwrite(str(texture_path), canvas)

    # Patch copied model files.
    for path in target_model.rglob("*"):
        if path.is_file() and path.suffix.lower() in [".sdf", ".config", ".material", ".txt"]:
            patch_text_file(path)

    cfg = {
        "aruco": {
            "dictionary": dictionary_name,
            "markers_x": markers_x,
            "markers_y": markers_y,
            "marker_length": marker_length_m,
            "marker_separation": marker_separation_m,
            "target_width": target_width_m,
            "target_height": target_height_m,
            "board_width": board_width_m,
            "board_height": board_height_m,
            "image_width_px": image_width_px,
            "image_height_px": image_height_px,
        }
    }

    cfg_path = Path("src/calib_lab/minimal_world/config/aruco_target.yaml")
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))

    print("Created ArUco target.")
    print(f"Model:   {target_model}")
    print(f"Texture: {texture_path}")
    print(f"Config:  {cfg_path}")
    print("")
    print("Geometry:")
    for k, v in cfg["aruco"].items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
