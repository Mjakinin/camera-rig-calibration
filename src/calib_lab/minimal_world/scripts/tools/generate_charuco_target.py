#!/usr/bin/env python3

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


def create_charuco_board(dictionary, squares_x, squares_y, square_length, marker_length):
    # OpenCV API differs by version.
    if hasattr(cv2.aruco, "CharucoBoard_create"):
        return cv2.aruco.CharucoBoard_create(
            squares_x,
            squares_y,
            square_length,
            marker_length,
            dictionary,
        )

    return cv2.aruco.CharucoBoard(
        (squares_x, squares_y),
        square_length,
        marker_length,
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


def patch_text_file(path: Path, target_width_m: float, target_height_m: float):
    text = path.read_text(errors="ignore")

    # Replace only known model names, not every occurrence of "aruco".
    text = text.replace("checkerboard_target", "charuco_target")
    text = text.replace("aruco_target", "charuco_target")
    text = text.replace("Checkerboard", "ChArUco")
    text = text.replace("checkerboard", "charuco")

    # Force texture references to the generated ChArUco board texture.
    text = re.sub(
        r"model://[^/]+/materials/textures/[^<>\s'\"]+\.png",
        "model://charuco_target/materials/textures/charuco_board.png",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r"(?<=<albedo_map>)[^<]*\.png(?=</albedo_map>)",
        "model://charuco_target/materials/textures/charuco_board.png",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r"(?<=<uri>)[^<]*\.png(?=</uri>)",
        "model://charuco_target/materials/textures/charuco_board.png",
        text,
        flags=re.IGNORECASE,
    )

    # Existing target model convention: thickness, width, height.
    text = re.sub(
        r"<size>\s*0\.02\s+[0-9.]+\s+[0-9.]+\s*</size>",
        f"<size>0.02 {target_width_m:.6f} {target_height_m:.6f}</size>",
        text,
        flags=re.IGNORECASE,
    )

    path.write_text(text)


def main():
    dictionary_name = "DICT_4X4_50"

    # ChArUco board: squares_x by squares_y squares.
    # ChArUco corner count is (squares_x - 1) * (squares_y - 1).
    squares_x = 7
    squares_y = 5
    square_length_m = 0.12
    marker_length_m = 0.09

    board_width_m = squares_x * square_length_m
    board_height_m = squares_y * square_length_m

    # Target/canvas includes white margin around the actual ChArUco board.
    # These dimensions are patched into model.sdf, so the texture scale is metric.
    target_width_m = 1.20
    target_height_m = 0.84

    image_width_px = 1200
    image_height_px = 840

    # Prefer copying the checkerboard target, since it already has the correct model structure.
    source_candidates = [
        Path("src/calib_lab/models/checkerboard_target"),
        Path("src/calib_lab/models/aruco_target"),
    ]
    source_model = next((p for p in source_candidates if p.exists()), None)

    if source_model is None:
        raise FileNotFoundError(
            "Missing source model. Expected one of: "
            + ", ".join(str(p) for p in source_candidates)
        )

    target_model = Path("src/calib_lab/models/charuco_target")

    if target_model.exists():
        shutil.rmtree(target_model)

    shutil.copytree(source_model, target_model)

    texture_dir = target_model / "materials" / "textures"
    texture_dir.mkdir(parents=True, exist_ok=True)

    dictionary = get_dictionary(dictionary_name)
    board = create_charuco_board(
        dictionary=dictionary,
        squares_x=squares_x,
        squares_y=squares_y,
        square_length=square_length_m,
        marker_length=marker_length_m,
    )

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

    texture_path = texture_dir / "charuco_board.png"
    cv2.imwrite(str(texture_path), canvas)

    for path in target_model.rglob("*"):
        if path.is_file() and path.suffix.lower() in [".sdf", ".config", ".material", ".txt"]:
            patch_text_file(path, target_width_m=target_width_m, target_height_m=target_height_m)

    cfg = {
        "charuco": {
            "dictionary": dictionary_name,
            "squares_x": squares_x,
            "squares_y": squares_y,
            "square_length": square_length_m,
            "marker_length": marker_length_m,
            "target_width": target_width_m,
            "target_height": target_height_m,
            "board_width": board_width_m,
            "board_height": board_height_m,
            "image_width_px": image_width_px,
            "image_height_px": image_height_px,
            "min_markers": 1,
            "min_charuco_corners": 4,
        }
    }

    cfg_path = Path("src/calib_lab/config/charuco_target.yaml")
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))

    print("Created ChArUco target.")
    print(f"Model:   {target_model}")
    print(f"Texture: {texture_path}")
    print(f"Config:  {cfg_path}")
    print("")
    print("Geometry:")
    for k, v in cfg["charuco"].items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
