#!/usr/bin/env python3

import cv2
import numpy as np
from pathlib import Path


# This script is located at:
# project/src/calib_lab/scripts/generate_aruco_marker.py
#
# package_dir becomes:
# project/src/calib_lab
package_dir = Path(__file__).resolve().parents[1]

# ArUco target settings
dictionary_name = "DICT_6X6_250"
dictionary_id = cv2.aruco.DICT_6X6_250
marker_id = 23

# Marker texture size in pixels
marker_px = 800

# White margin around the marker
margin_px = 100

img_w = marker_px + 2 * margin_px
img_h = marker_px + 2 * margin_px


def generate_marker(dictionary, marker_id, marker_px):
    marker_img = np.zeros((marker_px, marker_px), dtype=np.uint8)

    # Newer OpenCV
    if hasattr(cv2.aruco, "generateImageMarker"):
        cv2.aruco.generateImageMarker(
            dictionary,
            marker_id,
            marker_px,
            marker_img,
            borderBits=1,
        )

    # Older OpenCV
    else:
        cv2.aruco.drawMarker(
            dictionary,
            marker_id,
            marker_px,
            marker_img,
            borderBits=1,
        )

    return marker_img


def main():
    if not hasattr(cv2, "aruco"):
        raise RuntimeError(
            "cv2.aruco is not available. Install OpenCV with ArUco support."
        )

    dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)

    marker_img = generate_marker(dictionary, marker_id, marker_px)

    # White background with marker in the center
    img = np.ones((img_h, img_w), dtype=np.uint8) * 255
    img[
        margin_px:margin_px + marker_px,
        margin_px:margin_px + marker_px
    ] = marker_img

    out = package_dir / "models/aruco_target/materials/textures/aruco_6x6_250_id23.png"
    out.parent.mkdir(parents=True, exist_ok=True)

    ok = cv2.imwrite(str(out), img)
    if not ok:
        raise RuntimeError(f"Failed to write ArUco marker image to {out}")

    print(f"Saved ArUco marker to: {out}")
    print(f"Dictionary: {dictionary_name}")
    print(f"Marker ID: {marker_id}")
    print(f"Marker size: {marker_px} x {marker_px} px")
    print(f"Image size including margin: {img_w} x {img_h} px")
    print(f"White margin: {margin_px} px")


if __name__ == "__main__":
    main()