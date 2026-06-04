#!/usr/bin/env python3

from pathlib import Path
import argparse
import cv2
import numpy as np


def get_aruco_dictionary(name: str):
    if not hasattr(cv2, "aruco"):
        raise RuntimeError("cv2.aruco not available. Install OpenCV contrib with aruco support.")

    if not hasattr(cv2.aruco, name):
        raise RuntimeError(f"Unknown ArUco dictionary: {name}")

    return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, name))


def generate_marker_image(dictionary, marker_id: int, marker_px: int, canvas_px: int):
    """
    Generate a texture where the physical plane is the full white canvas and
    the actual ArUco marker is centered inside it.

    Example:
      plane_size_m  = 0.90
      marker_size_m = 0.60

    Then marker_px / canvas_px should equal 0.60 / 0.90.
    """
    canvas = np.full((canvas_px, canvas_px), 255, dtype=np.uint8)

    if hasattr(cv2.aruco, "generateImageMarker"):
        marker = cv2.aruco.generateImageMarker(dictionary, marker_id, marker_px)
    else:
        marker = np.zeros((marker_px, marker_px), dtype=np.uint8)
        cv2.aruco.drawMarker(dictionary, marker_id, marker_px, marker, 1)

    offset = (canvas_px - marker_px) // 2
    canvas[offset:offset + marker_px, offset:offset + marker_px] = marker

    return canvas


def write_model_config(model_dir: Path, model_name: str):
    (model_dir / "model.config").write_text(f"""<?xml version="1.0"?>
<model>
  <name>{model_name}</name>
  <version>1.0</version>
  <sdf version="1.7">model.sdf</sdf>
  <author>
    <name>Camera Rig Calibration Lab</name>
    <email>none</email>
  </author>
  <description>
    Flat textured ArUco marker model. The plane includes a white quiet-zone board.
  </description>
</model>
""")


def write_model_sdf(model_dir: Path, model_name: str, texture_name: str, plane_size_m: float):
    # The plane lies in the local XY plane with normal +Z.
    # Existing world poses can therefore stay unchanged:
    # - floor markers face upward
    # - side markers are rotated with roll +/-90 deg
    (model_dir / "model.sdf").write_text(f"""<?xml version="1.0"?>
<sdf version="1.7">
  <model name="{model_name}">
    <static>true</static>
    <link name="link">
      <visual name="textured_marker_plane">
        <cast_shadows>false</cast_shadows>
        <pose>0 0 0 0 0 0</pose>
        <geometry>
          <plane>
            <normal>0 0 1</normal>
            <size>{plane_size_m:.6f} {plane_size_m:.6f}</size>
          </plane>
        </geometry>
        <material>
          <ambient>1 1 1 1</ambient>
          <diffuse>1 1 1 1</diffuse>
          <specular>0 0 0 1</specular>
          <emissive>1 1 1 1</emissive>
          <pbr>
            <metal>
              <albedo_map>model://{model_name}/materials/textures/{texture_name}</albedo_map>
              <roughness>1.0</roughness>
              <metalness>0.0</metalness>
            </metal>
          </pbr>
        </material>
      </visual>
    </link>
  </model>
</sdf>
""")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_root", default="src/calib_lab/beintelli_bus_model/models/aruco_individual")
    parser.add_argument("--dictionary", default="DICT_4X4_50")
    parser.add_argument("--first_id", type=int, default=0)
    parser.add_argument("--last_id", type=int, default=9)
    parser.add_argument("--plane_size_m", type=float, default=0.90)
    parser.add_argument("--marker_size_m", type=float, default=0.60)
    parser.add_argument("--canvas_px", type=int, default=900)
    args = parser.parse_args()

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    dictionary = get_aruco_dictionary(args.dictionary)

    marker_px = round(args.canvas_px * args.marker_size_m / args.plane_size_m)
    if marker_px <= 0 or marker_px >= args.canvas_px:
        raise RuntimeError("Invalid marker_px. Check marker_size_m and plane_size_m.")

    print("[INFO] Textured ArUco marker generation")
    print(f"  dictionary:    {args.dictionary}")
    print(f"  ids:           {args.first_id}..{args.last_id}")
    print(f"  plane_size_m:  {args.plane_size_m}")
    print(f"  marker_size_m: {args.marker_size_m}")
    print(f"  canvas_px:     {args.canvas_px}")
    print(f"  marker_px:     {marker_px}")

    for marker_id in range(args.first_id, args.last_id + 1):
        model_name = f"aruco_marker_{marker_id:02d}"
        model_dir = output_root / model_name
        texture_dir = model_dir / "materials" / "textures"
        texture_dir.mkdir(parents=True, exist_ok=True)

        texture_name = f"{model_name}.png"
        texture_path = texture_dir / texture_name

        image = generate_marker_image(dictionary, marker_id, marker_px, args.canvas_px)
        ok = cv2.imwrite(str(texture_path), image)
        if not ok:
            raise RuntimeError(f"Failed to write texture: {texture_path}")

        write_model_config(model_dir, model_name)
        write_model_sdf(model_dir, model_name, texture_name, args.plane_size_m)

        print(f"[OK] {model_name}: {texture_path}")

    print("[DONE] Textured marker models generated.")


if __name__ == "__main__":
    main()
