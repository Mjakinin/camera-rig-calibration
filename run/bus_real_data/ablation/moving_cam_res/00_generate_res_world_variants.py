#!/usr/bin/env python3

import json
import math
import re
from pathlib import Path


BASE_WORLD = Path("src/calib_lab/bus_real_data/worlds/bus_real_data_a4_markers.sdf")
WORLD_OUT_ROOT = Path("src/calib_lab/bus_real_data/worlds/ablation/moving_cam_res")
RESULT_OUT_ROOT = Path("results/bus_real_data/ablation/moving_cam_res/00_world_variants")

HFOV_DEG = 69.1
HFOV_RAD = math.radians(HFOV_DEG)

INITIAL_POSE = "0.000000 0.000000 1.600000 0.000000 0.000000 0.000000"

PROFILES = {
    "res_320x180_extreme": (320, 180),
    "res_640x360": (640, 360),
    "res_960x540": (960, 540),
    "res_1280x720_baseline": (1280, 720),
    "res_1920x1080": (1920, 1080),
}


def moving_camera_block(width: int, height: int) -> str:
    return f"""
    <model name="moving_calib_camera">
      <static>true</static>
      <pose>{INITIAL_POSE}</pose>

      <link name="moving_calib_camera_link">
        <visual name="moving_calib_camera_visual">
          <geometry>
            <box>
              <size>0.12 0.07 0.07</size>
            </box>
          </geometry>
          <material>
            <ambient>1.0 0.2 0.1 1</ambient>
            <diffuse>1.0 0.2 0.1 1</diffuse>
          </material>
        </visual>

        <sensor name="moving_calib_camera_sensor" type="camera">
          <always_on>true</always_on>
          <update_rate>15</update_rate>
          <topic>/bus_real_data/moving_calib_camera/image</topic>

          <camera>
            <horizontal_fov>{HFOV_RAD:.10f}</horizontal_fov>
            <image>
              <width>{width}</width>
              <height>{height}</height>
              <format>R8G8B8</format>
            </image>
            <clip>
              <near>0.05</near>
              <far>100.0</far>
            </clip>
          </camera>
        </sensor>
      </link>
    </model>
"""


def make_camera_info(width: int, height: int):
    fx = width / (2.0 * math.tan(HFOV_RAD / 2.0))
    fy = fx
    cx = width / 2.0
    cy = height / 2.0

    K = [
        fx, 0.0, cx,
        0.0, fy, cy,
        0.0, 0.0, 1.0,
    ]

    P = [
        fx, 0.0, cx, 0.0,
        0.0, fy, cy, 0.0,
        0.0, 0.0, 1.0, 0.0,
    ]

    vfov_deg = math.degrees(2.0 * math.atan(height / (2.0 * fy)))

    return {
        "camera_name": "moving_calib_camera",
        "width": width,
        "height": height,
        "image_width": width,
        "image_height": height,
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
        "K": K,
        "k": K,
        "P": P,
        "p": P,
        "distortion": [0.0, 0.0, 0.0, 0.0, 0.0],
        "D": [0.0, 0.0, 0.0, 0.0, 0.0],
        "horizontal_fov_deg": HFOV_DEG,
        "vertical_fov_deg": vfov_deg,
        "model": "synthetic_pinhole_zero_distortion",
    }


def main():
    WORLD_OUT_ROOT.mkdir(parents=True, exist_ok=True)
    RESULT_OUT_ROOT.mkdir(parents=True, exist_ok=True)

    base = BASE_WORLD.read_text()

    base = re.sub(
        r'\n\s*<model name="moving_calib_camera">.*?</model>\s*\n',
        "\n",
        base,
        flags=re.DOTALL,
    )

    if "</world>" not in base:
        raise RuntimeError("Could not find </world> in base world.")

    for variant, (width, height) in PROFILES.items():
        world_text = base.replace("</world>", moving_camera_block(width, height) + "\n  </world>")

        world_path = WORLD_OUT_ROOT / f"bus_real_data_moving_camera_{variant}.sdf"
        world_path.write_text(world_text)

        variant_result = RESULT_OUT_ROOT / variant
        variant_result.mkdir(parents=True, exist_ok=True)

        camera_info = make_camera_info(width, height)
        (variant_result / "moving_calib_camera.json").write_text(json.dumps(camera_info, indent=2) + "\n")

        metadata = {
            "ablation_type": "moving_cam_res",
            "variant": variant,
            "world_file": str(world_path),
            "changed_camera": "moving_calib_camera",
            "static_cameras_unchanged": True,
            "moving_camera_width": width,
            "moving_camera_height": height,
            "moving_camera_hfov_deg": HFOV_DEG,
            "moving_camera_initial_pose": INITIAL_POSE,
            "trajectory_unchanged": True,
            "world_geometry_unchanged": True,
            "lighting_unchanged": True,
            "note": "Only moving-camera render resolution is changed. FOV remains fixed.",
        }
        (variant_result / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

        print(f"[OK] {variant}")
        print(f"     world:       {world_path}")
        print(f"     camera_info: {variant_result / 'moving_calib_camera.json'}")

    print()
    print("[OK] generated moving-camera resolution world variants")


if __name__ == "__main__":
    main()
