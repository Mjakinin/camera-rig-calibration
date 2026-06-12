#!/usr/bin/env python3

from pathlib import Path
import math
import re

ROOT = Path(__file__).resolve().parents[1]
BASE_WORLD = ROOT / "worlds" / "bus_real_data_a4_markers.sdf"
OUT_WORLD = ROOT / "worlds" / "bus_real_data_moving_camera.sdf"

WIDTH = 1280
HEIGHT = 720

# Use real camera-info-like HFOV as first moving-camera approximation.
HFOV_RAD = math.radians(69.1)

# Initial pose:
# x y z roll pitch yaw
# Gazebo convention here: camera renders along local +X.
INITIAL_POSE = "0.000000 0.000000 1.600000 0.000000 0.000000 0.000000"

MOVING_CAMERA_BLOCK = f"""
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
              <width>{WIDTH}</width>
              <height>{HEIGHT}</height>
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

def main():
    text = BASE_WORLD.read_text()

    # Remove old moving camera if script was run before.
    text = re.sub(
        r'\n\s*<model name="moving_calib_camera">.*?</model>\s*\n',
        "\n",
        text,
        flags=re.DOTALL,
    )

    if "</world>" not in text:
        raise RuntimeError("Could not find </world> in base world.")

    text = text.replace("</world>", MOVING_CAMERA_BLOCK + "\n  </world>")
    OUT_WORLD.write_text(text)

    print("[OK] wrote", OUT_WORLD)
    print("[OK] moving camera initial pose:", INITIAL_POSE)

if __name__ == "__main__":
    main()
