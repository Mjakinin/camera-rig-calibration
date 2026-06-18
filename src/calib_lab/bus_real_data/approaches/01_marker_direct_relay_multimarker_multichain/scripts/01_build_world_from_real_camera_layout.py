#!/usr/bin/env python3

from pathlib import Path
import json
import math
import yaml

ROOT = Path(__file__).resolve().parents[1]
TARGET_TRANSFORMS = ROOT / "config" / "target_transforms.json"
INTRINSICS_CONFIG = ROOT / "config" / "camera_intrinsics_by_camera.yaml"
OUT_WORLD = ROOT / "worlds" / "bus_real_data_camera_layout.sdf"

# Old working pose for the BeIntelli bus mesh from the previous Gazebo setup.
BUS_VISUAL_POSE = "-1.8927 25.7211 4.0344 1.5708 0 0"

CAMERA_KEYS = {
    "cam_edge_0": "edge_0_color_optical_frame",
    "cam_edge_1": "edge_1_color_optical_frame",
    "cam_edge_3": "edge_3_color_optical_frame",
    "cam_edge_5": "edge_5_color_optical_frame",
}

# Real base frame to Gazebo bus mesh frame.
# Current interpretation:
#   real y = bus longitudinal axis
#   gazebo x = bus longitudinal axis
# Mapping:
#   gazebo_x = -real_y + offset_x
#   gazebo_y =  real_x + offset_y
#   gazebo_z =  real_z + offset_z
BASE_TO_GAZEBO_YAW = 1.57079632679

# Current manually visually aligned global rig offset.
# Applied equally to all cameras. Relative camera-to-camera geometry is preserved.
CAMERA_LAYOUT_OFFSET_GAZEBO = [1.0, 0.1, 0.5]

# ROS optical frame:
#   +Z forward, +X right, +Y down
# Gazebo camera link convention here:
#   rendered camera looks along link +X
OPTICAL_TO_GAZEBO_CAMERA_LINK_RPY = [0.0, -1.57079632679, 1.57079632679]

# Extra local rotation for debugging. Keep zero unless views are systematically flipped.
CAMERA_EXTRA_LOCAL_RPY = [0.0, 0.0, 0.0]


def mm3(A, B):
    return [[sum(A[i][k] * B[k][j] for k in range(3)) for j in range(3)] for i in range(3)]


def mv3(A, v):
    return [sum(A[i][k] * v[k] for k in range(3)) for i in range(3)]


def rpy_to_R(roll, pitch, yaw):
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)

    Rx = [[1, 0, 0], [0, cr, -sr], [0, sr, cr]]
    Ry = [[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]]
    Rz = [[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]]

    return mm3(mm3(Rz, Ry), Rx)


def R_to_rpy(R):
    pitch = math.asin(max(-1.0, min(1.0, -R[2][0])))

    if abs(math.cos(pitch)) > 1e-8:
        roll = math.atan2(R[2][1], R[2][2])
        yaw = math.atan2(R[1][0], R[0][0])
    else:
        roll = 0.0
        yaw = math.atan2(-R[0][1], R[1][1])

    return [roll, pitch, yaw]


def get_first_tf(data, key):
    if key not in data:
        raise KeyError(f"Missing key in target_transforms.json: {key}")
    return data[key][0]


def camera_model(name, pose, intr):
    pose_txt = " ".join(f"{v:.8f}" for v in pose)
    topic = f"/bus_real_data/{name}/image"

    width = int(intr["width"])
    height = int(intr["height"])
    hfov_rad = math.radians(float(intr["horizontal_fov_deg"]))

    return f"""
    <model name="{name}">
      <static>true</static>
      <pose>{pose_txt}</pose>

      <link name="{name}_link">
        <visual name="{name}_visual">
          <geometry>
            <box>
              <size>0.10 0.06 0.06</size>
            </box>
          </geometry>
          <material>
            <ambient>0.1 0.1 1.0 1</ambient>
            <diffuse>0.1 0.1 1.0 1</diffuse>
          </material>
        </visual>

        <sensor name="{name}_sensor" type="camera">
          <always_on>true</always_on>
          <update_rate>15</update_rate>
          <topic>{topic}</topic>

          <camera>
            <horizontal_fov>{hfov_rad:.10f}</horizontal_fov>
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


def main():
    target_data = json.loads(TARGET_TRANSFORMS.read_text())
    intrinsics = yaml.safe_load(INTRINSICS_CONFIG.read_text())

    R_base_to_gz = rpy_to_R(0.0, 0.0, BASE_TO_GAZEBO_YAW)
    R_opt_to_gz_link = mm3(
        rpy_to_R(*OPTICAL_TO_GAZEBO_CAMERA_LINK_RPY),
        rpy_to_R(*CAMERA_EXTRA_LOCAL_RPY),
    )

    camera_blocks = []

    print("Building bus_real_data camera layout with real camera_info intrinsics")
    print(f"CAMERA_LAYOUT_OFFSET_GAZEBO = {CAMERA_LAYOUT_OFFSET_GAZEBO}")
    print(f"BASE_TO_GAZEBO_YAW = {BASE_TO_GAZEBO_YAW}")
    print()

    for name, key in CAMERA_KEYS.items():
        tf = get_first_tf(target_data, key)
        intr = intrinsics[name]

        xyz_real = [float(v) for v in tf["xyz"].split()]
        rpy_real_optical = [float(v) for v in tf["rpy"].split()]

        R_real_optical = rpy_to_R(*rpy_real_optical)

        xyz_gz_raw = mv3(R_base_to_gz, xyz_real)
        xyz_gz = [a + b for a, b in zip(xyz_gz_raw, CAMERA_LAYOUT_OFFSET_GAZEBO)]

        R_gz_optical = mm3(R_base_to_gz, R_real_optical)
        R_gz_camera_link = mm3(R_gz_optical, R_opt_to_gz_link)
        rpy_gz_camera_link = R_to_rpy(R_gz_camera_link)

        pose = xyz_gz + rpy_gz_camera_link

        print(f"{name}:")
        print("  source frame:       ", key)
        print("  source camera_info: ", intr.get("source_camera_info", "unknown"))
        print("  real xyz:           ", " ".join(f"{v:.6f}" for v in xyz_real))
        print("  gazebo xyz raw:     ", " ".join(f"{v:.6f}" for v in xyz_gz_raw))
        print("  gazebo xyz + offset:", " ".join(f"{v:.6f}" for v in xyz_gz))
        print("  HFOV deg:           ", f"{float(intr['horizontal_fov_deg']):.6f}")
        print("  fx fy:              ", f"{float(intr['fx']):.6f}", f"{float(intr['fy']):.6f}")
        print()

        camera_blocks.append(camera_model(name, pose, intr))

    world = f"""<?xml version="1.0" ?>
<sdf version="1.9">
  <world name="bus_real_data_camera_layout">

    <plugin filename="ignition-gazebo-physics-system"
            name="ignition::gazebo::systems::Physics"/>
    <plugin filename="ignition-gazebo-user-commands-system"
            name="ignition::gazebo::systems::UserCommands"/>
    <plugin filename="ignition-gazebo-scene-broadcaster-system"
            name="ignition::gazebo::systems::SceneBroadcaster"/>
    <plugin filename="ignition-gazebo-sensors-system"
            name="ignition::gazebo::systems::Sensors">
      <render_engine>ogre2</render_engine>
    </plugin>

    <physics name="default_physics" type="ignored">
      <max_step_size>0.01</max_step_size>
      <real_time_factor>1.0</real_time_factor>
    </physics>

    <scene>
      <ambient>0.7 0.7 0.7 1</ambient>
      <background>0.8 0.8 0.8 1</background>
    </scene>

    <light name="sun" type="directional">
      <pose>0 0 10 0 0 0</pose>
      <diffuse>0.8 0.8 0.8 1</diffuse>
      <specular>0.2 0.2 0.2 1</specular>
      <direction>-0.5 0.1 -0.9</direction>
    </light>

    <include>
      <uri>model://beintelli_bus</uri>
      <name>beintelli_bus</name>
      <pose>{BUS_VISUAL_POSE}</pose>
    </include>

{''.join(camera_blocks)}
  </world>
</sdf>
"""

    OUT_WORLD.write_text(world)
    print(f"[OK] wrote {OUT_WORLD}")


if __name__ == "__main__":
    main()
