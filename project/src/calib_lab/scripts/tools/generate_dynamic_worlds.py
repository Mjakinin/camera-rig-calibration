#!/usr/bin/env python3

from pathlib import Path
import argparse
import csv
import math
import re


POSE_PATTERN = re.compile(
    r'(<include>\s*<uri>model://[^<]+</uri>\s*<name>calibration_target</name>\s*<pose>)([^<]+)(</pose>)',
    re.MULTILINE
)


def replace_resolution(text: str, width: int, height: int) -> str:
    text = re.sub(r"<width>\d+</width>", f"<width>{width}</width>", text)
    text = re.sub(r"<height>\d+</height>", f"<height>{height}</height>", text)
    return text


def ensure_user_commands_plugin(text: str) -> str:
    if "UserCommands" in text:
        return text

    plugin = '''
    <plugin
      filename="ignition-gazebo-user-commands-system"
      name="ignition::gazebo::systems::UserCommands">
    </plugin>
'''
    return text.replace("</world>", plugin + "\n  </world>", 1)


def scenario_rows():
    rows = []

    for x in [1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8]:
        rows.append((f"dist_{str(x).replace('.', '_')}m", "distance", x, 0.0, 1.0, 0.0, 0.0, 0.0))

    for deg in [0, 10, 20, 30, 35, 40, 45, 50]:
        rows.append((f"yaw_{deg}deg", "yaw", 1.8, 0.0, 1.0, 0.0, 0.0, math.radians(deg)))

    for y in [-0.6, -0.4, -0.2, 0.2, 0.4, 0.6]:
        tag = "left" if y < 0 else "right"
        val = str(abs(y)).replace(".", "_")
        rows.append((f"shift_{tag}_{val}m", "shift", 1.8, y, 1.0, 0.0, 0.0, 0.0))

    for z in [0.6, 0.8, 1.0, 1.2, 1.4]:
        rows.append((f"height_{str(z).replace('.', '_')}m", "height", 1.8, 0.0, z, 0.0, 0.0, 0.0))

    rows += [
        ("far_2_4m_yaw_20deg", "mixed", 2.4, 0.0, 1.0, 0.0, 0.0, math.radians(20)),
        ("far_2_4m_yaw_30deg", "mixed", 2.4, 0.0, 1.0, 0.0, 0.0, math.radians(30)),
        ("close_1_4m_yaw_10deg", "mixed", 1.4, 0.0, 1.0, 0.0, 0.0, math.radians(10)),
        ("shift_left_0_2m_yaw_20deg", "mixed", 1.8, -0.2, 1.0, 0.0, 0.0, math.radians(20)),
        ("shift_right_0_2m_yaw_20deg", "mixed", 1.8, 0.2, 1.0, 0.0, 0.0, math.radians(20)),
    ]

    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True, choices=["checkerboard", "aruco", "charuco"])
    parser.add_argument("--resolution", required=True, choices=["res320x240", "res640x480"])
    parser.add_argument("--target_uri", default=None)
    parser.add_argument("--base_world", default="src/calib_lab/worlds/minimal_calib_world.sdf")
    args = parser.parse_args()

    width, height = {
        "res320x240": (320, 240),
        "res640x480": (640, 480),
    }[args.resolution]

    target_uri = args.target_uri
    if target_uri is None:
        target_uri = {
            "checkerboard": "model://checkerboard_target",
            "aruco": "model://aruco_target",
            "charuco": "model://charuco_target",
        }[args.method]

    base_path = Path(args.base_world)
    out_dir = Path("src/calib_lab/worlds/dynamic")
    out_dir.mkdir(parents=True, exist_ok=True)

    text = base_path.read_text()
    world_name = f"dynamic_{args.method}_{args.resolution}"

    text = re.sub(
        r'<world name="[^"]+">',
        f'<world name="{world_name}">',
        text,
        count=1,
    )

    text = re.sub(
        r'<uri>model://[^<]+</uri>\s*<name>calibration_target</name>',
        f'<uri>{target_uri}</uri>\n      <name>calibration_target</name>',
        text,
        count=1,
    )

    text, count = POSE_PATTERN.subn(
        r'\g<1>1.800 0.000 1.000 0.000 0.000 0.000000\g<3>',
        text,
    )

    if count != 1:
        raise RuntimeError(f"Could not replace target pose. replacements={count}")

    text = replace_resolution(text, width, height)
    text = ensure_user_commands_plugin(text)

    world_path = out_dir / f"{args.method}_{args.resolution}.sdf"
    world_path.write_text(text)

    pose_csv = out_dir / "scenario_poses.csv"

    with pose_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["scenario", "group", "x", "y", "z", "roll", "pitch", "yaw"])
        writer.writerows(scenario_rows())

    print(f"created world: {world_path}")
    print(f"world name: {world_name}")
    print(f"target uri: {target_uri}")
    print(f"resolution: {width}x{height}")
    print(f"created poses: {pose_csv}")


if __name__ == "__main__":
    main()
