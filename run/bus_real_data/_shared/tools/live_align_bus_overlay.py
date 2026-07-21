#!/usr/bin/env python3

from __future__ import annotations

import argparse
import math
import re
import shutil
import subprocess
import sys
import termios
import tty
from pathlib import Path


DEFAULT_WORLD_FILE = Path(
    "src/calib_lab/bus_real_data/worlds/"
    "bus_real_data_moving_camera.sdf"
)

DEFAULT_WORLD_NAME = "bus_real_data_camera_layout"
DEFAULT_MODEL_NAME = "beintelli_bus_interior_overlay"


def rpy_to_quaternion(
    roll: float,
    pitch: float,
    yaw: float,
) -> tuple[float, float, float, float]:
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)

    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + cr * sp * sy
    qz = cr * cp * sy - sr * sp * cy

    # Correct quaternion equations.
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy

    return qx, qy, qz, qw


def find_model_include(
    text: str,
    model_name: str,
) -> tuple[re.Match[str], str]:
    pattern = re.compile(
        r"<include>.*?"
        r"<name>\s*"
        + re.escape(model_name)
        + r"\s*</name>.*?"
        r"</include>",
        flags=re.DOTALL,
    )

    match = pattern.search(text)

    if match is None:
        raise RuntimeError(
            f"Could not find <include> for model {model_name!r}"
        )

    return match, match.group(0)


def read_persisted_pose(
    world_file: Path,
    model_name: str,
) -> list[float]:
    text = world_file.read_text()
    _, block = find_model_include(text, model_name)

    match = re.search(
        r"<pose>\s*([^<]+?)\s*</pose>",
        block,
        flags=re.DOTALL,
    )

    if match is None:
        raise RuntimeError(
            f"No <pose> found in include for {model_name!r}"
        )

    values = [
        float(value)
        for value in match.group(1).split()
    ]

    if len(values) != 6:
        raise RuntimeError(
            f"Expected six pose values, received: {values}"
        )

    return values


def persist_pose(
    world_file: Path,
    model_name: str,
    pose: list[float],
) -> None:
    text = world_file.read_text()
    include_match, block = find_model_include(
        text,
        model_name,
    )

    formatted = (
        f"{pose[0]:.9f} "
        f"{pose[1]:.9f} "
        f"{pose[2]:.9f} "
        f"{pose[3]:.9f} "
        f"{pose[4]:.9f} "
        f"{pose[5]:.9f}"
    )

    new_block, count = re.subn(
        r"<pose>\s*[^<]+?\s*</pose>",
        f"<pose>{formatted}</pose>",
        block,
        count=1,
        flags=re.DOTALL,
    )

    if count != 1:
        raise RuntimeError(
            "Could not replace overlay include pose"
        )

    updated = (
        text[:include_match.start()]
        + new_block
        + text[include_match.end():]
    )

    world_file.write_text(updated)

    print()
    print("[SAVED] World file updated:")
    print(f"        {world_file}")
    print(f"[SAVED] <pose>{formatted}</pose>")
    print()


def find_gazebo_cli() -> str:
    for candidate in ("ign", "gz"):
        path = shutil.which(candidate)

        if path is not None:
            return candidate

    raise RuntimeError(
        "Neither 'ign' nor 'gz' was found in PATH"
    )


def set_live_pose(
    cli: str,
    world_name: str,
    model_name: str,
    pose: list[float],
) -> bool:
    x, y, z, roll, pitch, yaw = pose
    qx, qy, qz, qw = rpy_to_quaternion(
        roll,
        pitch,
        yaw,
    )

    request = (
        f'name: "{model_name}" '
        f'position {{ '
        f'x: {x:.12f} '
        f'y: {y:.12f} '
        f'z: {z:.12f} '
        f'}} '
        f'orientation {{ '
        f'x: {qx:.12f} '
        f'y: {qy:.12f} '
        f'z: {qz:.12f} '
        f'w: {qw:.12f} '
        f'}}'
    )

    command = [
        cli,
        "service",
        "-s",
        f"/world/{world_name}/set_pose",
        "--reqtype",
        "ignition.msgs.Pose",
        "--reptype",
        "ignition.msgs.Boolean",
        "--timeout",
        "2000",
        "--req",
        request,
    ]

    process = subprocess.run(
        command,
        capture_output=True,
        text=True,
    )

    output = process.stdout + process.stderr

    success = (
        process.returncode == 0
        and (
            "data: true" in output.lower()
            or "data: 1" in output.lower()
        )
    )

    if not success:
        print()
        print("[ERROR] Gazebo rejected set_pose")
        print(output.strip())
        print()

    return success


def pose_line(
    pose: list[float],
    translation_step: float,
    rotation_step: float,
) -> str:
    return (
        f"x={pose[0]: .6f}  "
        f"y={pose[1]: .6f}  "
        f"z={pose[2]: .6f}  |  "
        f"r={math.degrees(pose[3]): .3f}°  "
        f"p={math.degrees(pose[4]): .3f}°  "
        f"y={math.degrees(pose[5]): .3f}°  |  "
        f"step={translation_step * 1000:.1f} mm  "
        f"angle={math.degrees(rotation_step):.3f}°"
    )


def print_help() -> None:
    print(
        """
LIVE BUS INTERIOR ALIGNMENT
===========================

Translation in Gazebo world coordinates:

  x / X    +X / -X
  y / Y    +Y / -Y
  z / Z    +Z / -Z

Rotation:

  u / U    +roll  / -roll
  i / I    +pitch / -pitch
  o / O    +yaw   / -yaw

Translation step:

  1        1 mm
  2        5 mm
  3        10 mm
  4        50 mm
  5        100 mm

Rotation step:

  [        smaller
  ]        larger

Other:

  c        print current pose
  0        reset to pose from script start
  s        save current pose into the world SDF
  Q        save and quit
  q        quit without saving
  h        show this help

Important:
The movements are Gazebo WORLD axes, not screen directions.
Test one axis with a large step first. If it moves in the
wrong direction, use the uppercase opposite key.
"""
    )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--world-file",
        type=Path,
        default=DEFAULT_WORLD_FILE,
    )

    parser.add_argument(
        "--world-name",
        default=DEFAULT_WORLD_NAME,
    )

    parser.add_argument(
        "--model-name",
        default=DEFAULT_MODEL_NAME,
    )

    args = parser.parse_args()

    world_file = args.world_file.resolve()

    if not world_file.is_file():
        raise RuntimeError(
            f"World file does not exist: {world_file}"
        )

    if not sys.stdin.isatty():
        raise RuntimeError(
            "This script must be run in an interactive terminal"
        )

    cli = find_gazebo_cli()

    pose = read_persisted_pose(
        world_file,
        args.model_name,
    )

    start_pose = pose.copy()

    translation_step = 0.05
    rotation_step = math.radians(0.5)

    print_help()
    print("[START]", pose_line(
        pose,
        translation_step,
        rotation_step,
    ))

    if not set_live_pose(
        cli,
        args.world_name,
        args.model_name,
        pose,
    ):
        raise RuntimeError(
            "Initial set_pose failed. Is Gazebo running?"
        )

    old_terminal_settings = termios.tcgetattr(
        sys.stdin.fileno()
    )

    try:
        tty.setraw(sys.stdin.fileno())

        while True:
            key = sys.stdin.read(1)

            old_pose = pose.copy()
            changed = False

            if key == "x":
                pose[0] += translation_step
                changed = True
            elif key == "X":
                pose[0] -= translation_step
                changed = True

            elif key == "y":
                pose[1] += translation_step
                changed = True
            elif key == "Y":
                pose[1] -= translation_step
                changed = True

            elif key == "z":
                pose[2] += translation_step
                changed = True
            elif key == "Z":
                pose[2] -= translation_step
                changed = True

            elif key == "u":
                pose[3] += rotation_step
                changed = True
            elif key == "U":
                pose[3] -= rotation_step
                changed = True

            elif key == "i":
                pose[4] += rotation_step
                changed = True
            elif key == "I":
                pose[4] -= rotation_step
                changed = True

            elif key == "o":
                pose[5] += rotation_step
                changed = True
            elif key == "O":
                pose[5] -= rotation_step
                changed = True

            elif key == "1":
                translation_step = 0.001
            elif key == "2":
                translation_step = 0.005
            elif key == "3":
                translation_step = 0.010
            elif key == "4":
                translation_step = 0.050
            elif key == "5":
                translation_step = 0.100

            elif key == "[":
                rotation_step = max(
                    math.radians(0.01),
                    rotation_step / 2.0,
                )
            elif key == "]":
                rotation_step = min(
                    math.radians(10.0),
                    rotation_step * 2.0,
                )

            elif key == "0":
                pose = start_pose.copy()
                changed = True

            elif key == "c":
                pass

            elif key == "h":
                termios.tcsetattr(
                    sys.stdin.fileno(),
                    termios.TCSADRAIN,
                    old_terminal_settings,
                )
                print_help()
                tty.setraw(sys.stdin.fileno())

            elif key == "s":
                termios.tcsetattr(
                    sys.stdin.fileno(),
                    termios.TCSADRAIN,
                    old_terminal_settings,
                )
                persist_pose(
                    world_file,
                    args.model_name,
                    pose,
                )
                tty.setraw(sys.stdin.fileno())

            elif key == "Q":
                termios.tcsetattr(
                    sys.stdin.fileno(),
                    termios.TCSADRAIN,
                    old_terminal_settings,
                )
                persist_pose(
                    world_file,
                    args.model_name,
                    pose,
                )
                print("[DONE] Saved and quit")
                return

            elif key == "q":
                termios.tcsetattr(
                    sys.stdin.fileno(),
                    termios.TCSADRAIN,
                    old_terminal_settings,
                )
                print()
                print("[DONE] Quit without saving")
                return

            elif ord(key) == 3:
                raise KeyboardInterrupt

            if changed:
                if not set_live_pose(
                    cli,
                    args.world_name,
                    args.model_name,
                    pose,
                ):
                    pose = old_pose

            # Raw-terminal-safe single-line status.
            sys.stdout.write(
                "\r"
                + pose_line(
                    pose,
                    translation_step,
                    rotation_step,
                )
                + " " * 8
            )
            sys.stdout.flush()

    finally:
        termios.tcsetattr(
            sys.stdin.fileno(),
            termios.TCSADRAIN,
            old_terminal_settings,
        )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[STOPPED]")
    except Exception as exc:
        print(f"\n[ERROR] {exc}")
        raise SystemExit(1)
