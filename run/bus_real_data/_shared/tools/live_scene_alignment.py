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
from typing import Callable


DEFAULT_WORLD_FILE = Path(
    "src/calib_lab/bus_real_data/worlds/"
    "bus_real_data_moving_camera.sdf"
)

DEFAULT_OVERLAY_MODEL_FILE = Path(
    "src/calib_lab/bus_real_data/models/"
    "beintelli_bus_interior_overlay/model.sdf"
)

BASE_BUS_NAME = "beintelli_bus"
OVERLAY_BUS_NAME = "beintelli_bus_interior_overlay"
OVERLAY_LINK_NAME = "interior_overlay_link"

STATIC_CAMERAS = (
    "cam_edge_0",
    "cam_edge_1",
    "cam_edge_3",
    "cam_edge_5",
)


# -----------------------------------------------------------------------------
# Pose math
# -----------------------------------------------------------------------------

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
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy

    return qx, qy, qz, qw


def rotation_matrix(
    roll: float,
    pitch: float,
    yaw: float,
) -> tuple[tuple[float, float, float], ...]:
    cr = math.cos(roll)
    sr = math.sin(roll)
    cp = math.cos(pitch)
    sp = math.sin(pitch)
    cy = math.cos(yaw)
    sy = math.sin(yaw)

    # R = Rz(yaw) @ Ry(pitch) @ Rx(roll)
    return (
        (
            cy * cp,
            cy * sp * sr - sy * cr,
            cy * sp * cr + sy * sr,
        ),
        (
            sy * cp,
            sy * sp * sr + cy * cr,
            sy * sp * cr - cy * sr,
        ),
        (
            -sp,
            cp * sr,
            cp * cr,
        ),
    )


def rotate_vector(
    rotation: tuple[tuple[float, float, float], ...],
    vector: list[float],
) -> list[float]:
    return [
        sum(rotation[row][column] * vector[column] for column in range(3))
        for row in range(3)
    ]


def format_pose(pose: list[float]) -> str:
    return " ".join(f"{value:.9f}" for value in pose)


# -----------------------------------------------------------------------------
# SDF text access
# -----------------------------------------------------------------------------

def find_named_block(
    text: str,
    tag: str,
    name: str,
) -> re.Match[str]:
    pattern = re.compile(
        rf"<{tag}\b[^>]*>.*?</{tag}>",
        flags=re.DOTALL,
    )

    for match in pattern.finditer(text):
        block = match.group(0)

        if tag == "include":
            name_match = re.search(
                r"<name>\s*([^<]+?)\s*</name>",
                block,
                flags=re.DOTALL,
            )

            if name_match and name_match.group(1).strip() == name:
                return match

        else:
            opening = re.match(
                rf"<{tag}\b[^>]*\bname=[\"']([^\"']+)[\"']",
                block,
            )

            if opening and opening.group(1) == name:
                return match

    raise RuntimeError(
        f"Could not find <{tag}> named {name!r}"
    )


def read_pose_from_block(block: str) -> list[float]:
    match = re.search(
        r"<pose>\s*([^<]+?)\s*</pose>",
        block,
        flags=re.DOTALL,
    )

    if match is None:
        raise RuntimeError("No <pose> found in SDF block")

    pose = [
        float(value)
        for value in match.group(1).split()
    ]

    if len(pose) != 6:
        raise RuntimeError(
            f"Expected six pose values, got: {pose}"
        )

    return pose


def replace_pose_in_block(
    block: str,
    pose: list[float],
) -> str:
    updated, count = re.subn(
        r"<pose>\s*[^<]+?\s*</pose>",
        f"<pose>{format_pose(pose)}</pose>",
        block,
        count=1,
        flags=re.DOTALL,
    )

    if count != 1:
        raise RuntimeError(
            "Could not replace <pose> in SDF block"
        )

    return updated


def read_named_pose(
    path: Path,
    tag: str,
    name: str,
) -> list[float]:
    text = path.read_text()
    match = find_named_block(text, tag, name)
    return read_pose_from_block(match.group(0))


def write_named_pose(
    path: Path,
    tag: str,
    name: str,
    pose: list[float],
) -> None:
    text = path.read_text()
    match = find_named_block(text, tag, name)

    old_block = match.group(0)
    new_block = replace_pose_in_block(
        old_block,
        pose,
    )

    updated = (
        text[:match.start()]
        + new_block
        + text[match.end():]
    )

    path.write_text(updated)


# -----------------------------------------------------------------------------
# Gazebo service
# -----------------------------------------------------------------------------

def find_cli() -> str:
    if shutil.which("ign"):
        return "ign"

    if shutil.which("gz"):
        return "gz"

    raise RuntimeError(
        "Neither 'ign' nor 'gz' exists in PATH"
    )


def detect_world_name(cli: str) -> str:
    process = subprocess.run(
        [cli, "service", "-l"],
        capture_output=True,
        text=True,
    )

    matches = re.findall(
        r"/world/([^/\s]+)/set_pose",
        process.stdout + process.stderr,
    )

    if not matches:
        raise RuntimeError(
            "No Gazebo /world/.../set_pose service found. "
            "Is Gazebo running?"
        )

    return matches[0]


def set_entity_pose(
    cli: str,
    world_name: str,
    entity_name: str,
    pose: list[float],
) -> bool:
    x, y, z, roll, pitch, yaw = pose
    qx, qy, qz, qw = rpy_to_quaternion(
        roll,
        pitch,
        yaw,
    )

    request = (
        f'name: "{entity_name}" '
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

    if cli == "ign":
        request_type = "ignition.msgs.Pose"
        response_type = "ignition.msgs.Boolean"
    else:
        request_type = "gz.msgs.Pose"
        response_type = "gz.msgs.Boolean"

    command = [
        cli,
        "service",
        "-s",
        f"/world/{world_name}/set_pose",
        "--reqtype",
        request_type,
        "--reptype",
        response_type,
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

    output = (
        process.stdout
        + process.stderr
    ).lower()

    success = (
        process.returncode == 0
        and (
            "data: true" in output
            or "data: 1" in output
        )
    )

    if not success:
        print()
        print("[ERROR] set_pose failed:")
        print(process.stdout)
        print(process.stderr)

    return success


# -----------------------------------------------------------------------------
# Interactive terminal
# -----------------------------------------------------------------------------

def print_common_help(
    allow_rotation: bool,
) -> None:
    print(
        """
Translation:

  x / X    +X / -X
  y / Y    +Y / -Y
  z / Z    +Z / -Z

Step size:

  1        1 mm
  2        5 mm
  3        10 mm
  4        50 mm
  5        100 mm
"""
    )

    if allow_rotation:
        print(
            """
Rotation:

  r / R    +roll  / -roll
  p / P    +pitch / -pitch
  o / O    +yaw   / -yaw

  [        smaller angle step
  ]        larger angle step
"""
        )

    print(
        """
Other:

  c        show current values
  0        reset to starting values
  Q        save and quit
  q        discard changes and quit
  h        show help
"""
    )


def interactive_loop(
    *,
    allow_rotation: bool,
    status: Callable[[], str],
    change_translation: Callable[[int, float], bool],
    change_rotation: Callable[[int, float], bool] | None,
    reset: Callable[[], bool],
    save: Callable[[], None],
    discard: Callable[[], None],
) -> None:
    translation_step = 0.05
    rotation_step = math.radians(0.5)

    print_common_help(allow_rotation)
    print("[START]", status())

    old_settings = termios.tcgetattr(
        sys.stdin.fileno()
    )

    try:
        tty.setraw(sys.stdin.fileno())

        while True:
            key = sys.stdin.read(1)

            changed = False

            if key == "x":
                changed = change_translation(
                    0,
                    translation_step,
                )
            elif key == "X":
                changed = change_translation(
                    0,
                    -translation_step,
                )

            elif key == "y":
                changed = change_translation(
                    1,
                    translation_step,
                )
            elif key == "Y":
                changed = change_translation(
                    1,
                    -translation_step,
                )

            elif key == "z":
                changed = change_translation(
                    2,
                    translation_step,
                )
            elif key == "Z":
                changed = change_translation(
                    2,
                    -translation_step,
                )

            elif allow_rotation and key == "r":
                changed = change_rotation(
                    0,
                    rotation_step,
                )
            elif allow_rotation and key == "R":
                changed = change_rotation(
                    0,
                    -rotation_step,
                )

            elif allow_rotation and key == "p":
                changed = change_rotation(
                    1,
                    rotation_step,
                )
            elif allow_rotation and key == "P":
                changed = change_rotation(
                    1,
                    -rotation_step,
                )

            elif allow_rotation and key == "o":
                changed = change_rotation(
                    2,
                    rotation_step,
                )
            elif allow_rotation and key == "O":
                changed = change_rotation(
                    2,
                    -rotation_step,
                )

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

            elif allow_rotation and key == "[":
                rotation_step = max(
                    math.radians(0.01),
                    rotation_step / 2.0,
                )
            elif allow_rotation and key == "]":
                rotation_step = min(
                    math.radians(10.0),
                    rotation_step * 2.0,
                )

            elif key == "0":
                changed = reset()

            elif key == "c":
                pass

            elif key == "h":
                termios.tcsetattr(
                    sys.stdin.fileno(),
                    termios.TCSADRAIN,
                    old_settings,
                )
                print_common_help(allow_rotation)
                tty.setraw(sys.stdin.fileno())

            elif key == "Q":
                termios.tcsetattr(
                    sys.stdin.fileno(),
                    termios.TCSADRAIN,
                    old_settings,
                )
                save()
                print("[DONE] Saved and quit")
                return

            elif key == "q":
                termios.tcsetattr(
                    sys.stdin.fileno(),
                    termios.TCSADRAIN,
                    old_settings,
                )
                discard()
                print("[DONE] Discarded and quit")
                return

            elif ord(key) == 3:
                raise KeyboardInterrupt

            sys.stdout.write(
                "\r"
                + status()
                + f" | step={translation_step * 1000:.1f} mm"
                + (
                    f" | angle={math.degrees(rotation_step):.3f}°"
                    if allow_rotation
                    else ""
                )
                + " " * 8
            )
            sys.stdout.flush()

    finally:
        termios.tcsetattr(
            sys.stdin.fileno(),
            termios.TCSADRAIN,
            old_settings,
        )


# -----------------------------------------------------------------------------
# New bus interior mode
# -----------------------------------------------------------------------------

def run_bus_mode(
    world_file: Path,
    overlay_model_file: Path,
    cli: str,
    world_name: str,
) -> None:
    base_world_pose = read_named_pose(
        world_file,
        "include",
        BASE_BUS_NAME,
    )

    overlay_world_pose = read_named_pose(
        world_file,
        "include",
        OVERLAY_BUS_NAME,
    )

    differences = [
        abs(a - b)
        for a, b in zip(
            base_world_pose,
            overlay_world_pose,
        )
    ]

    if max(differences) > 1e-6:
        raise RuntimeError(
            "Overlay world pose differs from old bus world pose.\n"
            f"Old bus: {format_pose(base_world_pose)}\n"
            f"Overlay: {format_pose(overlay_world_pose)}\n"
            "Run the previous pose-normalization fix first."
        )

    start_local_pose = read_named_pose(
        overlay_model_file,
        "link",
        OVERLAY_LINK_NAME,
    )

    if max(abs(value) for value in start_local_pose[3:]) > 1e-6:
        raise RuntimeError(
            "The interior link currently has non-zero rotation. "
            "Bus mode intentionally supports translation only."
        )

    current_local = start_local_pose[:3].copy()

    rotation = rotation_matrix(
        base_world_pose[3],
        base_world_pose[4],
        base_world_pose[5],
    )

    def apply_live() -> bool:
        local_delta = [
            current_local[index]
            - start_local_pose[index]
            for index in range(3)
        ]

        world_delta = rotate_vector(
            rotation,
            local_delta,
        )

        temporary_model_pose = base_world_pose.copy()

        for index in range(3):
            temporary_model_pose[index] += world_delta[index]

        return set_entity_pose(
            cli,
            world_name,
            OVERLAY_BUS_NAME,
            temporary_model_pose,
        )

    def status() -> str:
        return (
            "BUS LOCAL "
            f"x={current_local[0]: .6f} "
            f"y={current_local[1]: .6f} "
            f"z={current_local[2]: .6f} "
            "| rotation=0 0 0"
        )

    def change_translation(
        axis: int,
        amount: float,
    ) -> bool:
        previous = current_local[axis]
        current_local[axis] += amount

        if apply_live():
            return True

        current_local[axis] = previous
        return False

    def reset() -> bool:
        current_local[:] = start_local_pose[:3]
        return apply_live()

    def save() -> None:
        new_local_pose = [
            current_local[0],
            current_local[1],
            current_local[2],
            0.0,
            0.0,
            0.0,
        ]

        # Preserve the canonical shared bus world pose.
        write_named_pose(
            world_file,
            "include",
            OVERLAY_BUS_NAME,
            base_world_pose,
        )

        write_named_pose(
            overlay_model_file,
            "link",
            OVERLAY_LINK_NAME,
            new_local_pose,
        )

        print()
        print("[SAVED] New interior local pose:")
        print(format_pose(new_local_pose))
        print(
            "[IMPORTANT] Restart Gazebo once after saving. "
            "The runtime model was temporarily moved, while "
            "the persisted alignment is stored in the link pose."
        )

    def discard() -> None:
        set_entity_pose(
            cli,
            world_name,
            OVERLAY_BUS_NAME,
            base_world_pose,
        )

    print()
    print("MODE: NEW BUS INTERIOR")
    print("Only the new interior mesh will move.")
    print("Rotation remains exactly 0 0 0.")

    interactive_loop(
        allow_rotation=False,
        status=status,
        change_translation=change_translation,
        change_rotation=None,
        reset=reset,
        save=save,
        discard=discard,
    )


# -----------------------------------------------------------------------------
# ArUco mode
# -----------------------------------------------------------------------------

def marker_name(marker_id: int) -> str:
    if 0 <= marker_id <= 13:
        return f"marker_{marker_id:03d}"

    if marker_id == 14:
        return "aruco_ref_floor_14"

    raise ValueError(
        "Marker id must be between 0 and 14"
    )


def run_aruco_mode(
    world_file: Path,
    cli: str,
    world_name: str,
    marker_id: int,
) -> None:
    name = marker_name(marker_id)

    start_pose = read_named_pose(
        world_file,
        "include",
        name,
    )

    current_pose = start_pose.copy()

    def apply_live() -> bool:
        return set_entity_pose(
            cli,
            world_name,
            name,
            current_pose,
        )

    def status() -> str:
        return (
            f"ARUCO {marker_id:02d} "
            f"x={current_pose[0]: .6f} "
            f"y={current_pose[1]: .6f} "
            f"z={current_pose[2]: .6f} | "
            f"r={math.degrees(current_pose[3]): .2f}° "
            f"p={math.degrees(current_pose[4]): .2f}° "
            f"y={math.degrees(current_pose[5]): .2f}°"
        )

    def change_translation(
        axis: int,
        amount: float,
    ) -> bool:
        previous = current_pose[axis]
        current_pose[axis] += amount

        if apply_live():
            return True

        current_pose[axis] = previous
        return False

    def change_rotation(
        axis: int,
        amount: float,
    ) -> bool:
        pose_index = axis + 3
        previous = current_pose[pose_index]
        current_pose[pose_index] += amount

        if apply_live():
            return True

        current_pose[pose_index] = previous
        return False

    def reset() -> bool:
        current_pose[:] = start_pose
        return apply_live()

    def save() -> None:
        write_named_pose(
            world_file,
            "include",
            name,
            current_pose,
        )

        print()
        print(
            f"[SAVED] Marker {marker_id}: "
            f"{format_pose(current_pose)}"
        )

    def discard() -> None:
        set_entity_pose(
            cli,
            world_name,
            name,
            start_pose,
        )

    print()
    print(f"MODE: ARUCO MARKER {marker_id}")
    print(f"Entity: {name}")

    interactive_loop(
        allow_rotation=True,
        status=status,
        change_translation=change_translation,
        change_rotation=change_rotation,
        reset=reset,
        save=save,
        discard=discard,
    )


# -----------------------------------------------------------------------------
# Static camera rig mode
# -----------------------------------------------------------------------------

def run_rig_mode(
    world_file: Path,
    cli: str,
    world_name: str,
) -> None:
    start_poses = {
        camera: read_named_pose(
            world_file,
            "model",
            camera,
        )
        for camera in STATIC_CAMERAS
    }

    delta = [0.0, 0.0, 0.0]

    def camera_pose(camera: str) -> list[float]:
        pose = start_poses[camera].copy()

        for axis in range(3):
            pose[axis] += delta[axis]

        return pose

    def apply_live() -> bool:
        successful = True

        for camera in STATIC_CAMERAS:
            successful = (
                set_entity_pose(
                    cli,
                    world_name,
                    camera,
                    camera_pose(camera),
                )
                and successful
            )

        return successful

    def status() -> str:
        return (
            "STATIC CAMERA RIG "
            f"dx={delta[0]: .6f} "
            f"dy={delta[1]: .6f} "
            f"dz={delta[2]: .6f}"
        )

    def change_translation(
        axis: int,
        amount: float,
    ) -> bool:
        previous = delta[axis]
        delta[axis] += amount

        if apply_live():
            return True

        delta[axis] = previous
        apply_live()
        return False

    def reset() -> bool:
        delta[:] = [0.0, 0.0, 0.0]
        return apply_live()

    def save() -> None:
        for camera in STATIC_CAMERAS:
            write_named_pose(
                world_file,
                "model",
                camera,
                camera_pose(camera),
            )

        print()
        print(
            "[SAVED] Applied common rig translation:"
        )
        print(
            f"dx={delta[0]:.9f} "
            f"dy={delta[1]:.9f} "
            f"dz={delta[2]:.9f}"
        )
        print(
            "[SAVED] Camera rotations and all "
            "pairwise distances were preserved."
        )

    def discard() -> None:
        for camera in STATIC_CAMERAS:
            set_entity_pose(
                cli,
                world_name,
                camera,
                start_poses[camera],
            )

    print()
    print("MODE: STATIC CAMERA RIG")
    print(
        "Moving together:",
        ", ".join(STATIC_CAMERAS),
    )
    print(
        "Only world X/Y/Z translation is allowed. "
        "Rotations remain unchanged."
    )

    interactive_loop(
        allow_rotation=False,
        status=status,
        change_translation=change_translation,
        change_rotation=None,
        reset=reset,
        save=save,
        discard=discard,
    )


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Live alignment editor for the new bus interior, "
            "individual ArUco markers, or the static camera rig."
        )
    )

    parser.add_argument(
        "--world-file",
        type=Path,
        default=DEFAULT_WORLD_FILE,
    )

    parser.add_argument(
        "--world-name",
        default=None,
        help=(
            "Gazebo world name. By default it is detected "
            "from the running set_pose service."
        ),
    )

    subparsers = parser.add_subparsers(
        dest="mode",
        required=True,
    )

    bus_parser = subparsers.add_parser(
        "bus",
        help="Translate only the new interior mesh.",
    )

    bus_parser.add_argument(
        "--overlay-model-file",
        type=Path,
        default=DEFAULT_OVERLAY_MODEL_FILE,
    )

    aruco_parser = subparsers.add_parser(
        "aruco",
        help="Edit one ArUco marker.",
    )

    aruco_parser.add_argument(
        "--id",
        type=int,
        required=True,
        dest="marker_id",
    )

    subparsers.add_parser(
        "rig",
        help=(
            "Translate all four static cameras as one rigid rig."
        ),
    )

    args = parser.parse_args()

    world_file = args.world_file.resolve()

    if not world_file.is_file():
        raise RuntimeError(
            f"World file not found: {world_file}"
        )

    if not sys.stdin.isatty():
        raise RuntimeError(
            "Run this script in an interactive terminal"
        )

    cli = find_cli()

    world_name = (
        args.world_name
        if args.world_name
        else detect_world_name(cli)
    )

    print(f"[OK] Gazebo CLI: {cli}")
    print(f"[OK] World: {world_name}")

    if args.mode == "bus":
        overlay_model_file = (
            args.overlay_model_file.resolve()
        )

        if not overlay_model_file.is_file():
            raise RuntimeError(
                "Overlay model file not found: "
                f"{overlay_model_file}"
            )

        run_bus_mode(
            world_file,
            overlay_model_file,
            cli,
            world_name,
        )

    elif args.mode == "aruco":
        run_aruco_mode(
            world_file,
            cli,
            world_name,
            args.marker_id,
        )

    elif args.mode == "rig":
        run_rig_mode(
            world_file,
            cli,
            world_name,
        )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[STOPPED]")
    except Exception as exc:
        print(f"\n[ERROR] {exc}")
        raise SystemExit(1)
