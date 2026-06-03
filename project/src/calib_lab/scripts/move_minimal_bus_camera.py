#!/usr/bin/env python3

import math
import shutil
import subprocess
import time


WORLD_NAME = "minimal_bus_corridor_world"
MODEL_NAME = "moving_camera"


def get_cmd():
    if shutil.which("ign"):
        return "ign", "ignition.msgs.Pose", "ignition.msgs.Boolean"
    if shutil.which("gz"):
        return "gz", "gz.msgs.Pose", "gz.msgs.Boolean"
    raise RuntimeError("Neither ign nor gz command found")


def wait_for_service(cmd, service_name):
    print(f"Waiting for {service_name}")

    while True:
        result = subprocess.run(
            [cmd, "service", "-l"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        if service_name in result.stdout:
            print("Service found.")
            return

        print("Still waiting...")
        time.sleep(1.0)


def rpy_to_quaternion(roll, pitch, yaw):
    """
    Convert roll, pitch, yaw to quaternion.

    In our Gazebo setup:
    - +X is the corridor direction
    - +Z is upward
    - positive pitch tilts the camera down toward the floor
    """
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)

    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy

    return qx, qy, qz, qw


def send_pose(cmd, req_type, rep_type, service_name, x, y, z, roll, pitch, yaw):
    qx, qy, qz, qw = rpy_to_quaternion(roll, pitch, yaw)

    req = (
        f'name: "{MODEL_NAME}" '
        f'position: {{x: {x} y: {y} z: {z}}} '
        f'orientation: {{x: {qx} y: {qy} z: {qz} w: {qw}}}'
    )

    result = subprocess.run(
        [
            cmd,
            "service",
            "-s",
            service_name,
            "--reqtype",
            req_type,
            "--reptype",
            rep_type,
            "--timeout",
            "1000",
            "--req",
            req,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    if result.stdout.strip():
        print(result.stdout.strip())

    if result.stderr.strip():
        print("ERROR:", result.stderr.strip())


def main():
    cmd, req_type, rep_type = get_cmd()
    service_name = f"/world/{WORLD_NAME}/set_pose"

    wait_for_service(cmd, service_name)

    print("Moving camera along corridor while looking down at the floor.")
    print("Stop with CTRL + C.")

    start_time = time.time()

    # Bewegung entlang des Korridors.
    # x pendelt zwischen -3.4 m und +3.4 m.
    center_x = 0.0
    amplitude = 3.4
    speed = 0.22

    # Kamera bleibt mittig im Korridor.
    y = 0.0

    # Kamera-Höhe. Etwas höher, damit sie sinnvoll auf den Boden schaut.
    z = 1.6

    # Orientierung:
    # yaw = 0 bedeutet Blickrichtung grob entlang +X.
    # pitch = +80 Grad bedeutet stark nach unten Richtung Boden.
    roll = 0.0
    pitch = math.radians(80.0)
    yaw = 0.0

    while True:
        t = time.time() - start_time

        x = center_x + amplitude * math.sin(speed * t)

        print(
            f"Setting {MODEL_NAME}: "
            f"x={x:.2f}, y={y:.2f}, z={z:.2f}, "
            f"pitch_down=80 deg"
        )

        send_pose(
            cmd=cmd,
            req_type=req_type,
            rep_type=rep_type,
            service_name=service_name,
            x=x,
            y=y,
            z=z,
            roll=roll,
            pitch=pitch,
            yaw=yaw,
        )

        time.sleep(0.5)


if __name__ == "__main__":
    main()