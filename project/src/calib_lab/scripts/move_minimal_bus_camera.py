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


def send_pose(cmd, req_type, rep_type, service_name, x, y, z, yaw):
    qz = math.sin(yaw / 2.0)
    qw = math.cos(yaw / 2.0)

    req = (
        f'name: "{MODEL_NAME}" '
        f'position: {{x: {x} y: {y} z: {z}}} '
        f'orientation: {{x: 0 y: 0 z: {qz} w: {qw}}}'
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

    print("Moving camera along bus corridor with debug output.")
    print("Stop with CTRL + C.")

    start_time = time.time()

    # Bewegung entlang der x-Achse:
    # x pendelt zwischen -3.2 m und +3.2 m.
    center_x = 0.0
    amplitude = 3.2
    speed = 0.25

    y = 0.0
    z = 1.3

    # Kamera schaut konstant in +X-Richtung.
    yaw = 0.0

    while True:
        t = time.time() - start_time

        x = center_x + amplitude * math.sin(speed * t)

        print(f"Setting {MODEL_NAME}: x={x:.2f}, y={y:.2f}, z={z:.2f}")
        send_pose(cmd, req_type, rep_type, service_name, x, y, z, yaw)

        time.sleep(0.5)


if __name__ == "__main__":
    main()