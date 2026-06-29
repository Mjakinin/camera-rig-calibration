#!/usr/bin/env python3
import argparse
import csv
import curses
import json
import math
import shutil
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image


TOPIC = "/bus_real_data/moving_calib_camera/image"


def rpy_to_quat(roll, pitch, yaw):
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


def set_pose(world, name, pose):
    x, y, z, roll, pitch, yaw = pose
    qx, qy, qz, qw = rpy_to_quat(roll, pitch, yaw)

    req = (
        f'name: "{name}" '
        f'position {{x: {x} y: {y} z: {z}}} '
        f'orientation {{x: {qx} y: {qy} z: {qz} w: {qw}}}'
    )

    cmd = [
        "ign", "service",
        "-s", f"/world/{world}/set_pose",
        "--reqtype", "ignition.msgs.Pose",
        "--reptype", "ignition.msgs.Boolean",
        "--timeout", "1000",
        "--req", req,
    ]

    proc = subprocess.run(cmd, capture_output=True, text=True)
    return proc.returncode == 0 and "data: true" in (proc.stdout + proc.stderr)


def image_msg_to_bgr(msg):
    h = msg.height
    w = msg.width
    enc = msg.encoding.lower()
    data = np.frombuffer(msg.data, dtype=np.uint8)

    if enc in ["rgb8", "bgr8"]:
        arr = data.reshape(h, msg.step)[:, :w * 3].reshape(h, w, 3).copy()
        if enc == "rgb8":
            return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        return arr

    if enc in ["rgba8", "bgra8"]:
        arr = data.reshape(h, msg.step)[:, :w * 4].reshape(h, w, 4).copy()
        if enc == "rgba8":
            return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
        return cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)

    if enc in ["mono8", "8uc1"]:
        arr = data.reshape(h, msg.step)[:, :w].reshape(h, w).copy()
        return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)

    raise RuntimeError(f"Unsupported image encoding: {msg.encoding}")


class ImageGrabber(Node):
    def __init__(self):
        super().__init__("manual_moving_camera_drive_capture")
        self.last_msg = None
        self.counter = 0
        self.sub = self.create_subscription(Image, TOPIC, self.cb, 10)

    def cb(self, msg):
        self.last_msg = msg
        self.counter += 1


def save_outputs(out_dir, records, trans_step, rot_step):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "route_commanded.csv"
    fields = ["frame", "segment", "x", "y", "z", "roll", "pitch", "yaw", "image", "set_pose_ok"]

    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in records:
            w.writerow({k: r.get(k, "") for k in fields})

    frames = []
    keyframes = []
    for i, r in enumerate(records):
        pose = [
            float(r["x"]), float(r["y"]), float(r["z"]),
            float(r["roll"]), float(r["pitch"]), float(r["yaw"])
        ]
        frames.append({
            "frame": int(r["frame"]),
            "segment": int(r["segment"]),
            "x": pose[0],
            "y": pose[1],
            "z": pose[2],
            "roll": pose[3],
            "pitch": pose[4],
            "yaw": pose[5],
        })
        keyframes.append({
            "name": f"manual_{i:04d}",
            "pose": pose,
        })

    (out_dir / "manual_route_interpolated.json").write_text(json.dumps({
        "description": "Manual moving-camera route captured with keyboard drive tool.",
        "pose_format": "x y z roll pitch yaw",
        "num_frames": len(frames),
        "translation_step_m": trans_step,
        "rotation_step_rad": rot_step,
        "frames": frames,
    }, indent=2) + "\n")

    (out_dir / "manual_route_keyframes.json").write_text(json.dumps({
        "description": "Manual moving-camera route keyframes captured with keyboard drive tool.",
        "pose_format": "x y z roll pitch yaw",
        "keyframes": keyframes,
    }, indent=2) + "\n")

    (out_dir / "README.txt").write_text(
        "Manual moving-camera drive capture\n"
        "==================================\n\n"
        f"Topic: {TOPIC}\n"
        f"Frames captured: {len(records)}\n\n"
        "Files:\n"
        "- images/frame_XXXXXX.png\n"
        "- route_commanded.csv\n"
        "- manual_route_interpolated.json\n"
        "- manual_route_keyframes.json\n"
    )


def draw_screen(stdscr, pose, frame_idx, records, trans_step, rot_step, auto_record, last_status):
    x, y, z, roll, pitch, yaw = pose
    stdscr.erase()
    stdscr.addstr(0, 0, "Manual Moving-Camera Drive + Capture")
    stdscr.addstr(2, 0, "W/S forward/back | A/D strafe | Q/E down/up | arrows yaw/pitch | Z/X roll")
    stdscr.addstr(3, 0, "SPACE capture | R auto-record on movement | +/- move step | [/ ] rot step | ESC quit/save")
    stdscr.addstr(5, 0, f"pose: x={x:.3f} y={y:.3f} z={z:.3f} roll={roll:.3f} pitch={pitch:.3f} yaw={yaw:.3f}")
    stdscr.addstr(6, 0, f"next frame: {frame_idx:06d} | captured: {len(records)} | auto_record={auto_record}")
    stdscr.addstr(7, 0, f"trans_step={trans_step:.3f} m | rot_step={rot_step:.3f} rad")
    stdscr.addstr(9, 0, f"status: {last_status[:140]}")
    stdscr.refresh()


def capture_frame(node, out_dir, frame_idx, pose, set_pose_ok, records):
    if node.last_msg is None:
        return False, "no image received yet"

    img_dir = Path(out_dir) / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    bgr = image_msg_to_bgr(node.last_msg)
    img_path = img_dir / f"frame_{frame_idx:06d}.png"
    cv2.imwrite(str(img_path), bgr)

    x, y, z, roll, pitch, yaw = pose
    records.append({
        "frame": frame_idx,
        "segment": max(0, frame_idx - 1),
        "x": x,
        "y": y,
        "z": z,
        "roll": roll,
        "pitch": pitch,
        "yaw": yaw,
        "image": str(img_path),
        "set_pose_ok": set_pose_ok,
    })

    return True, f"captured {img_path}"


def run_curses(stdscr, args, node):
    curses.curs_set(0)
    stdscr.nodelay(True)
    stdscr.keypad(True)

    out_dir = Path(args.out)
    if args.clean and out_dir.exists():
        shutil.rmtree(out_dir)
    (out_dir / "images").mkdir(parents=True, exist_ok=True)

    pose = [args.x, args.y, args.z, args.roll, args.pitch, args.yaw]
    trans_step = args.trans_step
    rot_step = args.rot_step

    records = []
    frame_idx = 0
    auto_record = False
    last_status = "starting"
    set_pose_ok = set_pose(args.world, args.name, pose)

    last_draw = 0.0

    while True:
        rclpy.spin_once(node, timeout_sec=0.01)

        now = time.time()
        if now - last_draw > 0.05:
            draw_screen(stdscr, pose, frame_idx, records, trans_step, rot_step, auto_record, last_status)
            last_draw = now

        ch = stdscr.getch()
        if ch == -1:
            time.sleep(0.01)
            continue

        moved = False
        captured = False

        # ESC
        if ch == 27:
            break

        # Normalize ASCII keys
        key = chr(ch).lower() if 0 <= ch < 256 else ""

        x, y, z, roll, pitch, yaw = pose

        # Local movement relative to yaw.
        fwd = np.array([math.cos(yaw), math.sin(yaw)], dtype=float)
        left = np.array([math.cos(yaw + math.pi / 2), math.sin(yaw + math.pi / 2)], dtype=float)

        if key == "w":
            x += trans_step * fwd[0]
            y += trans_step * fwd[1]
            moved = True
        elif key == "s":
            x -= trans_step * fwd[0]
            y -= trans_step * fwd[1]
            moved = True
        elif key == "a":
            x += trans_step * left[0]
            y += trans_step * left[1]
            moved = True
        elif key == "d":
            x -= trans_step * left[0]
            y -= trans_step * left[1]
            moved = True
        elif key == "e":
            z += trans_step
            moved = True
        elif key == "q":
            z -= trans_step
            moved = True
        elif ch == curses.KEY_LEFT:
            yaw += rot_step
            moved = True
        elif ch == curses.KEY_RIGHT:
            yaw -= rot_step
            moved = True
        elif ch == curses.KEY_UP:
            pitch -= rot_step
            moved = True
        elif ch == curses.KEY_DOWN:
            pitch += rot_step
            moved = True
        elif key == "z":
            roll += rot_step
            moved = True
        elif key == "x":
            roll -= rot_step
            moved = True
        elif key in ["+", "="]:
            trans_step *= 1.25
            last_status = f"trans_step={trans_step:.3f}"
        elif key in ["-", "_"]:
            trans_step = max(0.01, trans_step / 1.25)
            last_status = f"trans_step={trans_step:.3f}"
        elif key == "]":
            rot_step *= 1.25
            last_status = f"rot_step={rot_step:.3f}"
        elif key == "[":
            rot_step = max(0.005, rot_step / 1.25)
            last_status = f"rot_step={rot_step:.3f}"
        elif key == "r":
            auto_record = not auto_record
            last_status = f"auto_record={auto_record}"
        elif ch == ord(" "):
            ok, msg = capture_frame(node, out_dir, frame_idx, pose, set_pose_ok, records)
            last_status = msg
            if ok:
                frame_idx += 1
            captured = True

        if moved:
            pose = [x, y, z, roll, pitch, yaw]
            set_pose_ok = set_pose(args.world, args.name, pose)
            last_status = f"moved, set_pose_ok={set_pose_ok}"

            if auto_record and not captured:
                ok, msg = capture_frame(node, out_dir, frame_idx, pose, set_pose_ok, records)
                last_status = msg
                if ok:
                    frame_idx += 1

    save_outputs(out_dir, records, trans_step, rot_step)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/bus_manual_route")
    ap.add_argument("--world", default="bus_real_data_camera_layout")
    ap.add_argument("--name", default="moving_calib_camera")
    ap.add_argument("--clean", action="store_true")

    ap.add_argument("--x", type=float, default=-4.5)
    ap.add_argument("--y", type=float, default=0.0)
    ap.add_argument("--z", type=float, default=2.2)
    ap.add_argument("--roll", type=float, default=0.0)
    ap.add_argument("--pitch", type=float, default=0.30)
    ap.add_argument("--yaw", type=float, default=0.0)

    ap.add_argument("--trans-step", type=float, default=0.08)
    ap.add_argument("--rot-step", type=float, default=0.04)
    args = ap.parse_args()

    rclpy.init()
    node = ImageGrabber()

    try:
        curses.wrapper(run_curses, args, node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

    print("[OK] wrote:", args.out)
    print("[OK] images:", Path(args.out) / "images")
    print("[OK] route:", Path(args.out) / "manual_route_interpolated.json")


if __name__ == "__main__":
    main()
