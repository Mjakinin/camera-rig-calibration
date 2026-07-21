#!/usr/bin/env python3
import argparse
import csv
import curses
import json
import math
import os
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
    cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
    cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
    cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)

    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    return qx, qy, qz, qw


def pose_command(world, name, pose):
    x, y, z, roll, pitch, yaw = pose
    qx, qy, qz, qw = rpy_to_quat(roll, pitch, yaw)

    req = (
        f'name: "{name}" '
        f'position {{x: {x} y: {y} z: {z}}} '
        f'orientation {{x: {qx} y: {qy} z: {qz} w: {qw}}}'
    )

    return [
        "ign", "service",
        "-s", f"/world/{world}/set_pose",
        "--reqtype", "ignition.msgs.Pose",
        "--reptype", "ignition.msgs.Boolean",
        "--timeout", "200",
        "--req", req,
    ]


def send_pose_async(world, name, pose):
    return subprocess.Popen(
        pose_command(world, name, pose),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def image_msg_to_bgr(msg):
    h, w = msg.height, msg.width
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

    raise RuntimeError(f"Unsupported encoding: {msg.encoding}")


class ImageGrabber(Node):
    def __init__(self):
        super().__init__("manual_drive_fast_capture")
        self.last_msg = None
        self.counter = 0
        self.create_subscription(Image, TOPIC, self.cb, 10)

    def cb(self, msg):
        self.last_msg = msg
        self.counter += 1


def capture_frame(node, out_dir, frame_idx, pose, records):
    if node.last_msg is None:
        return False, "no image yet"

    img_dir = Path(out_dir) / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    img = image_msg_to_bgr(node.last_msg)
    img_path = img_dir / f"frame_{frame_idx:06d}.png"
    cv2.imwrite(str(img_path), img)

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
        "set_pose_ok": "async",
    })
    return True, f"captured frame_{frame_idx:06d}.png"


def save_route(out_dir, records):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    fields = ["frame", "segment", "x", "y", "z", "roll", "pitch", "yaw", "image", "set_pose_ok"]
    with (out / "route_commanded.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in records:
            w.writerow({k: r.get(k, "") for k in fields})

    frames = []
    keyframes = []
    for i, r in enumerate(records):
        pose = [
            float(r["x"]), float(r["y"]), float(r["z"]),
            float(r["roll"]), float(r["pitch"]), float(r["yaw"]),
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

    (out / "manual_route_interpolated.json").write_text(json.dumps({
        "description": "Manual keyboard route, already captured as frames.",
        "pose_format": "x y z roll pitch yaw",
        "num_frames": len(frames),
        "frames": frames,
    }, indent=2) + "\n")

    (out / "manual_route_keyframes.json").write_text(json.dumps({
        "description": "Manual keyboard route keyframes.",
        "pose_format": "x y z roll pitch yaw",
        "keyframes": keyframes,
    }, indent=2) + "\n")

    (out / "README.txt").write_text(
        "Fast manual moving-camera capture\n"
        "=================================\n\n"
        f"Frames captured: {len(records)}\n"
        f"Topic: {TOPIC}\n"
    )


def draw(stdscr, pose, step, rot, send_rate, auto_record, records, status):
    x, y, z, roll, pitch, yaw = pose
    stdscr.erase()
    stdscr.addstr(0, 0, "FAST manual drive capture")
    stdscr.addstr(2, 0, "W/S forward/back | A/D strafe | Q/E down/up")
    stdscr.addstr(3, 0, "Arrow left/right yaw | Arrow up/down pitch | Z/X roll")
    stdscr.addstr(4, 0, "SPACE capture | R auto-record | +/- move step | [/ ] rot step | ESC save+exit")
    stdscr.addstr(6, 0, f"x={x:.3f} y={y:.3f} z={z:.3f} roll={roll:.3f} pitch={pitch:.3f} yaw={yaw:.3f}")
    stdscr.addstr(7, 0, f"captured={len(records)} auto_record={auto_record} step={step:.3f} rot={rot:.3f} send_rate={send_rate:.1f}Hz")
    stdscr.addstr(9, 0, f"status: {status[:120]}")
    stdscr.refresh()


def run(stdscr, args, node):
    curses.curs_set(0)
    stdscr.nodelay(True)
    stdscr.keypad(True)

    out = Path(args.out)
    if args.clean and out.exists():
        shutil.rmtree(out)
    (out / "images").mkdir(parents=True, exist_ok=True)

    pose = [args.x, args.y, args.z, args.roll, args.pitch, args.yaw]
    step = args.trans_step
    rot = args.rot_step
    send_dt = 1.0 / args.send_rate

    records = []
    frame_idx = 0
    auto_record = False
    status = "ready"

    pending_proc = None
    pending_started_at = 0.0
    dirty_pose = True
    last_send = 0.0
    last_draw = 0.0
    last_autosave_count = 0

    while True:
        rclpy.spin_once(node, timeout_sec=0.001)

        # Tastatur komplett drainen, damit kein alter Buffer langsam nachläuft.
        keys = []
        while True:
            ch = stdscr.getch()
            if ch == -1:
                break
            keys.append(ch)

        # Bei riesigem Buffer nur die letzten Events behalten.
        if len(keys) > 25:
            keys = keys[-25:]
            status = "keyboard backlog dropped"

        moved = False
        want_capture = False

        for ch in keys:
            if ch == 27:  # ESC
                save_route(out, records)
                return

            key = chr(ch).lower() if 0 <= ch < 256 else ""
            x, y, z, roll, pitch, yaw = pose

            fwd = np.array([math.cos(yaw), math.sin(yaw)], dtype=float)
            left = np.array([math.cos(yaw + math.pi / 2), math.sin(yaw + math.pi / 2)], dtype=float)

            if key == "w":
                x += step * fwd[0]; y += step * fwd[1]; moved = True
            elif key == "s":
                x -= step * fwd[0]; y -= step * fwd[1]; moved = True
            elif key == "a":
                x += step * left[0]; y += step * left[1]; moved = True
            elif key == "d":
                x -= step * left[0]; y -= step * left[1]; moved = True
            elif key == "q":
                z -= step; moved = True
            elif key == "e":
                z += step; moved = True
            elif ch == curses.KEY_LEFT:
                yaw += rot; moved = True
            elif ch == curses.KEY_RIGHT:
                yaw -= rot; moved = True
            elif ch == curses.KEY_UP:
                pitch -= rot; moved = True
            elif ch == curses.KEY_DOWN:
                pitch += rot; moved = True
            elif key == "z":
                roll += rot; moved = True
            elif key == "x":
                roll -= rot; moved = True
            elif key in ["+", "="]:
                step *= 1.25
                status = f"step={step:.3f}"
            elif key in ["-", "_"]:
                step = max(0.01, step / 1.25)
                status = f"step={step:.3f}"
            elif key == "]":
                rot *= 1.25
                status = f"rot={rot:.3f}"
            elif key == "[":
                rot = max(0.005, rot / 1.25)
                status = f"rot={rot:.3f}"
            elif key == "r":
                auto_record = not auto_record
                status = f"auto_record={auto_record}"
            elif ch == ord(" "):
                want_capture = True

            if moved:
                pose = [x, y, z, roll, pitch, yaw]
                dirty_pose = True

        now = time.time()

        # Kill stale ign service clients. Otherwise one hung set_pose call can freeze control.
        if pending_proc is not None and pending_proc.poll() is None:
            if now - pending_started_at > args.pose_timeout:
                try:
                    pending_proc.kill()
                except Exception:
                    pass
                pending_proc = None
                dirty_pose = True
                status = "stale set_pose client killed"

        # Async Pose-Send: keine Warteschlange, immer nur letzte Pose.
        if dirty_pose and now - last_send >= send_dt:
            if pending_proc is None or pending_proc.poll() is not None:
                pending_proc = send_pose_async(args.world, args.name, pose)
                pending_started_at = now
                last_send = now
                dirty_pose = False
                status = "pose sent async"

                if auto_record:
                    ok, msg = capture_frame(node, out, frame_idx, pose, records)
                    status = msg
                    if ok:
                        frame_idx += 1

                    if records and len(records) != last_autosave_count and len(records) % args.autosave_every == 0:
                        save_route(out, records)
                        last_autosave_count = len(records)
                        status += " | autosaved"

        if want_capture:
            ok, msg = capture_frame(node, out, frame_idx, pose, records)
            status = msg
            if ok:
                frame_idx += 1
                save_route(out, records)
                last_autosave_count = len(records)
                status += " | autosaved"

        if now - last_draw > 0.05:
            draw(stdscr, pose, step, rot, args.send_rate, auto_record, records, status)
            last_draw = now

        time.sleep(0.005)


def main():
    os.environ.setdefault("ESCDELAY", "25")

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

    ap.add_argument("--trans-step", type=float, default=0.12)
    ap.add_argument("--rot-step", type=float, default=0.06)
    ap.add_argument("--send-rate", type=float, default=8.0)
    ap.add_argument("--pose-timeout", type=float, default=0.35)
    ap.add_argument("--autosave-every", type=int, default=10)
    args = ap.parse_args()

    rclpy.init()
    node = ImageGrabber()
    try:
        curses.wrapper(run, args, node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

    print("[OK] wrote:", args.out)


if __name__ == "__main__":
    main()
