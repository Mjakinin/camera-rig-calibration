#!/usr/bin/env python3

import json
import math
from pathlib import Path

KEYFRAME_PATH = Path("src/calib_lab/bus_real_data/config/moving_camera_route_keyframes.json")
OUT_JSON = Path("src/calib_lab/bus_real_data/config/moving_camera_route_interpolated.json")
OUT_CSV = Path("results/bus_real_data/03_moving_camera_route/moving_camera_route_interpolated.csv")

# Smaller = more frames / smoother motion.
MAX_TRANSLATION_STEP_M = 0.15
MAX_YAW_STEP_RAD = 0.10
MAX_PITCH_STEP_RAD = 0.06
MAX_ROLL_STEP_RAD = 0.08


def translation_dist(a, b):
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    dz = b[2] - a[2]
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def segment_steps(a, b):
    trans_steps = math.ceil(translation_dist(a, b) / MAX_TRANSLATION_STEP_M)
    roll_steps = math.ceil(abs(b[3] - a[3]) / MAX_ROLL_STEP_RAD)
    pitch_steps = math.ceil(abs(b[4] - a[4]) / MAX_PITCH_STEP_RAD)
    yaw_steps = math.ceil(abs(b[5] - a[5]) / MAX_YAW_STEP_RAD)

    return max(2, trans_steps, roll_steps, pitch_steps, yaw_steps)


def lerp(a, b, t):
    # Important: yaw is intentionally NOT wrapped.
    # This preserves routes like 0 -> -6.283 -> -9.8.
    return [a[i] + t * (b[i] - a[i]) for i in range(6)]


def main():
    data = json.loads(KEYFRAME_PATH.read_text())
    keyframes = data["keyframes"]

    route = []
    frame_idx = 0

    print("[INFO] generating route from keyframes:")
    for k in keyframes:
        print(k["name"], k["pose"])

    print()
    print("[INFO] segment interpolation:")

    for seg_idx in range(len(keyframes) - 1):
        a = keyframes[seg_idx]["pose"]
        b = keyframes[seg_idx + 1]["pose"]
        steps = segment_steps(a, b)

        print(
            f"segment {seg_idx}: "
            f"{keyframes[seg_idx]['name']} -> {keyframes[seg_idx + 1]['name']} | "
            f"steps={steps} | "
            f"dx={b[0]-a[0]:.2f}, dyaw={b[5]-a[5]:.2f}, dpitch={b[4]-a[4]:.2f}"
        )

        for s in range(steps):
            t = s / steps
            pose = lerp(a, b, t)

            route.append({
                "frame": frame_idx,
                "segment": seg_idx,
                "x": pose[0],
                "y": pose[1],
                "z": pose[2],
                "roll": pose[3],
                "pitch": pose[4],
                "yaw": pose[5]
            })
            frame_idx += 1

    last_pose = keyframes[-1]["pose"]
    route.append({
        "frame": frame_idx,
        "segment": len(keyframes) - 2,
        "x": last_pose[0],
        "y": last_pose[1],
        "z": last_pose[2],
        "roll": last_pose[3],
        "pitch": last_pose[4],
        "yaw": last_pose[5]
    })

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    OUT_JSON.write_text(json.dumps({
        "description": "Interpolated moving camera route generated from moving_camera_route_keyframes.json. Translation and rotation are both bounded per frame.",
        "pose_format": "x y z roll pitch yaw",
        "num_frames": len(route),
        "max_translation_step_m": MAX_TRANSLATION_STEP_M,
        "max_yaw_step_rad": MAX_YAW_STEP_RAD,
        "max_pitch_step_rad": MAX_PITCH_STEP_RAD,
        "frames": route
    }, indent=2) + "\n")

    with OUT_CSV.open("w") as f:
        f.write("frame,segment,x,y,z,roll,pitch,yaw\n")
        for r in route:
            f.write(
                f"{r['frame']},{r['segment']},"
                f"{r['x']:.6f},{r['y']:.6f},{r['z']:.6f},"
                f"{r['roll']:.8f},{r['pitch']:.8f},{r['yaw']:.8f}\n"
            )

    print()
    print("[OK] wrote:", OUT_JSON)
    print("[OK] wrote:", OUT_CSV)
    print("[OK] number of route frames:", len(route))


if __name__ == "__main__":
    main()
