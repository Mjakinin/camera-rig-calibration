#!/usr/bin/env python3
import csv
import math
from pathlib import Path

OUT = Path("results/beintelli_bus_model/station_anchor_search/problem_station_orbit_candidates.csv")
OUT.parent.mkdir(parents=True, exist_ok=True)

stations = {
    "R3_rear_right_seat_angled": {
        "group": "rear",
        "target_static_camera": "rear_static_camera",
        "board_pose": "4.27 0.93 1.55 0 0.3 1.57079633",
        "radii": [0.8, 1.0, 1.2, 1.5],
        "angles_deg": [-160, -130, -100, -70, -40, -10, 20, 50, 80],
        "z_values": [1.55, 1.70, 1.85],
    },
    "F4_front_right_table_or_box": {
        "group": "front",
        "target_static_camera": "front_static_camera",
        "board_pose": "-2.46 -0.75 1.62 1.57079633 1.57079633 0",
        "radii": [0.7, 0.9, 1.1, 1.4],
        "angles_deg": [20, 50, 80, 110, 140, 170, -160, -130, -100],
        "z_values": [1.80, 2.00, 2.20, 2.35],
    },
}

def parse_pose(s):
    return [float(x) for x in s.split()]

def look_at_pose(cam_x, cam_y, cam_z, target_x, target_y, target_z):
    dx = target_x - cam_x
    dy = target_y - cam_y
    dz = target_z - cam_z

    yaw = math.atan2(dy, dx)
    horizontal = math.sqrt(dx * dx + dy * dy)
    pitch = math.atan2(cam_z - target_z, max(horizontal, 1e-6))

    return [cam_x, cam_y, cam_z, 0.0, pitch, yaw]

rows = []

for station_name, cfg in stations.items():
    bx, by, bz, br, bp, byaw = parse_pose(cfg["board_pose"])
    idx = 0

    for radius in cfg["radii"]:
        for angle_deg in cfg["angles_deg"]:
            a = math.radians(angle_deg)
            for z in cfg["z_values"]:
                cam_x = bx + radius * math.cos(a)
                cam_y = by + radius * math.sin(a)
                cam_z = z

                pose = look_at_pose(cam_x, cam_y, cam_z, bx, by, bz)

                rows.append({
                    "station_name": station_name,
                    "group": cfg["group"],
                    "target_static_camera": cfg["target_static_camera"],
                    "candidate_name": f"{station_name}_orbit_{idx:03d}",
                    "board_pose": cfg["board_pose"],
                    "moving_pose": " ".join(f"{v:.6f}" for v in pose),
                    "radius": radius,
                    "angle_deg": angle_deg,
                    "reason": "orbit_look_at_board",
                })
                idx += 1

with OUT.open("w", newline="") as f:
    fieldnames = [
        "station_name", "group", "target_static_camera",
        "candidate_name", "board_pose", "moving_pose",
        "radius", "angle_deg", "reason"
    ]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"[OK] wrote {OUT}")
print(f"[OK] candidates: {len(rows)}")
print("Preview:")
for r in rows[:10]:
    print(r["candidate_name"], r["moving_pose"])
