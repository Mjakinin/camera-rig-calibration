#!/usr/bin/env python3
import csv
import math
from pathlib import Path

OUT_DIR = Path("results/beintelli_bus_model/colmap/moving_route_v6_anchor_optimized_static_world")
OUT = OUT_DIR / "route_plan_v6.csv"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PARK = {
    "board_name": "PARK",
    "board_x": 0.0,
    "board_y": 0.0,
    "board_z": -5.0,
    "board_roll": 0.0,
    "board_pitch": 0.0,
    "board_yaw": 0.0,
}

# name, x, y, z, roll, pitch, yaw, hold_frames
# This route intentionally contains the good F3 and R1_0312 anchor poses.
waypoints = [
    ("anchor_F3_front_near_left_seat", -3.855882, 0.045762, 1.650000, 0.0, 0.000000, 0.638235, 14),

    ("front_bridge_1", -3.50, 0.030000, 1.650000, 0.0, 0.000000, 0.550000, 4),
    ("front_bridge_2", -3.00, 0.010000, 1.650000, 0.0, 0.000000, 0.450000, 4),
    ("front_bridge_3", -2.40, 0.000000, 1.650000, 0.0, 0.000000, 0.450000, 4),

    ("aisle_1", -1.60, 0.000000, 1.650000, 0.0, 0.000000, 0.650000, 4),
    ("aisle_2", -0.80, 0.000000, 1.650000, 0.0, 0.000000, 0.900000, 4),
    ("aisle_3",  0.00, 0.000000, 1.650000, 0.0, 0.000000, 1.200000, 4),
    ("aisle_4",  0.80, 0.000000, 1.650000, 0.0, 0.000000, 1.550000, 4),
    ("aisle_5",  1.60, 0.000000, 1.650000, 0.0, 0.000000, 1.900000, 4),
    ("aisle_6",  2.40, 0.000000, 1.650000, 0.0, 0.000000, 2.250000, 4),

    ("rear_bridge_1", 3.00, 0.020000, 1.650000, 0.0, 0.000000, 2.600000, 4),
    ("rear_bridge_2", 3.50, 0.040000, 1.650000, 0.0, 0.000000, 2.950000, 4),
    ("rear_bridge_3", 3.90, 0.050000, 1.650000, 0.0, 0.000000, 3.150000, 4),

    ("anchor_R1_rear_left_seat_leaned", 4.164286, 0.048746, 1.650000, 0.0, 0.000000, 3.239085, 14),
]

def unwrap_angle(a, ref):
    while a - ref > math.pi:
        a -= 2 * math.pi
    while a - ref < -math.pi:
        a += 2 * math.pi
    return a

def lerp(a, b, t):
    return a + (b - a) * t

rows = []
idx = 0

for wi, wp in enumerate(waypoints):
    tag, x, y, z, roll, pitch, yaw, hold = wp

    for _ in range(hold):
        rows.append((idx, tag, x, y, z, roll, pitch, yaw))
        idx += 1

    if wi < len(waypoints) - 1:
        nt, nx, ny, nz, nr, np, nyaw, _ = waypoints[wi + 1]
        nyaw = unwrap_angle(nyaw, yaw)

        dist = math.sqrt((nx - x)**2 + (ny - y)**2 + (nz - z)**2)
        steps = max(8, int(dist / 0.08))

        for s in range(1, steps + 1):
            t = s / (steps + 1)
            rows.append((
                idx,
                f"transition_{tag}_to_{nt}",
                lerp(x, nx, t),
                lerp(y, ny, t),
                lerp(z, nz, t),
                lerp(roll, nr, t),
                lerp(pitch, np, t),
                lerp(yaw, nyaw, t),
            ))
            idx += 1

with OUT.open("w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "image_name", "x", "y", "z", "roll", "pitch", "yaw",
        "tag",
        "board_name", "board_x", "board_y", "board_z",
        "board_roll", "board_pitch", "board_yaw",
        "image_path",
    ])

    for i, tag, x, y, z, roll, pitch, yaw in rows:
        image_name = f"moving_{i:04d}.jpg"
        image_path = str(OUT_DIR / "images" / image_name)
        writer.writerow([
            image_name,
            f"{x:.6f}", f"{y:.6f}", f"{z:.6f}",
            f"{roll:.6f}", f"{pitch:.6f}", f"{yaw:.6f}",
            tag,
            PARK["board_name"],
            f"{PARK['board_x']:.6f}",
            f"{PARK['board_y']:.6f}",
            f"{PARK['board_z']:.6f}",
            f"{PARK['board_roll']:.6f}",
            f"{PARK['board_pitch']:.6f}",
            f"{PARK['board_yaw']:.6f}",
            image_path,
        ])

print(f"[OK] wrote: {OUT}")
print(f"[OK] frames: {len(rows)}")
print()
print("Anchor frames:")
for i, tag, x, y, z, roll, pitch, yaw in rows:
    if tag.startswith("anchor_"):
        print(f"  moving_{i:04d}.jpg  {tag}  --pose {x:.6f} {y:.6f} {z:.6f} {roll:.6f} {pitch:.6f} {yaw:.6f}")
