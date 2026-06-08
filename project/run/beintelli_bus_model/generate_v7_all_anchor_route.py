#!/usr/bin/env python3
import csv
import math
from pathlib import Path

OUT_DIR = Path("results/beintelli_bus_model/colmap/moving_route_v7_all_success_anchors_static_world")
OUT = OUT_DIR / "route_plan_v7.csv"
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
waypoints = [
    # Front anchor success cases
    ("anchor_F3_front_near_left_seat",
     -3.855882, 0.045762, 1.650000, 0.0, 0.000000, 0.638235, 12),

    ("front_bridge_to_F4_1",
     -3.40, 0.020000, 1.750000, 0.0, 0.150000, 0.300000, 4),

    ("anchor_F4_front_right_table_or_box",
     -2.767818, 0.095723, 2.350000, 0.0, 0.681479, -1.221730, 12),

    # Smooth return from high/downward F4 to aisle/backbone
    ("front_recover_1",
     -2.70, 0.050000, 2.100000, 0.0, 0.450000, -0.600000, 4),
    ("front_recover_2",
     -2.40, 0.020000, 1.850000, 0.0, 0.200000, 0.100000, 4),
    ("front_aisle",
     -2.00, 0.000000, 1.650000, 0.0, 0.000000, 0.350000, 4),

    # Backbone through bus
    ("aisle_1", -1.20, 0.000000, 1.650000, 0.0, 0.000000, 0.700000, 4),
    ("aisle_2", -0.40, 0.000000, 1.650000, 0.0, 0.000000, 1.000000, 4),
    ("aisle_3",  0.40, 0.000000, 1.650000, 0.0, 0.000000, 1.350000, 4),
    ("aisle_4",  1.20, 0.000000, 1.650000, 0.0, 0.000000, 1.750000, 4),
    ("aisle_5",  2.00, 0.000000, 1.650000, 0.0, 0.000000, 2.150000, 4),
    ("aisle_6",  2.80, 0.000000, 1.650000, 0.0, 0.000000, 2.550000, 4),
    ("rear_aisle", 3.40, 0.020000, 1.650000, 0.0, 0.000000, 2.900000, 4),

    # R3 side anchor
    ("rear_bridge_to_R3",
     3.80, -0.120000, 1.750000, 0.0, 0.100000, 1.900000, 4),

    ("anchor_R3_rear_right_seat_angled",
     4.061622, -0.251769, 1.850000, 0.0, 0.244979, 1.396263, 12),

    # R2 table anchor
    ("rear_bridge_to_R2_1",
     4.40, -0.050000, 1.800000, 0.0, 0.200000, 2.200000, 4),

    ("anchor_R2_rear_table_flat",
     5.430000, 0.310000, 1.750000, 0.0, 0.308753, -2.553590, 12),

    # R1 final rear anchor
    ("rear_bridge_to_R1_1",
     4.90, 0.180000, 1.700000, 0.0, 0.100000, -2.900000, 4),
    ("rear_bridge_to_R1_2",
     4.50, 0.100000, 1.650000, 0.0, 0.000000, 3.050000, 4),

    ("anchor_R1_rear_left_seat_leaned",
     4.164286, 0.048746, 1.650000, 0.0, 0.000000, 3.239085, 12),
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
        angle_dist = abs(nyaw - yaw)
        steps = max(10, int(dist / 0.08), int(angle_dist / 0.06))

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
