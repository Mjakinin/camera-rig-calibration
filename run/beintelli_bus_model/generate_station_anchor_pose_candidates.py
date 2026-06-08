#!/usr/bin/env python3
import csv
import math
from pathlib import Path

IN = Path("src/calib_lab/beintelli_bus_model/config/board_stations/aruco_medium_station_candidates.csv")
OUT = Path("results/beintelli_bus_model/station_anchor_search/station_anchor_pose_candidates.csv")
OUT.parent.mkdir(parents=True, exist_ok=True)

def parse_pose(s):
    vals = [float(x) for x in s.strip().split()]
    if len(vals) != 6:
        raise ValueError(f"bad pose: {s}")
    return vals

def norm_angle(a):
    while a > math.pi:
        a -= 2 * math.pi
    while a < -math.pi:
        a += 2 * math.pi
    return a

def look_at_pose(cam_x, cam_y, cam_z, target_x, target_y, target_z):
    dx = target_x - cam_x
    dy = target_y - cam_y
    dz = target_z - cam_z

    yaw = math.atan2(dy, dx)
    horizontal = math.sqrt(dx * dx + dy * dy)

    # In your current Gazebo camera setup, positive pitch has worked for downward-looking table cases.
    pitch = math.atan2(cam_z - target_z, max(horizontal, 1e-6))

    return [cam_x, cam_y, cam_z, 0.0, pitch, yaw]

def add_candidate(rows, station, group, target_static, idx, pose, reason):
    rows.append({
        "station_name": station,
        "group": group,
        "target_static_camera": target_static,
        "candidate_name": f"{station}_cand_{idx:02d}",
        "moving_pose": " ".join(f"{v:.6f}" for v in pose),
        "reason": reason,
    })

rows = []

with IN.open() as f:
    reader = csv.DictReader(f)
    for row in reader:
        station = row["station_name"]
        group = row["group"]
        target_static = row["target_static_camera"]
        bx, by, bz, br, bp, byaw = parse_pose(row["board_pose"])

        idx = 0

        # Use existing manually stored moving_pose first, if available.
        if row.get("moving_pose", "").strip():
            add_candidate(rows, station, group, target_static, idx, parse_pose(row["moving_pose"]), "csv_existing_moving_pose")
            idx += 1

        if group == "front":
            # Candidate cameras in front/middle aisle, looking at board.
            offsets = [
                (-0.8, -0.2, 1.65),
                (-1.0,  0.0, 1.70),
                (-1.2,  0.2, 1.75),
                (-0.6,  0.4, 1.75),
                (-0.6, -0.4, 1.90),
            ]
            for ox, oy, cz in offsets:
                cam = (bx + ox, by + oy, cz)
                pose = look_at_pose(*cam, bx, by, bz)
                add_candidate(rows, station, group, target_static, idx, pose, "front_look_at_board")
                idx += 1

        elif group == "rear":
            # Candidate cameras behind/towards rear aisle, looking at board.
            offsets = [
                (0.8,  0.0, 1.65),
                (1.0,  0.2, 1.70),
                (1.0, -0.2, 1.70),
                (1.2,  0.0, 1.80),
                (0.6,  0.4, 1.75),
            ]
            for ox, oy, cz in offsets:
                cam = (bx + ox, by + oy, cz)
                pose = look_at_pose(*cam, bx, by, bz)
                add_candidate(rows, station, group, target_static, idx, pose, "rear_look_at_board")
                idx += 1

        elif group == "floor":
            # Higher camera, downward-looking. Try both front/rear-ish approaches.
            cams = [
                (bx - 0.8, by + 0.0, 2.10),
                (bx + 0.8, by + 0.0, 2.10),
                (bx - 0.4, by + 0.4, 2.30),
                (bx + 0.4, by - 0.4, 2.30),
                (bx + 0.0, by + 0.0, 2.40),
            ]
            for cam in cams:
                pose = look_at_pose(*cam, bx, by, bz)
                add_candidate(rows, station, group, target_static, idx, pose, "floor_downward_look_at_board")
                idx += 1

        else:
            # Generic fallback
            cams = [
                (bx + 1.0, by, 1.70),
                (bx - 1.0, by, 1.70),
                (bx, by + 1.0, 1.70),
                (bx, by - 1.0, 1.70),
            ]
            for cam in cams:
                pose = look_at_pose(*cam, bx, by, bz)
                add_candidate(rows, station, group, target_static, idx, pose, "generic_look_at_board")
                idx += 1

with OUT.open("w", newline="") as f:
    fieldnames = ["station_name", "group", "target_static_camera", "candidate_name", "moving_pose", "reason"]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"[OK] wrote {OUT}")
print(f"[OK] candidates: {len(rows)}")
for r in rows[:20]:
    print(r["candidate_name"], r["moving_pose"], r["reason"])
