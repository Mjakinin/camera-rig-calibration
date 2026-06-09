#!/usr/bin/env python3
import subprocess
import time
from pathlib import Path

STATIONS = [
    {
        "name": "F3_front_near_left_seat",
        "first_id": 0,
        "moving_pose": "-3.855882 0.045762 1.650000 0.000000 0.000000 0.638235",
    },
    {
        "name": "F4_front_right_table_or_box",
        "first_id": 6,
        "moving_pose": "-2.767818 0.095723 2.350000 0.000000 0.681479 -1.221730",
    },
    {
        "name": "R3_rear_right_seat_angled",
        "first_id": 12,
        "moving_pose": "4.061622 -0.251769 1.850000 0.000000 0.244979 1.396263",
    },
    {
        "name": "R2_rear_table_flat",
        "first_id": 18,
        "moving_pose": "5.430000 0.310000 1.750000 0.000000 0.308753 -2.553590",
    },
    {
        "name": "R1_rear_left_seat_leaned_occluded",
        "first_id": 24,
        "moving_pose": "4.164286 0.048746 1.650000 0.000000 0.000000 3.239085",
    },
    {
        "name": "F1_front_mid_far_seat_leaned",
        "first_id": 30,
        "moving_pose": "2.490000 0.000000 1.650000 0.000000 0.000000 3.14159265",
    },
    {
        "name": "F2_front_mid_high_left_seat",
        "first_id": 36,
        "moving_pose": "2.570000 0.000000 1.750000 0.000000 0.000000 3.14159265",
    },
    {
        "name": "G_floor_mid",
        "first_id": 42,
        "moving_pose": "2.100000 0.000000 1.650000 0.000000 0.523599 3.14159265",
    },
]

SET_POSE = "src/calib_lab/beintelli_bus_model/scripts/tools/live_set_entity_pose.py"
ESTIMATOR = "src/calib_lab/beintelli_bus_model/scripts/aruco_board/bus_aruco_board_pose_estimator.py"
OUT_ROOT = Path("results/beintelli_bus_model/multi_static_8_station_visibility")

def run(cmd):
    print("")
    print("[CMD]", " ".join(cmd))
    subprocess.run(cmd, check=True)

def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    for station in STATIONS:
        name = station["name"]
        first_id = station["first_id"]
        pose_values = station["moving_pose"].split()
        out_dir = OUT_ROOT / f"{name}_ids_{first_id:02d}_{first_id+5:02d}"

        print("")
        print("=" * 100)
        print(f"[STATION] {name}")
        print(f"[IDS]     {first_id}..{first_id+5}")
        print(f"[POSE]    {station['moving_pose']}")
        print("=" * 100)

        run([
            "python3", SET_POSE,
            "--name", "moving_calib_camera",
            "--pose", *pose_values,
        ])

        time.sleep(0.5)

        run([
            "python3", ESTIMATOR,
            "--dictionary", "DICT_4X4_50",
            "--first_id", str(first_id),
            "--output_dir", str(out_dir),
            "--wait_sec", "3",
        ])

    print("")
    print("[OK] Visibility check finished.")
    print(f"[OK] Results: {OUT_ROOT}")

if __name__ == "__main__":
    main()
