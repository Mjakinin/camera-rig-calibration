#!/usr/bin/env python3
import csv
import subprocess
import sys
from pathlib import Path

CANDIDATES = Path("results/beintelli_bus_model/station_anchor_search/station_anchor_pose_candidates.csv")
STATIONS = Path("src/calib_lab/beintelli_bus_model/config/board_stations/aruco_medium_station_candidates.csv")
OUT_ROOT = Path("results/beintelli_bus_model/station_anchor_search/live_results")

SET_POSE = Path("src/calib_lab/beintelli_bus_model/scripts/tools/live_set_entity_pose.py")
ESTIMATOR = Path("src/calib_lab/beintelli_bus_model/scripts/aruco_board/bus_aruco_board_pose_estimator.py")

def run(cmd):
    print()
    print("[RUN]", " ".join(str(c) for c in cmd))
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print("[WARN] command failed:", result.returncode)
    return result.returncode

def read_stations():
    d = {}
    with STATIONS.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            d[row["station_name"]] = row
    return d

def main():
    stations = read_stations()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    with CANDIDATES.open() as f:
        reader = csv.DictReader(f)
        candidates = list(reader)

    print(f"[INFO] candidates: {len(candidates)}")

    for i, cand in enumerate(candidates, start=1):
        station_name = cand["station_name"]
        station = stations[station_name]

        board_pose = station["board_pose"].split()
        moving_pose = cand["moving_pose"].split()

        out_dir = OUT_ROOT / cand["station_name"] / cand["candidate_name"]

        print()
        print("=" * 80)
        print(f"[{i}/{len(candidates)}] {cand['candidate_name']}")
        print("station:", station_name)
        print("board_pose:", station["board_pose"])
        print("moving_pose:", cand["moving_pose"])
        print("out_dir:", out_dir)

        run(["python3", str(SET_POSE), "--name", "calibration_board", "--pose", *board_pose])
        run(["python3", str(SET_POSE), "--name", "moving_calib_camera", "--pose", *moving_pose])
        run([
            "python3", str(ESTIMATOR),
            "--dictionary", "DICT_4X4_50",
            "--output_dir", str(out_dir),
            "--wait_sec", "2.5",
        ])

if __name__ == "__main__":
    main()
