#!/usr/bin/env python3
import csv
import subprocess
from pathlib import Path

CANDIDATES = Path("results/beintelli_bus_model/station_anchor_search/F4_orbit_candidates_only.csv")
OUT_ROOT = Path("results/beintelli_bus_model/station_anchor_search/orbit_results")

SET_POSE = Path("src/calib_lab/beintelli_bus_model/scripts/tools/live_set_entity_pose.py")
ESTIMATOR = Path("src/calib_lab/beintelli_bus_model/scripts/aruco_board/bus_aruco_board_pose_estimator.py")

def run(cmd):
    print("[RUN]", " ".join(str(c) for c in cmd))
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print("[WARN] command failed:", result.returncode)
    return result.returncode

OUT_ROOT.mkdir(parents=True, exist_ok=True)

with CANDIDATES.open() as f:
    reader = csv.DictReader(f)
    rows = list(reader)

print(f"[INFO] F4 candidates: {len(rows)}")

for i, row in enumerate(rows, start=1):
    station = row["station_name"]
    cand = row["candidate_name"]

    board_pose = row["board_pose"].split()
    moving_pose = row["moving_pose"].split()
    out_dir = OUT_ROOT / station / cand

    # Skip already completed candidates.
    if (out_dir / "aruco_board_pose_observations.csv").exists():
        print(f"[SKIP] already exists: {out_dir}")
        continue

    print()
    print("=" * 80)
    print(f"[{i}/{len(rows)}] {cand}")
    print("station:", station)
    print("moving_pose:", row["moving_pose"])

    run(["python3", str(SET_POSE), "--name", "calibration_board", "--pose", *board_pose])
    run(["python3", str(SET_POSE), "--name", "moving_calib_camera", "--pose", *moving_pose])
    run([
        "python3", str(ESTIMATOR),
        "--dictionary", "DICT_4X4_50",
        "--output_dir", str(out_dir),
        "--wait_sec", "2.0",
    ])
