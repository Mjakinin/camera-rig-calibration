#!/usr/bin/env python3

import argparse
import csv
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="src/calib_lab/beintelli_bus_model/config/board_stations/aruco_medium_station_candidates.csv",
    )
    parser.add_argument(
        "--result_base",
        default="results/beintelli_bus_model/aruco_board_pose/station_candidates_medium_a1",
    )
    parser.add_argument(
        "--output_csv",
        default="results/beintelli_bus_model/aruco_board_pose/station_candidates_medium_a1_summary.csv",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    result_base = Path(args.result_base)
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    with config_path.open() as f:
        station_rows = list(csv.DictReader(f))

    rows_out = []

    for station in station_rows:
        station_name = station["station_name"]
        obs_csv = result_base / station_name / "aruco_board_pose_observations.csv"

        if not obs_csv.exists():
            rows_out.append({
                "station_name": station_name,
                "group": station.get("group", ""),
                "target_static_camera": station.get("target_static_camera", ""),
                "description": station.get("description", ""),
                "board_pose": station.get("board_pose", ""),
                "camera": "MISSING_RESULT",
                "status": "missing",
                "num_detected": "",
                "num_used_markers": "",
                "num_points": "",
                "reprojection_rmse_px": "",
                "used_ids": "",
                "tvec_x_m": "",
                "tvec_y_m": "",
                "tvec_z_m": "",
                "notes": station.get("notes", ""),
            })
            continue

        with obs_csv.open() as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows_out.append({
                    "station_name": station_name,
                    "group": station.get("group", ""),
                    "target_static_camera": station.get("target_static_camera", ""),
                    "description": station.get("description", ""),
                    "board_pose": station.get("board_pose", ""),
                    "camera": row["camera"],
                    "status": row["status"],
                    "num_detected": row["num_detected"],
                    "num_used_markers": row["num_used_markers"],
                    "num_points": row["num_points"],
                    "reprojection_rmse_px": row["reprojection_rmse_px"],
                    "used_ids": row["used_ids"],
                    "tvec_x_m": row["tvec_x_m"],
                    "tvec_y_m": row["tvec_y_m"],
                    "tvec_z_m": row["tvec_z_m"],
                    "notes": station.get("notes", ""),
                })

    fieldnames = [
        "station_name",
        "group",
        "target_static_camera",
        "camera",
        "status",
        "num_detected",
        "num_used_markers",
        "num_points",
        "reprojection_rmse_px",
        "used_ids",
        "tvec_x_m",
        "tvec_y_m",
        "tvec_z_m",
        "board_pose",
        "description",
        "notes",
    ]

    with output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_out)

    print(f"[OK] Wrote summary CSV: {output_csv}")
    print("")
    print("station,camera,status,markers,rmse,used_ids")
    for row in rows_out:
        print(
            f"{row['station_name']},"
            f"{row['camera']},"
            f"{row['status']},"
            f"{row['num_used_markers']},"
            f"{row['reprojection_rmse_px']},"
            f"\"{row['used_ids']}\""
        )


if __name__ == "__main__":
    main()
