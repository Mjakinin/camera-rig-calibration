# Current BeIntelli Bus Calibration Results

This folder contains the current clean result structure.

## 01_station_id_detection

ArUco ID detection and station suitability.

Valid front board candidates:
- F3
- F4

Valid rear board candidates:
- R1
- R3

Why only four front-rear board pairs:
F3/F4 are valid front-side boards and R1/R3 are valid rear-side boards.
Therefore:
- F3_R1
- F3_R3
- F4_R1
- F4_R3

## 02_colmap_v8_moving_sequence

Latest V8 moving-camera image sequence and COLMAP sparse reconstruction.

Important files:
- images/
- sparse_txt/
- route_commanded.csv
- aruco_no_gt_detections/

## 03_colmap_moving_pose_vs_gt

Evaluation of the COLMAP reconstructed moving-camera trajectory against the simulation trajectory.

Important:
This is evaluation only.
The final no-GT calibration pipeline does not use the simulation trajectory.

## 04_front_rear_chain_no_gt_colmap

Final no-GT front_static_camera -> rear_static_camera chain using:
- ArUco/PnP board observations
- COLMAP moving-camera motion
- known board size for metric scale

Best current pair:
- F3_R3

F3_R3:
- baseline_est_m:       9.620700
- baseline_gt_m:        9.600000
- baseline_error_cm:    2.07
- translation_error_cm: 5.03
- rotation_error_deg:   1.24

Important:
Ground truth is only used for final evaluation.
