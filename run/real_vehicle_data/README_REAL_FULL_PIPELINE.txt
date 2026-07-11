REAL-DATA FULL PIPELINE BUNDLE
================================

Purpose
-------
Run AP01, AP02 and AP03 on:

  results/real_vehicle_data/real_05x_4k_3hz_v1/00_shared_input

without modifying or overwriting results/bus_real_data.

Method roots
------------
AP01:
  results/real_vehicle_data/real_05x_4k_3hz_v1/02_ap01_real

AP02:
  results/real_vehicle_data/real_05x_4k_3hz_v1/03_ap02_real

AP03:
  results/real_vehicle_data/real_05x_4k_3hz_v1/04_ap03_real

Final report:
  results/real_vehicle_data/real_05x_4k_3hz_v1/
  99_FINAL_RESULTS/REAL_DATA_ALL_METHODS.txt

Input contract
--------------
Static images:
  raw_images/static/cam_edge_0.png
  raw_images/static/cam_edge_1.png
  raw_images/static/cam_edge_3.png
  raw_images/static/cam_edge_5.png

Static intrinsics:
  raw_images/camera_info/cam_edge_*.json
  These must come from the provided camera_info ZIP.

Moving images:
  raw_images/moving/frame_*.png

Moving intrinsics:
  raw_images/camera_info/moving_calib_camera.json
  These apply only to the iPhone moving frames.

ArUco observations:
  results/.../00_shared_input/aruco_observations/shared_*_aruco_observations.csv

Reference marker:
  results/.../00_shared_input/aruco_observations/REFERENCE_MARKER_ID.txt

Method definitions
------------------
AP01
  - target-based direct static-camera overlap where available
  - moving-camera COLMAP trajectory
  - ArUco-based metric scale
  - moving-camera relay for cameras without direct overlap
  - no ground truth used

AP02
  - existing repository reference-marker graph initializer
  - existing distortion-aware graph bundle adjustment
  - reference marker selected from real observations
  - no ground truth used

AP03
  - grouped calibrated targetless COLMAP
  - one fixed camera model per physical static camera
  - one fixed moving-camera model
  - exhaustive matching by default
  - metric scale from known ArUco side length
  - no ground truth used

Installation
------------
Copy these five files into run/real_vehicle_data:

  07_run_ap01_real.py
  08_run_ap02_real.py
  09_run_ap03_real.py
  10_write_real_final_report.py
  run_full_real_pipeline.sh

Then:

  chmod +x run/real_vehicle_data/run_full_real_pipeline.sh
  python3 -m py_compile run/real_vehicle_data/0{7,8,9}_run_*.py
  python3 -m py_compile run/real_vehicle_data/10_write_real_final_report.py
  bash -n run/real_vehicle_data/run_full_real_pipeline.sh

Execution
---------
CPU COLMAP:

  bash run/real_vehicle_data/run_full_real_pipeline.sh --gpu 0

GPU COLMAP, only if the installed COLMAP build supports CUDA:

  bash run/real_vehicle_data/run_full_real_pipeline.sh --gpu 1

Reuse existing AP01/AP03 COLMAP reconstructions:

  bash run/real_vehicle_data/run_full_real_pipeline.sh \
    --gpu 0 \
    --reuse-colmap

Run one method:

  bash run/real_vehicle_data/run_full_real_pipeline.sh --only ap02
  bash run/real_vehicle_data/run_full_real_pipeline.sh --only ap01 --gpu 0
  bash run/real_vehicle_data/run_full_real_pipeline.sh --only ap03 --gpu 0

Regenerate only the final report:

  bash run/real_vehicle_data/run_full_real_pipeline.sh --only report

Important interpretation
------------------------
There is no complete real 6-DoF ground truth.

The final report therefore provides:
  - method execution status
  - static-camera coverage
  - six pairwise metric distances
  - optional measured-reference errors
  - cross-method disagreement
  - method-specific diagnostics

The file measured_reference_distances.json is created with null values.
Only independent physical measurements or trusted mounting transforms should
be entered there.
