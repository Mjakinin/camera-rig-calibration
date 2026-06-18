# bus_real_data Results

This folder contains the generated results for the current camera-rig calibration pipeline.

## Start here

```text
07_final_extrinsics_cam3_reference/
```

This folder contains the final extrinsics with `cam_edge_3` as reference.

## Folder structure

### 01_static_a4_marker_detection/

Static ArUco detections for the static cameras.

Important files:

```text
detections.csv
summary_by_camera.csv
debug_images/
raw_images/
```

### 02_moving_camera_route/

Generated moving-camera route.

### 03_moving_camera_sequence/

Captured moving-camera images and moving ArUco detections.

Important files:

```text
route_commanded.csv
moving_detections.csv
moving_summary_by_frame.csv
moving_coverage_by_marker.csv
moving_detection_report.txt
best_marker_frames/
debug_images/
```

### 04_colmap_moving_sequence/

COLMAP moving-camera reconstruction.

Important subfolder:

```text
aruco_metric_scale/
```

This contains the no-GT metric COLMAP scale estimated from known 0.170 m ArUco marker observations.

### 05_direct_static_cam3_cam1/

Direct static-to-static baseline for:

```text
cam_edge_3 -> cam_edge_1
```

### 06_moving_relay_chain_eval/

Moving-camera relay evaluation for:

```text
cam_edge_3 -> cam_edge_0
cam_edge_3 -> cam_edge_5
```

Includes:

```text
GT_motion
  oracle/sanity baseline

COLMAP_motion
  real-life-near estimate using COLMAP motion and ArUco metric scale
```

### 07_final_extrinsics_cam3_reference/

Final camera rig extrinsics summary.

Important files:

```text
FINAL_CAMERA_RIG_OVERVIEW.txt
FINAL_CAMERA_RIG_OVERVIEW.md
final_extrinsics_summary.csv
final_extrinsics_cam3_reference.json
pairwise_extrinsics_summary.csv
README.txt
```

## Ground truth usage

Ground truth is used only for evaluation errors.

The real-life-near moving relay estimate uses:

- static ArUco detections
- moving-camera ArUco detections
- known 0.170 m ArUco marker size
- COLMAP relative moving-camera motion
- ArUco-derived metric COLMAP scale
