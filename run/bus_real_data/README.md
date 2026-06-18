# bus_real_data Pipeline Scripts

This folder contains the executable experiment/evaluation pipeline.

The scripts are numbered in execution order. The world-generation scripts are located in:

```text
src/calib_lab/bus_real_data/scripts/
```

Those scripts build the Gazebo worlds and marker assets. This folder contains the actual calibration, COLMAP, evaluation and final export pipeline.

## Script order

```text
01_detect_static_a4_markers.py
02_generate_moving_camera_route.py
03_preview_moving_camera_route.py
04_capture_moving_camera_route.py
05_detect_moving_a4_markers.py
06_run_colmap_moving_sequence.py
07_evaluate_colmap_position_vs_gt.py
08_make_colmap_error_tables.py
09_evaluate_colmap_rotation_vs_gt.py
10_eval_direct_static_cam3_cam1.py
11_make_direct_static_report_cam3_cam1.py
12_estimate_colmap_scale_from_aruco.py
14_eval_moving_relay_chains.py
15_export_final_extrinsics_cam3_reference.py
```

The two unnumbered scripts are manual helper tools:

```text
live_set_marker_pose.py
save_live_marker_poses_from_gazebo.py
```

## Recommended full execution order

```bash
python3 run/bus_real_data/01_detect_static_a4_markers.py --clean

python3 run/bus_real_data/02_generate_moving_camera_route.py
python3 run/bus_real_data/03_preview_moving_camera_route.py --sleep 0.08
python3 run/bus_real_data/04_capture_moving_camera_route.py --clean

python3 run/bus_real_data/05_detect_moving_a4_markers.py \
  --sequence results/bus_real_data/03_moving_camera_sequence \
  --clean-debug

python3 run/bus_real_data/06_run_colmap_moving_sequence.py \
  --sequence results/bus_real_data/03_moving_camera_sequence \
  --out results/bus_real_data/04_colmap_moving_sequence \
  --clean \
  --matcher exhaustive \
  --use-gpu 0

python3 run/bus_real_data/07_evaluate_colmap_position_vs_gt.py \
  --sequence results/bus_real_data/03_moving_camera_sequence \
  --colmap results/bus_real_data/04_colmap_moving_sequence

python3 run/bus_real_data/08_make_colmap_error_tables.py

python3 run/bus_real_data/09_evaluate_colmap_rotation_vs_gt.py \
  --sequence results/bus_real_data/03_moving_camera_sequence \
  --colmap results/bus_real_data/04_colmap_moving_sequence

python3 run/bus_real_data/10_eval_direct_static_cam3_cam1.py
python3 run/bus_real_data/11_make_direct_static_report_cam3_cam1.py

python3 run/bus_real_data/12_estimate_colmap_scale_from_aruco.py
python3 run/bus_real_data/14_eval_moving_relay_chains.py
python3 run/bus_real_data/15_export_final_extrinsics_cam3_reference.py
```

## Script purpose

### 01_detect_static_a4_markers.py

Detects A4 ArUco markers in static camera images.

Output:

```text
results/bus_real_data/01_static_a4_marker_detection/
```

### 02_generate_moving_camera_route.py

Generates the interpolated moving-camera route from manual keyframes.

Input:

```text
src/calib_lab/bus_real_data/config/moving_camera_route_keyframes.json
```

Output:

```text
src/calib_lab/bus_real_data/config/moving_camera_route_interpolated.json
results/bus_real_data/02_moving_camera_route/
```

### 03_preview_moving_camera_route.py

Previews the moving-camera route in Gazebo.

### 04_capture_moving_camera_route.py

Captures moving-camera images along the generated route.

Output:

```text
results/bus_real_data/03_moving_camera_sequence/
```

### 05_detect_moving_a4_markers.py

Detects A4 ArUco markers in the moving-camera sequence.

Important output:

```text
results/bus_real_data/03_moving_camera_sequence/moving_detections.csv
```

### 06_run_colmap_moving_sequence.py

Runs COLMAP on the moving-camera image sequence.

Output:

```text
results/bus_real_data/04_colmap_moving_sequence/
```

### 07_evaluate_colmap_position_vs_gt.py

Diagnostic COLMAP position evaluation against Gazebo GT using Sim3 alignment.

Important: this is diagnostic only. The final no-GT relay calibration does not use Sim3-GT scale.

### 08_make_colmap_error_tables.py

Creates readable COLMAP error tables.

### 09_evaluate_colmap_rotation_vs_gt.py

Evaluates COLMAP orientation against GT.

### 10_eval_direct_static_cam3_cam1.py

Evaluates direct static-to-static calibration from `cam_edge_3` to `cam_edge_1`.

Output:

```text
results/bus_real_data/05_direct_static_cam3_cam1/
```

### 11_make_direct_static_report_cam3_cam1.py

Creates the readable direct-static report.

### 12_estimate_colmap_scale_from_aruco.py

Estimates metric COLMAP scale from known 0.170 m ArUco marker observations.

Output:

```text
results/bus_real_data/04_colmap_moving_sequence/aruco_metric_scale/
```

### 14_eval_moving_relay_chains.py

Evaluates moving-camera relay calibration for:

```text
cam_edge_3 -> cam_edge_0
cam_edge_3 -> cam_edge_5
```

Output:

```text
results/bus_real_data/06_moving_relay_chain_eval/
```

### 15_export_final_extrinsics_cam3_reference.py

Exports final extrinsics with `cam_edge_3` as reference.

Output:

```text
results/bus_real_data/07_final_extrinsics_cam3_reference/
```

## Final validation

After cleanup/renaming, this command should run successfully:

```bash
python3 run/bus_real_data/15_export_final_extrinsics_cam3_reference.py
```
