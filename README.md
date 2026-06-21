# Arbitrary Camera Rig Calibration

This repository contains the APPRAS project **Arbitrary Camera Rig Calibration**. The project builds a repeatable simulation and evaluation pipeline for calibrating an arbitrary multi-camera rig on a BeIntelli/Intellibus-style vehicle.

The current focus is not only to obtain one calibration result, but to compare calibration approaches under controlled conditions. Gazebo provides known ground truth, OpenCV/ArUco provides marker observations, COLMAP provides targetless structure-from-motion motion estimates, and the generated result reports quantify translation and rotation error against the known reference.

## Current validated status

The current bus real-data pipeline has been validated with the full run sequence:

```bash
PYTHONPATH=run/bus_real_data find run/bus_real_data -name "*.py" -print0 | xargs -0 python3 -m py_compile
bash run/bus_real_data/approach1_marker_direct_relay/run_approach1_full_pipeline.sh
bash run/bus_real_data/approach2_ref_marker_graph_ba/run_approach2_full_pipeline.sh --skip-shared-baseline
bash run/bus_real_data/approach3_targetless_colmap_aruco_scale/run_approach3_full_pipeline.sh
```

The expected validation order is:

```text
00_py_compile
01_AP01_full
02_AP02_full
03_AP03_full_COLMAP
```

AP03 is the longest step because it runs a full COLMAP sparse reconstruction. Bundle-adjustment messages such as `Termination : No convergence` inside COLMAP are not automatically fatal. The important condition is that the shell script reaches `[OK] AP03 full pipeline complete.` and writes the final report.

## Repository layout

```text
src/calib_lab/
```

ROS 2/Gazebo package and simulation assets. This contains the Gazebo worlds, camera configuration, marker placement configuration, model assets and world-building scripts. It is the simulation/source side of the project.

```text
run/bus_real_data/
```

Executable calibration and evaluation pipelines. This is the main place to run the three approaches and shared preprocessing.

```text
results/bus_real_data/
```

Generated datasets, ArUco observations, COLMAP reconstructions, numerical evaluations, reports and comparison outputs.

## Current bus real-data approaches

### AP01: marker direct relay / multimarker / multichain

Path:

```text
run/bus_real_data/approach1_marker_direct_relay/
results/bus_real_data/01_marker_direct_relay_multimarker_multichain/
```

AP01 uses direct static ArUco calibration where there is sufficient marker visibility and moving-camera relay chains for cameras without direct overlap. It keeps a hidden, reproducible AP01 compatibility cache generated from the shared baseline.

Final AP01 extrinsics:

```text
results/bus_real_data/01_marker_direct_relay_multimarker_multichain/07_final_extrinsics_cam3_reference/
```

### AP02: reference-marker graph + bundle adjustment

Path:

```text
run/bus_real_data/approach2_ref_marker_graph_ba/
results/bus_real_data/02_ref_marker_graph_ba/
```

AP02 imports the shared ArUco observations and builds a reference-marker graph. It then performs graph initialization and bundle adjustment variants for static-only and moving-camera-assisted calibration.

### AP03: targetless COLMAP + ArUco scale registration

Path:

```text
run/bus_real_data/approach3_targetless_colmap_aruco_scale/
results/bus_real_data/03_targetless_colmap_aruco_scale/
```

AP03 first reconstructs all static and moving images with COLMAP without using marker constraints for the SfM frontend. It then detects the reference ArUco marker, triangulates marker corners in COLMAP coordinates, estimates a Sim(3) alignment to metric marker coordinates and evaluates static-camera poses against Gazebo ground truth.

Final AP03 report:

```text
results/bus_real_data/03_targetless_colmap_aruco_scale/06_triangulated_ref_aruco_registration/AP03_FINAL_REPO_LIKE_SCALE_REGISTRATION_REPORT.txt
```

## Shared baseline contract

All current bus real-data approaches use the shared raw dataset and shared ArUco detections:

```text
results/bus_real_data/00_shared_baseline/bus_real_data_ref_marker_v1/
  raw_images/
  aruco_observations/
  metadata/
```

Important invariant:

```text
results/bus_real_data/00_raw_images/bus_real_data_ref_marker_v1
```

is a legacy path and should not be used by active pipelines anymore. The active raw-image path is:

```text
results/bus_real_data/00_shared_baseline/bus_real_data_ref_marker_v1/raw_images
```

## Main Gazebo world

Current main world:

```text
src/calib_lab/bus_real_data/worlds/bus_real_data_moving_camera.sdf
```

Launch example:

```bash
cd /workspaces/project

unset IGN_GAZEBO_RESOURCE_PATH
unset GZ_SIM_RESOURCE_PATH

REPO="$PWD"
export IGN_GAZEBO_RESOURCE_PATH="$REPO/src/calib_lab/bus_real_data/models"
export GZ_SIM_RESOURCE_PATH="$IGN_GAZEBO_RESOURCE_PATH"

ign gazebo -r "$REPO/src/calib_lab/bus_real_data/worlds/bus_real_data_moving_camera.sdf"
```

## ROS/Gazebo moving-camera bridge

```bash
source /opt/ros/humble/setup.bash

ros2 run ros_gz_bridge parameter_bridge \
  /bus_real_data/moving_calib_camera/image@sensor_msgs/msg/Image@gz.msgs.Image \
  /bus_real_data/moving_calib_camera/camera_info@sensor_msgs/msg/CameraInfo@gz.msgs.CameraInfo
```

View the image stream with:

```bash
source /opt/ros/humble/setup.bash
rqt_image_view
```

Select:

```text
/bus_real_data/moving_calib_camera/image
```

## Marker geometry

```text
A4 sheet:      0.210 m x 0.297 m
ArUco marker: 0.170 m x 0.170 m
Dictionary:   DICT_4X4_50
Reference:    marker 14 / cam_edge_3 reference frame
```

The known marker size is used by marker-based OpenCV/PnP steps and by AP03 after COLMAP for metric Sim(3) scale registration.

## Recommended final checks before submission

```bash
cd /workspaces/project

git status --short

grep -RIn "results/bus_real_data/00_raw_images/bus_real_data_ref_marker_v1" \
  run/bus_real_data results/bus_real_data \
  --exclude-dir=.git \
  --exclude-dir=_overnight_pipeline_logs \
  --exclude-dir=_visible_full_pipeline_logs || true

find -L results/bus_real_data/00_shared_baseline -xtype l -print
```

The old raw-path grep and broken-symlink check should print nothing.
