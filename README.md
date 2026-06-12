# Arbitrary Camera Rig Calibration

This repository contains the APPRAS project **Arbitrary Camera Rig Calibration**.

The project builds a synthetic Gazebo/Ignition benchmark for calibrating a multi-camera rig in a BeIntelli/Intellibus-style vehicle setup.

## Main result

Final extrinsics are here:

```text
results/bus_real_data/07_final_extrinsics_cam3_reference/
```

Reference camera:

```text
cam_edge_3
```

Estimated calibration graph:

```text
cam_edge_3 -> cam_edge_1:
  direct static ArUco calibration

cam_edge_3 -> cam_edge_0:
  moving-camera relay calibration

cam_edge_3 -> cam_edge_5:
  moving-camera relay calibration
```

Ground truth is used only for evaluation.

## Main folders

```text
src/calib_lab/bus_real_data/
```

Gazebo world setup, camera config, marker placements, route config and world-building scripts.

```text
run/bus_real_data/
```

Numbered experiment/evaluation pipeline scripts.

```text
results/bus_real_data/
```

Generated outputs, reports, COLMAP reconstruction, relay evaluation and final extrinsics.

```text
src/calib_lab/beintelli_bus_model/
```

BeIntelli bus model and related assets.

```text
src/calib_lab/minimal_world/
```

Earlier minimal setup kept for reference.

## Current main world

```text
src/calib_lab/bus_real_data/worlds/bus_real_data_moving_camera.sdf
```

Launch:

```bash
cd "/mnt/c/Users/maxim/Desktop/Application of Robotics and Autonomous Systems/camera-rig-calibration"

unset IGN_GAZEBO_RESOURCE_PATH
unset GZ_SIM_RESOURCE_PATH

REPO="$PWD"

export IGN_GAZEBO_RESOURCE_PATH="$REPO/src/calib_lab/bus_real_data/models:$REPO/src/calib_lab/beintelli_bus_model/models:$REPO/src/calib_lab/minimal_world/models"
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

View image:

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
```

The known marker size is used for OpenCV PnP and for metric COLMAP scale estimation.

## Pipeline

See:

```text
run/bus_real_data/README.md
```

## Final output

See:

```text
results/bus_real_data/07_final_extrinsics_cam3_reference/
```

## Research focus

The project is intended as an ablation study for camera-rig calibration:

- direct static calibration vs moving-camera relay
- GT moving-camera motion vs COLMAP-estimated motion
- metric COLMAP scale from known ArUco markers
- marker quality and single-marker PnP failure cases
- route density and COLMAP registration quality
- calibration of cameras without overlapping fields of view
