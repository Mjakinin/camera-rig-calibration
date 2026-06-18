# bus_real_data Worlds

This folder contains Gazebo/Ignition SDF worlds for the bus_real_data setup.

## Current main world

Use:

```text
bus_real_data_moving_camera.sdf
```

It contains:

- BeIntelli bus model
- static cameras
- A4 ArUco markers
- moving calibration camera

## Intermediate worlds

### bus_real_data_camera_layout.sdf

Static camera layout only.

### bus_real_data_a4_markers.sdf

Static cameras plus A4 ArUco markers.

These intermediate worlds are kept for reproducibility/debugging.

## Launch

```bash
cd "/mnt/c/Users/maxim/Desktop/Application of Robotics and Autonomous Systems/camera-rig-calibration"

unset IGN_GAZEBO_RESOURCE_PATH
unset GZ_SIM_RESOURCE_PATH

REPO="$PWD"

export IGN_GAZEBO_RESOURCE_PATH="$REPO/src/calib_lab/bus_real_data/models:$REPO/src/calib_lab/beintelli_bus_model/models"
export GZ_SIM_RESOURCE_PATH="$IGN_GAZEBO_RESOURCE_PATH"

ign gazebo -r "$REPO/src/calib_lab/bus_real_data/worlds/bus_real_data_moving_camera.sdf"
```
