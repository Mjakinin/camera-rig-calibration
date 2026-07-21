# bus_real_data Worlds

This folder contains Gazebo/Ignition SDF worlds for the `bus_real_data` setup.

## Main world

```text
bus_real_data_moving_camera.sdf
```

Current benchmark world with:

- BeIntelli bus model
- static cameras
- A4 ArUco markers
- moving calibration camera

## Intermediate/debug worlds

```text
bus_real_data_camera_layout.sdf
```

Static camera layout world.

```text
bus_real_data_a4_markers.sdf
```

Static camera layout plus A4 ArUco markers.

These intermediate worlds are useful for debugging and for reproducing incremental setup steps.

## Launch

```bash
cd /workspaces/project

unset IGN_GAZEBO_RESOURCE_PATH
unset GZ_SIM_RESOURCE_PATH

REPO="$PWD"
export IGN_GAZEBO_RESOURCE_PATH="$REPO/src/calib_lab/bus_real_data/models"
export GZ_SIM_RESOURCE_PATH="$IGN_GAZEBO_RESOURCE_PATH"

ign gazebo -r "$REPO/src/calib_lab/bus_real_data/worlds/bus_real_data_moving_camera.sdf"
```

## Editing rule

Prefer changing config files and regenerating worlds via scripts. Manual SDF edits are okay for small fixes, but they should be documented and checked in Gazebo.
