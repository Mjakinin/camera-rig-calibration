# bus_real_data Models

This folder contains Gazebo/Ignition model assets used by the `bus_real_data` worlds.

## Main content

```text
beintelli_bus/
```

Vehicle model used as the central object in the benchmark world.

```text
a4_aruco_marker_*/
```

Generated A4 ArUco marker models. Each marker directory usually contains a `model.config` and `model.sdf` describing one marker model.

## Model path

When launching Gazebo manually, include this folder in the Gazebo resource path:

```bash
REPO="$PWD"
export IGN_GAZEBO_RESOURCE_PATH="$REPO/src/calib_lab/bus_real_data/models"
export GZ_SIM_RESOURCE_PATH="$IGN_GAZEBO_RESOURCE_PATH"
```

## Editing rule

Generated marker model folders should normally be regenerated from scripts instead of edited by hand. The marker layout source-of-truth is in:

```text
src/calib_lab/bus_real_data/config/a4_marker_placements.json
```
