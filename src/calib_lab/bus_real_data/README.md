# bus_real_data Setup

This folder contains the Gazebo/Ignition setup for the current BeIntelli/Intellibus-style camera-rig calibration benchmark.

## Purpose

Reference camera:

```text
cam_edge_3
```

Calibration graph:

```text
cam_edge_3 -> cam_edge_1:
  direct static ArUco calibration

cam_edge_3 -> cam_edge_0:
  moving-camera relay calibration

cam_edge_3 -> cam_edge_5:
  moving-camera relay calibration
```

## Folder structure

```text
config/
```

Camera intrinsics, marker placements, moving-camera route keyframes and alignment notes.

```text
models/
```

Local models and generated A4 ArUco marker assets.

```text
scripts/
```

World-generation scripts.

```text
worlds/
```

Gazebo/Ignition SDF worlds.

## Main world

```text
worlds/bus_real_data_moving_camera.sdf
```

## Marker geometry

```text
A4 sheet:      0.210 m x 0.297 m
ArUco marker: 0.170 m x 0.170 m
```

Ground truth is used only for simulation evaluation, not for the final no-GT COLMAP-motion relay estimate.
