# bus_real_data Config

This folder contains configuration files for the current bus_real_data setup.

## Important files

### camera_intrinsics_by_camera.yaml

Camera intrinsics used by OpenCV ArUco/PnP and moving-camera detection.

### a4_marker_placements.json

Current A4 ArUco marker placements in the bus world.

### moving_camera_route_keyframes.json

Manual moving-camera keyframes.

Edit this file to change the moving-camera path.

### moving_camera_route_interpolated.json

Generated dense route used by capture and preview scripts.

Generated from:

```text
moving_camera_route_keyframes.json
```

via:

```bash
python3 run/bus_real_data/02_generate_moving_camera_route.py
```

### target_transforms.json

Reference camera transform information used when building the camera layout.

### README_ALIGNMENT.md

Notes about Gazebo link frames, OpenCV optical frames and transform conventions.

## Subfolders

### camera_info/

Original camera info files.

### marker_snapshots/

Baseline marker placement snapshot. Kept for reproducibility.

## Marker geometry

```text
A4 sheet:      0.210 m x 0.297 m
ArUco marker: 0.170 m x 0.170 m
```
