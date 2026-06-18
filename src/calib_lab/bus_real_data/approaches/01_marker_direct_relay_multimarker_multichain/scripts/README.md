# bus_real_data World-Building Scripts

These scripts generate or update the Gazebo/Ignition world files and marker assets.

They are setup/build scripts, not the main experiment pipeline.

The main calibration/evaluation pipeline is in:

```text
run/bus_real_data/
```

## Scripts

### 01_build_world_from_real_camera_layout.py

Builds the static camera layout world.

### 02_generate_a4_single_aruco_marker_models.py

Generates A4 single ArUco marker models.

Marker geometry:

```text
A4 sheet:      0.210 m x 0.297 m
ArUco marker: 0.170 m x 0.170 m
```

### 03_build_world_with_a4_markers.py

Builds the world with static cameras and A4 ArUco markers.

### 04_build_world_with_moving_camera.py

Builds the current main world with static cameras, A4 markers and the moving calibration camera.

## Main generated world

```text
src/calib_lab/bus_real_data/worlds/bus_real_data_moving_camera.sdf
```
