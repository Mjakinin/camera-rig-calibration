# bus_real_data World-Building Scripts

This folder contains setup/build scripts for generating marker models and SDF worlds.

These scripts are not the main calibration pipelines. The calibration/evaluation pipelines live in:

```text
src/camera_rig_calibration/
```

## Typical scripts

```text
01_build_world_from_real_camera_layout.py
```

Builds the static camera layout world from the camera layout/configuration.

```text
02_generate_a4_single_aruco_marker_models.py
```

Generates A4 single-marker Gazebo models.

```text
03_build_world_with_a4_markers.py
```

Builds the static-camera world with A4 ArUco markers.

```text
04_build_world_with_moving_camera.py
```

Builds the current main world with static cameras, A4 markers and moving calibration camera.

## Output

The main generated world is:

```text
src/calib_lab/bus_real_data/worlds/bus_real_data_moving_camera.sdf
```

## Marker geometry

```text
A4 sheet:      0.210 m x 0.297 m
ArUco marker: 0.170 m x 0.170 m
```

## Editing rule

Use these scripts to regenerate worlds after changing camera config, marker placements or route config. Do not treat generated worlds as independent source-of-truth unless a manual edit is explicitly intended.
