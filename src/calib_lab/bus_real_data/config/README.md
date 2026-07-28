# bus_real_data Config

This folder contains configuration files for the current `bus_real_data` Gazebo setup.

These files are source-of-truth inputs for world generation, marker placement, route generation and camera metadata. They should be treated as configuration, not generated calibration results.

## Important files

```text
camera_intrinsics_by_camera.yaml
```

Camera intrinsics used by the simulation setup and by OpenCV/PnP-based calibration steps.

```text
a4_marker_placements.json
```

A4 ArUco marker poses in the bus world. This controls where known markers appear in Gazebo.

```text
central_aruco_reference.json
```

Central/reference marker metadata used by the calibration setup. Marker 14 is the main Ref-ArUco anchor used in the current AP03 registration.

```text
moving_camera_route1_keyframes_final.json
moving_camera_route2_keyframes_final.json
```

Manual keyframes for the two retained moving-camera routes.

```text
moving_camera_route1_interpolated_final.json
moving_camera_route2_interpolated_final.json
```

Dense generated routes used by capture tooling. Route 2 is the selected
baseline; Route 1 is retained as an acquisition-robustness ablation.

```text
target_transforms.json
```

Camera/target transform configuration used by setup scripts.

```text
README_ALIGNMENT.md
```

Notes about transform conventions, Gazebo link frames and OpenCV optical frames.

## Subfolders

```text
camera_info/
```

Camera-info YAML files for the named cameras.

```text
marker_snapshots/
```

Saved marker placement snapshots for reproducibility/debugging.

## Editing rule

Edit keyframes and placement config first, then regenerate worlds/scripts outputs. Do not manually edit generated result CSVs to compensate for configuration mistakes.
