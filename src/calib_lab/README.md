# calib_lab ROS 2/Gazebo Package

`src/calib_lab` is the ROS 2 package and simulation-source area for the APPRAS camera-rig calibration project.

This folder is intentionally separated from the executable package in
`src/camera_rig_calibration/`. `src/calib_lab` defines the world, camera
layout, marker layout and model assets; the package consumes those assets and
publishes calibration/evaluation outputs in `results/`.

## Package role

The package provides:

- Gazebo/Ignition SDF worlds for the BeIntelli/Intellibus-style camera rig.
- Camera intrinsic/extrinsic configuration used to build worlds and evaluate results.
- A4 ArUco marker placement configuration and generated marker models.
- BeIntelli bus model assets used by the simulation world.
- Python utility scripts for generating/updating world files.

## Main subfolders

```text
bus_real_data/
```

Current main simulation setup. This is the active Gazebo world and configuration tree for the final bus real-data benchmark.

## ROS 2 package files

```text
CMakeLists.txt
package.xml
```

These identify `calib_lab` as an `ament_cmake` ROS 2 package. The current project mostly uses Python and Gazebo assets, but the ROS package structure keeps the simulation assets buildable and discoverable in a ROS workspace.

## Relationship to the pipelines

Use `src/calib_lab` when you want to change the simulation setup:

- camera placement
- camera intrinsics
- marker placements
- moving-camera route configuration
- generated SDF worlds
- model assets

Use the `rigcal` CLI when you want to run calibration approaches:

- AP01 marker direct relay
- AP02 reference-marker graph + BA
- AP03 targetless COLMAP + ArUco scale

Use `results/simulation` when you want to inspect generated outputs, reports
and final metrics.

## Main world

```text
bus_real_data/worlds/bus_real_data_moving_camera.sdf
```

The current world includes static edge cameras, A4 ArUco markers and the moving calibration camera.
