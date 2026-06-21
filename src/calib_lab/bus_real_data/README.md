# bus_real_data Simulation Setup

`src/calib_lab/bus_real_data` is the current active Gazebo/Ignition setup for the bus camera-rig calibration benchmark.

It defines the simulated BeIntelli/Intellibus-style scene: static cameras, the moving calibration camera, A4 ArUco reference targets, vehicle model assets and SDF worlds.

## Folder structure

```text
config/
```

Camera intrinsics, camera-info YAMLs, marker placements, moving-camera route configuration and transform/alignment notes.

```text
models/
```

Gazebo model assets. This includes the bus model and generated A4 ArUco marker models.

```text
scripts/
```

World-building and asset-generation scripts. These create/update SDF worlds and generated marker models.

```text
worlds/
```

Gazebo/Ignition SDF worlds. The main final world is `bus_real_data_moving_camera.sdf`.

## Main world

```text
worlds/bus_real_data_moving_camera.sdf
```

This is the main current benchmark world. It contains:

- the BeIntelli bus model
- static cameras around the bus
- A4 ArUco markers
- moving calibration camera

## Marker geometry

```text
A4 sheet:      0.210 m x 0.297 m
ArUco marker: 0.170 m x 0.170 m
Dictionary:   DICT_4X4_50
```

## Relationship to results

This folder defines the simulation setup. The generated real-data style images and observations used by the calibration pipelines live under:

```text
results/bus_real_data/00_shared_baseline/bus_real_data_ref_marker_v1/
```

The active raw-image path is:

```text
results/bus_real_data/00_shared_baseline/bus_real_data_ref_marker_v1/raw_images
```

Do not reintroduce the legacy raw-image path:

```text
results/bus_real_data/00_raw_images/bus_real_data_ref_marker_v1
```
