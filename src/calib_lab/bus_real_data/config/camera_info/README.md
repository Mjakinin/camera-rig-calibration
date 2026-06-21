# Camera Info Configs

`camera_info` contains per-camera YAML metadata for the simulated bus camera setup.

These files describe camera intrinsics and image geometry used by ROS/Gazebo and by calibration tooling. They are part of the simulation contract: changing them changes the rendered data and downstream calibration assumptions.

## Typical files

Examples include camera-info files for cameras such as:

```text
front_left.yaml
front_right.yaml
center_left.yaml
back_right.yaml
```

The exact camera set may evolve with the bus layout, but file names should remain aligned with the camera names used in worlds, routes and results.

## Usage

These files are read by setup/build scripts and are conceptually mirrored by generated `camera_info` JSON files in the shared result baseline:

```text
results/bus_real_data/00_shared_baseline/bus_real_data_ref_marker_v1/raw_images/camera_info/
```

## Editing rule

When changing intrinsics, regenerate the relevant worlds/data and rerun the calibration pipelines. Mixing old image data with new camera-info configuration invalidates the evaluation.
