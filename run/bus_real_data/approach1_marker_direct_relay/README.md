# AP01: Marker Direct Relay + Moving-Camera Multichain

AP01 estimates a camera rig using ArUco marker observations, direct static-camera marker overlap, a moving-camera COLMAP trajectory, and multichain relay aggregation.

## Method summary

- `cam_edge_3` is the deployable reference camera.
- `cam_edge_1` is estimated through direct static-camera multi-marker overlap.
- `cam_edge_0` and `cam_edge_5` are estimated through moving-camera relay chains.
- COLMAP motion is scaled using ArUco marker side length, not Gazebo ground truth.
- Ground truth is used only for evaluation.

## Main scripts

```text
06_run_colmap_moving_sequence.py
07_evaluate_colmap_position_vs_gt.py
08_make_colmap_error_tables.py
09_evaluate_colmap_rotation_vs_gt.py
10_eval_direct_static_cam3_cam1.py
11_make_direct_static_report_cam3_cam1.py
12_estimate_colmap_scale_from_aruco.py
13_eval_direct_static_cam3_cam1_multimarker.py
14_eval_moving_relay_chains.py
15_export_final_extrinsics_cam3_reference.py
run_approach1_full_pipeline.sh
```

## Internal compatibility cache

AP01 uses a hidden generated cache:

```text
results/bus_real_data/01_marker_direct_relay_multimarker_multichain/.ap01_compat_cache/
```

This cache is generated from the shared baseline and should not be committed.

## Final result

```text
results/bus_real_data/01_marker_direct_relay_multimarker_multichain/07_final_extrinsics_cam3_reference/
```

Validated final errors:

```text
cam_edge_1: 12.597 cm, 1.862 deg
cam_edge_0:  8.121 cm, 2.113 deg
cam_edge_5:  5.135 cm, 2.790 deg
```
