Final extrinsics summary, cam_edge_3 reference
================================================

Reference camera:
  cam_edge_3

This folder summarizes the current calibrated camera rig.

Files:
  final_extrinsics_summary.csv
    Human-readable summary table with selected estimates and errors.

  final_extrinsics_cam3_reference.json
    Full 4x4 transforms, translations, Euler angles and quaternions.

Included estimates:
  1. cam_edge_3 -> cam_edge_1
     Method: direct static ArUco
     Role: direct-static baseline with shared marker.

  2. cam_edge_3 -> cam_edge_0
     Method: moving-camera relay with COLMAP motion and ArUco metric scale.
     Role: real-life-near relay estimate.

  3. cam_edge_3 -> cam_edge_5
     Method: moving-camera relay with COLMAP motion and ArUco metric scale.
     Role: real-life-near relay estimate.

  4. GT_motion relay variants
     Role: oracle/sanity baselines only.
     Not real-life deployable.

Important:
  Ground truth is used only for evaluation errors.
  The COLMAP relay estimates use no-GT metric scale from known 0.17 m ArUco markers.

Generated from:
  run/bus_real_data/13_eval_moving_relay_chains.py
  results/bus_real_data/06_moving_relay_chain_eval/relay_chain_results.csv
