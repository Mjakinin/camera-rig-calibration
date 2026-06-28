Final extrinsics summary, cam_edge_3 reference
================================================

Main calibration root:
  cam_edge_3

Pipeline:
  1. cam_edge_3 -> cam_edge_0 via moving-camera relay multichain + COLMAP motion + ArUco metric scale.
  2. cam_edge_3 -> cam_edge_1 via moving-camera relay multichain + COLMAP motion + ArUco metric scale.
  3. cam_edge_3 -> cam_edge_5 via moving-camera relay multichain + COLMAP motion + ArUco metric scale.

Main no-GT results:
  cam_edge_0: 10.154 cm, 2.198 deg via moving_relay_multichain_colmap_motion_aruco_metric_scale
  cam_edge_1: 49.667 cm, 4.767 deg via moving_relay_multichain_colmap_motion_aruco_metric_scale
  cam_edge_5: 552.320 cm, 3.927 deg via moving_relay_multichain_colmap_motion_aruco_metric_scale

Multichain rule:
  All valid marker/frame combinations are evaluated.
  Outliers are removed without GT using median+3*MAD consistency filtering.
  Final relay estimate is the weighted mean of inlier transforms.

Marker14:
  Marker14 export is GT/evaluation-only and does not replace the cam_edge_3-rooted pipeline.
