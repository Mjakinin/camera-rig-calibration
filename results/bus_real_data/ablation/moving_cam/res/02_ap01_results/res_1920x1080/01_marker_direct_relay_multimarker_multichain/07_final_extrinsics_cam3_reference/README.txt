Final extrinsics summary, cam_edge_3 reference
================================================

Main calibration root:
  cam_edge_3

Pipeline:
  1. cam_edge_3 -> cam_edge_0 via moving-camera relay multichain + COLMAP motion + ArUco metric scale.
  2. cam_edge_3 -> cam_edge_1 via moving-camera relay multichain + COLMAP motion + ArUco metric scale.
  3. cam_edge_3 -> cam_edge_5 via moving-camera relay multichain + COLMAP motion + ArUco metric scale.

Main no-GT results:
  cam_edge_0: 7.572 cm, 2.164 deg via moving_relay_multichain_colmap_motion_aruco_metric_scale
  cam_edge_1: 39.595 cm, 4.292 deg via moving_relay_multichain_colmap_motion_aruco_metric_scale
  cam_edge_5: 6.590 cm, 2.793 deg via moving_relay_multichain_colmap_motion_aruco_metric_scale

Multichain rule:
  All valid marker/frame combinations are evaluated.
  Outliers are removed without GT using median+3*MAD consistency filtering.
  Final relay estimate is the weighted mean of inlier transforms.

Marker14:
  Marker14 export is GT/evaluation-only and does not replace the cam_edge_3-rooted pipeline.
