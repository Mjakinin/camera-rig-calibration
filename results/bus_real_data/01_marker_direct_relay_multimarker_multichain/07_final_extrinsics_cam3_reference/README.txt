Final extrinsics summary, cam_edge_3 reference
================================================

Main calibration root:
  cam_edge_3

Pipeline:
  1. cam_edge_3 -> cam_edge_0 via moving-camera relay multichain + COLMAP motion + ArUco metric scale.
  2. cam_edge_3 -> cam_edge_1 via direct static ArUco multimarker overlap.
  3. cam_edge_3 -> cam_edge_5 via moving-camera relay multichain + COLMAP motion + ArUco metric scale.

Main no-GT results:
  cam_edge_0: 12.052 cm, 2.876 deg via moving_relay_multichain_colmap_motion_aruco_metric_scale
  cam_edge_1: 12.597 cm, 1.862 deg via direct_static_aruco_multimarker_weighted_mad_inliers
  cam_edge_5: 10.507 cm, 2.837 deg via moving_relay_multichain_colmap_motion_aruco_metric_scale

Multichain rule:
  All valid marker/frame combinations are evaluated.
  Outliers are removed without GT using median+3*MAD consistency filtering.
  Final relay estimate is the weighted mean of inlier transforms.

Marker14:
  Marker14 export is GT/evaluation-only and does not replace the cam_edge_3-rooted pipeline.
