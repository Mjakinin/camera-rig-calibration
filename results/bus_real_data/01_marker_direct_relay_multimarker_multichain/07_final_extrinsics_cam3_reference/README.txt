AP01 calibration estimate output
================================

Reference camera: cam_edge_3

This directory contains estimator outputs, not the canonical
scientific comparison report.

Canonical evaluation:
- Primary: static camera-to-camera extrinsics for all six pairs.
- Secondary: SE(3)-aligned full static-camera map.

The former Ref14 GT-anchor report is intentionally disabled
because it assigned zero error to the anchor camera by construction.

Available AP01 target estimates:
- cam_edge_0 via moving_relay_multichain_colmap_motion_aruco_metric_scale
- cam_edge_1 via direct_static_aruco_multimarker_quality_filtered_preferred_marker_no_gt_selection
- cam_edge_5 via moving_relay_multichain_colmap_motion_aruco_metric_scale
