V6 ArUco Relay Success
======================

Goal:
Estimate T_front_rear between front_static_camera and rear_static_camera without a shared direct target view.

Method:
- A moving_calib_camera is used as relay.
- COLMAP estimates the moving-camera trajectory in a static Gazebo bus scene.
- The calibration board is parked during COLMAP capture.
- The same moving-camera anchor poses are replayed with the ArUco board placed at F3 and R1.
- ArUco board observations connect static cameras to selected moving-camera frames.

Active board:
- model: aruco_gridboard_target_a1_080x060
- board size: 0.8 x 0.6 m
- texture: 1440 x 1080 px
- marker_px: 300
- gap_px: 80
- dictionary: DICT_4X4_50
- IDs: 0..5

V6 COLMAP:
- dataset: results/beintelli_bus_model/colmap/moving_route_v6_anchor_optimized_static_world
- route frames: 190
- registered frames: 188
- registration rate: 98.9%
- position RMSE: 5.78 cm
- position mean: 4.93 cm
- position max: 10.07 cm

Anchor frames:
- F3 front anchor: moving_0012.jpg
- R1 rear anchor: moving_0188.jpg

Observations:
- F3 obs: results/beintelli_bus_model/colmap_anchor_observations/F3_moving_0012_correct_pose/aruco_board_pose_observations.csv
- R1 obs: results/beintelli_bus_model/colmap_anchor_observations/R1_moving_0312_correct_pose/aruco_board_pose_observations.csv

Final relay result:
- baseline_est_m: 9.697485
- baseline_gt_m: 9.600000
- baseline_error_cm: 9.75
- translation_error_cm: 11.86
- rotation_error_deg: 1.97

Stability check:
Alternative hold-frame pairs produced effectively the same result:
- moving_0010 + moving_0186: translation_error_cm 11.86
- moving_0005 + moving_0180: translation_error_cm 11.86

Interpretation:
This is the first successful end-to-end proof-of-concept for no-overlap static camera calibration using a moving-camera relay, COLMAP trajectory estimation, and ArUco GridBoard target anchors.
