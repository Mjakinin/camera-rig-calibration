V7 Multi-Station ArUco Relay Result
===================================

Goal:
Estimate the transform T_front_rear between front_static_camera and rear_static_camera using a moving_calib_camera relay, COLMAP trajectory estimation, and ArUco GridBoard anchor observations.

Important note:
This is a simulation proof-of-concept. The current relay script aligns the COLMAP trajectory to Gazebo ground truth using a Sim(3) alignment for metric scale and evaluation. In a real setup, this GT-assisted alignment must be replaced by a metric prior such as known target constraints, multi-target observations, odometry, VIO, IMU, stereo/RGB-D, or known board/map geometry.

Active ArUco board:
- model: aruco_gridboard_target_a1_080x060
- size: 0.8 x 0.6 m
- texture: 1440 x 1080 px
- marker_px: 300
- gap_px: 80
- dictionary: DICT_4X4_50
- IDs: 0..5

V7 COLMAP route:
- dataset: results/beintelli_bus_model/colmap/moving_route_v7_all_success_anchors_static_world
- route frames: 350
- registered frames: 350
- registration rate: 100%
- position RMSE: 11.33 cm
- position mean: 5.89 cm
- position median: 4.66 cm
- position max: 127.86 cm
- note: first F3 hold frames contained outliers; selected best anchor frames were used.

Selected anchor frames:
- F3: moving_0002.jpg
- F4: moving_0061.jpg
- R1: moving_0343.jpg
- R2: moving_0289.jpg
- R3: moving_0240.jpg

Success stations:
- F3_front_near_left_seat
- F4_front_right_table_or_box
- R1_rear_left_seat_leaned
- R2_rear_table_flat
- R3_rear_right_seat_angled

Failure / limit-case stations:
- F1_front_mid_far_seat_leaned: front_static_camera detected 0 markers
- F2_front_mid_high_left_seat: front_static_camera detected 0 markers
- G1_floor_mid_yaw_pi: front/rear static cameras detected 0 markers
- G2_floor_mid_yaw_0: front/rear static cameras detected 0 markers

Pair results:
- F3-R1: translation error 9.34 cm, rotation error 1.88 deg
- F3-R2: translation error 8.95 cm, rotation error 1.38 deg
- F3-R3: translation error 3.81 cm, rotation error 1.06 deg
- F4-R1: translation error 17.81 cm, rotation error 2.72 deg
- F4-R2: translation error 7.93 cm, rotation error 0.21 deg
- F4-R3: translation error 11.75 cm, rotation error 1.51 deg

Interpretation:
The replay-based ArUco relay pipeline works in simulation for no-overlap static camera calibration. The moving camera connects front and rear static cameras through COLMAP-estimated trajectory segments and ArUco board anchor observations. Station selection strongly affects final calibration accuracy.
