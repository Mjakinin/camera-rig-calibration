# 04 Front-Rear Chain: No-GT COLMAP + ArUco Relay

This folder contains the final front_static_camera -> rear_static_camera calibration results.

What is estimated:
T_front_rear = pose of rear_static_camera in the coordinate frame of front_static_camera.

Pipeline:
1. front_static_camera observes a front ArUco board.
2. moving_calib_camera observes the same front board.
3. COLMAP estimates the moving-camera motion from the front-side frame to the rear-side frame.
4. moving_calib_camera observes a rear ArUco board.
5. rear_static_camera observes the same rear board.
6. These transforms are composed into T_front_rear.

No-GT method:
The estimation does not use route_gt.csv.
The moving-camera motion comes from COLMAP.
The metric scale comes from ArUco-PnP observations of known-size boards.
Ground truth is only used after estimation for evaluation.

Valid station selection:
A valid front anchor must be visible by:
- front_static_camera
- moving_calib_camera

A valid rear anchor must be visible by:
- rear_static_camera
- moving_calib_camera

Valid front anchors:
- F3
- F4

Valid rear anchors:
- R1
- R3

Therefore the evaluated valid pairs are:
- F3_R1
- F3_R3
- F4_R1
- F4_R3

Other stations are not used because:
- F1/F2: moving-camera observation is not reliable enough
- R2: rear_static_camera sees it, but moving_calib_camera does not see enough markers
- G: floor/general limit case, not used as front-rear anchor

Main metric:
translation_error_cm = full 3D translation error of T_front_rear against static-camera GT.

Current best pair by full 3D translation error:
F3_R3

F3_R3 result:
- baseline_est_m:       9.620700
- baseline_gt_m:        9.600000
- baseline_error_cm:    2.07
- translation_error_cm: 5.03
- rotation_error_deg:   1.24
