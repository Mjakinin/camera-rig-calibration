# Simple Interpretation of Chain Component Ablation

The front-rear transform chain consists of three logical parts:

T_front_rear =
front camera-board connection
* moving-camera motion
* rear camera-board connection

## What are camera-board connections?

The camera-board connections are the transforms estimated through the ArUco boards:

- front_static_camera to front_board
- moving_calib_camera at the front frame to front_board
- moving_calib_camera at the rear frame to rear_board
- rear_static_camera to rear_board

These connections can either come from ground truth board/camera poses or from ArUco/PnP estimates.

## Cases

### all_gt

Uses:
- GT static camera poses
- GT board poses
- GT moving-camera poses at the exact selected frames

Result:
- zero error

Meaning:
- the transform chain and frame conventions are correct.

### gt_board_links_colmap_moving

Uses:
- GT camera-board connections
- COLMAP moving-camera motion

Meaning:
- this isolates the COLMAP moving-camera motion error.

### pnp_board_links_gt_moving

Uses:
- ArUco/PnP estimated camera-board connections
- GT moving-camera motion

Meaning:
- this isolates the ArUco/PnP camera-board connection error.

### pnp_board_links_colmap_moving

Uses:
- ArUco/PnP estimated camera-board connections
- COLMAP moving-camera motion

Meaning:
- this is the final no-GT pipeline.

Important:
This final case can look better than the two isolated error cases because the ArUco/PnP errors and COLMAP motion errors can partially compensate each other.

This does NOT mean COLMAP is better than GT.
It means that the final composed transform has lower residual error in this specific setup.

## Main conclusion

The all-GT case gives zero error, so the transform chain is correct.

The final no-GT pipeline works as a proof of concept.

Best current no-GT pair:
F3_R3

F3_R3:
- translation error: 5.03 cm
- rotation error: 1.24 deg

Next step:
Run more trajectories, more board placements and noise/FOV variations to check whether this error compensation is stable or accidental.
