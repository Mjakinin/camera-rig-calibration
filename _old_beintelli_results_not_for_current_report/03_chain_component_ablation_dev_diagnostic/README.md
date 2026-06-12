# 03 Chain Component Ablation

This folder compares different levels of ground-truth and estimated components.

Cases:

## all_gt / gt_anchors_gt_moving
Uses:
- GT static camera poses
- GT board poses
- GT moving-camera pose at the exact selected frames

Expected:
- zero error
- validates transform algebra and frame conventions

## gt_anchors_colmap_moving
Uses:
- GT static camera poses
- GT board poses
- COLMAP moving-camera relative motion

Purpose:
- isolates the moving-camera trajectory error from COLMAP

## pnp_anchors_gt_moving
Uses:
- PnP-estimated board-camera anchor links
- GT moving-camera relative motion

Purpose:
- isolates ArUco/PnP anchor errors

## pnp_anchors_colmap_moving
Uses:
- PnP-estimated board-camera anchor links
- COLMAP moving-camera relative motion

Purpose:
- closest controlled reproduction of the no-GT chain

Important:
GT moving should be best when all other components are held constant.
If mixed cases appear better or worse, this can be caused by error compensation between PnP anchor errors and COLMAP motion errors.
