# Final Extrinsics, cam_edge_3 Reference

This folder contains the final camera rig extrinsics with `cam_edge_3` as reference.

## Main files

```text
FINAL_CAMERA_RIG_OVERVIEW.txt
```

Short human-readable final overview with estimated poses, GT poses and errors.

```text
FINAL_CAMERA_RIG_OVERVIEW.md
```

Markdown version of the final overview.

```text
final_extrinsics_summary.csv
```

Compact table of selected final estimates.

```text
final_extrinsics_cam3_reference.json
```

Full 4x4 transforms, translations, Euler angles and quaternions.

```text
pairwise_extrinsics_summary.csv
```

All derived camera-to-camera pairs from the final `cam_edge_3` reference estimates.

## Included estimates

```text
cam_edge_3 -> cam_edge_1:
  direct static ArUco baseline

cam_edge_3 -> cam_edge_0:
  moving-camera relay with COLMAP motion and ArUco metric scale

cam_edge_3 -> cam_edge_5:
  moving-camera relay with COLMAP motion and ArUco metric scale
```

## Ground truth usage

Ground truth is used only to compute evaluation errors.

The COLMAP relay estimates use no-GT metric scale from known 0.170 m ArUco markers.
