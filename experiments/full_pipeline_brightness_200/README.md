# Full Pipeline Ablation: Moving Camera Brightness 200

Branch: ablation-study
Commit: 72629fd

Degradation:
- Type: moving-camera-only lighting degradation
- Mode: brightness
- Value: 200
- Static camera detections: baseline / clean
- Moving camera sequence: brightened / potential overexposure

Purpose:
This experiment evaluates whether the moving-camera relay pipeline remains
valid under increased image brightness.

Pipeline:
- Moving ArUco detections are re-run on brightened moving-camera frames.
- COLMAP is re-run on the brightened moving-camera sequence.
- ArUco-based metric scale is estimated from the brightened sequence.
- Relay chains are evaluated using clean static anchors and degraded moving data.
- Final extrinsics are exported relative to cam_edge_3.

Interpretation:
This run isolates the effect of increased brightness / possible overexposure
on moving-camera marker detection, COLMAP reconstruction, metric scale
estimation, relay-chain selection, and final extrinsics.
