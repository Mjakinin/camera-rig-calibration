# Full Pipeline Ablation: Moving Camera Brightness 50

Branch: ablation-study
Commit: 082d1dc

Degradation:
- Type: moving-camera-only lighting degradation
- Mode: brightness
- Value: 50
- Static camera detections: baseline / clean
- Moving camera sequence: darkened

Purpose:
This experiment evaluates whether the moving-camera relay pipeline remains
valid under reduced image brightness.

Pipeline:
- Moving ArUco detections are re-run on darkened moving-camera frames.
- COLMAP is re-run on the darkened moving-camera sequence.
- ArUco-based metric scale is estimated from the darkened sequence.
- Relay chains are evaluated using clean static anchors and degraded moving data.
- Final extrinsics are exported relative to cam_edge_3.

Interpretation:
This run isolates lighting sensitivity in the moving-camera relay capture.
