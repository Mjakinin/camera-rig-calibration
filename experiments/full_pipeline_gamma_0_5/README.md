# Full Pipeline Ablation: Moving Camera Gamma 0.5

Branch: ablation-study
Commit: 95a66b7

Degradation:
- Type: moving-camera-only lighting degradation
- Mode: gamma
- Value: 0.5
- Static camera detections: baseline / clean
- Moving camera sequence: nonlinear gamma-adjusted sequence

Purpose:
This experiment evaluates whether the moving-camera relay pipeline remains
valid under nonlinear lighting / contrast change.

Pipeline:
- Moving ArUco detections are re-run on gamma-adjusted moving-camera frames.
- COLMAP is re-run on the gamma-adjusted moving-camera sequence.
- ArUco-based metric scale is estimated from the gamma-adjusted sequence.
- Relay chains are evaluated using clean static anchors and degraded moving data.
- Final extrinsics are exported relative to cam_edge_3.

Interpretation:
This run isolates the effect of nonlinear lighting change on moving-camera
marker detection, COLMAP reconstruction, metric scale estimation, relay-chain
selection, and final extrinsics.
