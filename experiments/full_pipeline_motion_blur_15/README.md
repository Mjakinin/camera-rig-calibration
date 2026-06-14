# Full Pipeline Ablation: Motion Blur length=15

Branch: ablation-study
Commit: be5c27e

Degradation:
- Mode: motion_blur
- Length: 15
- Input sequence: results/bus_real_data/03_moving_camera_sequence
- Degraded sequence: results/bus_real_data/03_moving_camera_sequence_motion_blur_15

Detection:
- Frames: 131
- Total detections: 91
- Missing marker IDs: [2, 4, 6, 7]
- Max consecutive frames without marker: 24

Pipeline:
- COLMAP reconstruction: completed
- ArUco metric scale estimation: completed
- Moving-camera relay evaluation: completed
- Final extrinsics export: completed

Interpretation:
- Motion blur length 15 causes severe marker detection degradation.
- Critical markers for relay chains are missing.
- This run should be interpreted as a strong image-degradation stress test.
