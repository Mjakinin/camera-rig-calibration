# Ablation: Motion Blur length=15

Branch: ablation-study
Commit: 76b7eaf

Degradation:
- Mode: motion_blur
- Length: 15
- Input: results/bus_real_data/03_moving_camera_sequence
- Output: results/bus_real_data/03_moving_camera_sequence_motion_blur_15

Detection command:

```bash
python3 run/bus_real_data/05_detect_moving_a4_markers.py \
  --sequence results/bus_real_data/03_moving_camera_sequence_motion_blur_15 \
  --clean-debug \
  --no-debug-images \
  --no-axes
```

Result:
- Frames: 131
- Total detections: 91
- Unique detected IDs: [0, 1, 3, 5, 8, 9, 10, 11, 12, 13]
- Missing IDs: [2, 4, 6, 7]
- Max consecutive frames without marker: 24

Interpretation:
- Directed motion blur causes a major ArUco detection degradation.
- Missing marker 4 is especially important for cam3 -> cam0 relay.
- Missing marker 7 removes one of the strongest root-side COLMAP relay anchors.
