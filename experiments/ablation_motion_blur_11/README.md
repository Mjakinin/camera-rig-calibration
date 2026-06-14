# Ablation: Motion Blur length=11

Branch: ablation-study
Commit: 1d2836e

Degradation:
- Mode: motion_blur
- Length: 11
- Input: results/bus_real_data/03_moving_camera_sequence
- Output: results/bus_real_data/03_moving_camera_sequence_motion_blur_11

Detection command:

```bash
python3 run/bus_real_data/05_detect_moving_a4_markers.py \
  --sequence results/bus_real_data/03_moving_camera_sequence_motion_blur_11 \
  --clean-debug \
  --no-debug-images \
  --no-axes
```

Result:
- Frames: 131
- Total detections: 133
- Unique detected IDs: [0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13]
- Missing IDs: [4]
- Max consecutive frames without marker: 7

Interpretation:
- Motion blur length 11 is a transition point.
- Marker ID coverage is no longer complete.
- Marker 4 is lost completely, which is critical for cam3 -> cam0 relay chains.
