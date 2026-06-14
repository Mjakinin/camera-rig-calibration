# Ablation: Motion Blur length=7

Branch: ablation-study
Commit: 12d1189

Degradation:
- Mode: motion_blur
- Length: 7
- Input: results/bus_real_data/03_moving_camera_sequence
- Output: results/bus_real_data/03_moving_camera_sequence_motion_blur_7

Detection command:

```bash
python3 run/bus_real_data/05_detect_moving_a4_markers.py \
  --sequence results/bus_real_data/03_moving_camera_sequence_motion_blur_7 \
  --clean-debug \
  --no-debug-images \
  --no-axes
```

Result:
- Frames: 131
- Total detections: 163
- Unique detected IDs: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
- Missing IDs: []
- Max consecutive frames without marker: 6

Interpretation:
- Moderate directed motion blur reduces the number of marker detections.
- However, all expected marker IDs are still observed at least once.
- Compared with motion_blur_15, this shows a gradual degradation trend.
