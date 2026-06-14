# Ablation: Gaussian Blur k=5

Branch: ablation-study
Commit: a097dba

Degradation:
- Mode: gaussian_blur
- Kernel size: 5
- Input: results/bus_real_data/03_moving_camera_sequence
- Output: results/bus_real_data/03_moving_camera_sequence_blur_k5

Detection command:

```bash
python3 run/bus_real_data/05_detect_moving_a4_markers.py \
  --sequence results/bus_real_data/03_moving_camera_sequence_blur_k5 \
  --clean-debug \
  --no-debug-images \
  --no-axes
```

Result:
- Frames: 131
- Total detections: 222
- Unique detected IDs: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
- Missing IDs: []
- Max consecutive frames without marker: 6

Interpretation:
- Moderate Gaussian blur still preserves all marker IDs.
- Detection count is slightly higher than blur_k3 and should be compared against baseline.
