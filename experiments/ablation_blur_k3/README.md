# Ablation: Gaussian Blur k=3

Branch: ablation-study
Commit: 52334e2

Degradation:
- Mode: gaussian_blur
- Kernel size: 3
- Input: results/bus_real_data/03_moving_camera_sequence
- Output: results/bus_real_data/03_moving_camera_sequence_blur_k3

Detection command:

```bash
python3 run/bus_real_data/05_detect_moving_a4_markers.py \
  --sequence results/bus_real_data/03_moving_camera_sequence_blur_k3 \
  --clean-debug \
  --no-debug-images \
  --no-axes
```

Result:
- Frames: 131
- Total detections: 218
- Unique detected IDs: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
- Missing IDs: []
- Max consecutive frames without marker: 6

Interpretation:
- Mild Gaussian blur still preserves all marker IDs.
- Detection count and empty-frame gaps should be compared against baseline.
