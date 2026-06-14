# Ablation: Gaussian Blur k=9

Branch: ablation-study
Commit: 143aa32

Degradation:
- Mode: gaussian_blur
- Kernel size: 9
- Input: results/bus_real_data/03_moving_camera_sequence
- Output: results/bus_real_data/03_moving_camera_sequence_blur_k9

Detection command:

```bash
python3 run/bus_real_data/05_detect_moving_a4_markers.py \
  --sequence results/bus_real_data/03_moving_camera_sequence_blur_k9 \
  --clean-debug \
  --no-debug-images \
  --no-axes
```

Result:
- Frames: 131
- Total detections: 220
- Unique detected IDs: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
- Missing IDs: []
- Max consecutive frames without marker: 6

Interpretation:
- Strong Gaussian blur still preserves all marker IDs.
- ArUco detection remains robust for this synthetic sequence.
