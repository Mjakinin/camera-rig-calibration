# Ablation: Brightness 150%

Branch: ablation-study
Commit: 561a65f

Degradation:
- Mode: brightness
- Factor: 1.5
- Input: results/bus_real_data/03_moving_camera_sequence
- Output: results/bus_real_data/03_moving_camera_sequence_brightness_150

Detection command:

```bash
python3 run/bus_real_data/05_detect_moving_a4_markers.py \
  --sequence results/bus_real_data/03_moving_camera_sequence_brightness_150 \
  --clean-debug \
  --no-debug-images \
  --no-axes
```

Result:
- Frames: 131
- Total detections: 221
- Unique detected IDs: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
- Missing IDs: []
- Max consecutive frames without marker: 6

Interpretation:
- Increasing image brightness to 150% did not reduce marker ID coverage.
- ArUco detection remains stable under this overexposure-like degradation.
