# Full Pipeline Ablation: Motion Blur length=7

Branch: ablation-study
Commit: 37cadf2

Degradation:
- Mode: motion_blur
- Length: 7
- Input sequence: results/bus_real_data/03_moving_camera_sequence
- Degraded sequence: results/bus_real_data/03_moving_camera_sequence_motion_blur_7

Detection:
- Frames: 131
- Total detections: 163
- Missing marker IDs: []
- Max consecutive frames without marker: 6

COLMAP:
- Registered poses loaded in relay: 106
- ArUco COLMAP metric scale: 0.769679645756

Final COLMAP-motion estimates:
- cam_edge_3 -> cam_edge_0:
  - translation error: 14.92 cm
  - rotation error: 5.89 deg
  - selected chain: marker 8 frame 91 -> marker 4 frame 72

- cam_edge_3 -> cam_edge_5:
  - translation error: 22.14 cm
  - rotation error: 2.89 deg
  - selected chain: marker 8 frame 91 -> marker 10 frame 105

Baseline comparison:
- Baseline cam_edge_3 -> cam_edge_0 COLMAP: 20.91 cm, 2.24 deg
- Baseline cam_edge_3 -> cam_edge_5 COLMAP: 9.72 cm, 4.08 deg

Interpretation:
- Motion blur length 7 reduces COLMAP registration from 129 baseline poses to 106 poses.
- The selected relay chains changed compared with the baseline.
- cam3 -> cam0 improves in translation but worsens in rotation.
- cam3 -> cam5 worsens in translation but improves in rotation.
