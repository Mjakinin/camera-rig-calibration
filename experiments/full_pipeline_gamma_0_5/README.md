# Full Pipeline Ablation: Moving Camera Gamma 0.5

Branch: ablation-study
Commit: 7b3e332

Degradation:
- Type: moving-camera-only lighting degradation
- Mode: gamma
- Value: 0.5
- Static camera detections: baseline / clean
- Moving camera sequence: gamma-adjusted

Detection:
- Frames: 131
- Total detections: 220
- Missing marker IDs: []
- Max consecutive frames without marker: 6

COLMAP:
- Registered images in best model: 131
- COLMAP poses loaded in relay: 131
- ArUco metric scale: 0.728339316525

Final COLMAP-motion estimates:
- cam_edge_3 -> cam_edge_0:
  - translation error: 17.54 cm
  - rotation error: 5.29 deg
  - selected chain: marker 2 frame 17 -> marker 4 frame 71

- cam_edge_3 -> cam_edge_5:
  - translation error: 11.80 cm
  - rotation error: 2.58 deg
  - selected chain: marker 1 frame 32 -> marker 10 frame 105

Baseline comparison:
- Baseline cam_edge_3 -> cam_edge_0 COLMAP: 20.91 cm, 2.24 deg
- Baseline cam_edge_3 -> cam_edge_5 COLMAP: 9.72 cm, 4.08 deg

Interpretation:
The gamma_0_5 ablation preserves all ArUco marker IDs and allows COLMAP to
register all moving-camera frames. Compared with linear brightness scaling,
this gamma adjustment is less harmful for feature-based reconstruction. Final
relay calibration remains valid for both cam_edge_0 and cam_edge_5.
