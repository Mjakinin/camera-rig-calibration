# Full Pipeline Ablation: Moving Camera Brightness 50

Branch: ablation-study
Commit: 94125d8

Degradation:
- Type: moving-camera-only lighting degradation
- Mode: brightness
- Value: 50
- Static camera detections: baseline / clean
- Moving camera sequence: darkened

Detection:
- Frames: 131
- Total detections: 225
- Missing marker IDs: []
- Max consecutive frames without marker: 6

COLMAP:
- Registered images in best model: 51
- COLMAP poses loaded in relay: 50
- ArUco metric scale: 0.476305729033

Final COLMAP-motion estimates:
- cam_edge_3 -> cam_edge_0:
  - translation error: 15.95 cm
  - rotation error: 4.04 deg
  - selected chain: marker 7 frame 80 -> marker 4 frame 71

- cam_edge_3 -> cam_edge_5:
  - translation error: 21.39 cm
  - rotation error: 5.08 deg
  - selected chain: marker 8 frame 91 -> marker 11 frame 91

Baseline comparison:
- Baseline cam_edge_3 -> cam_edge_0 COLMAP: 20.91 cm, 2.24 deg
- Baseline cam_edge_3 -> cam_edge_5 COLMAP: 9.72 cm, 4.08 deg

Interpretation:
The moving-camera brightness_50 ablation preserves all ArUco marker IDs, but
strongly reduces COLMAP registration. This suggests that low brightness affects
feature-based reconstruction more than marker-based detection. Relay calibration
remains possible, but final extrinsics change, especially for cam_edge_5.
