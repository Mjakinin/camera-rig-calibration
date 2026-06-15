# Full Pipeline Ablation: Moving Camera Brightness 200

Branch: ablation-study
Commit: a8e633b

Degradation:
- Type: moving-camera-only lighting degradation
- Mode: brightness
- Value: 200
- Static camera detections: baseline / clean
- Moving camera sequence: brightened

Detection:
- Frames: 131
- Total detections: 225
- Missing marker IDs: []
- Max consecutive frames without marker: 6

COLMAP:
- Registered images in best model: 54
- COLMAP poses loaded in relay: 53
- ArUco metric scale: 0.511757742881

Final COLMAP-motion estimates:
- cam_edge_3 -> cam_edge_0:
  - translation error: 14.37 cm
  - rotation error: 2.39 deg
  - selected chain: marker 7 frame 80 -> marker 4 frame 71

- cam_edge_3 -> cam_edge_5:
  - translation error: 9.11 cm
  - rotation error: 4.24 deg
  - selected chain: marker 7 frame 80 -> marker 12 frame 92

Baseline comparison:
- Baseline cam_edge_3 -> cam_edge_0 COLMAP: 20.91 cm, 2.24 deg
- Baseline cam_edge_3 -> cam_edge_5 COLMAP: 9.72 cm, 4.08 deg

Interpretation:
The brightness_200 ablation preserves all ArUco marker IDs. COLMAP registration
is reduced compared with the baseline, but the remaining registered frames are
sufficient for valid relay calibration. Final extrinsics remain close to the
baseline, especially for cam_edge_5. This indicates that brightness scaling is
less destructive than motion blur in this setup.
