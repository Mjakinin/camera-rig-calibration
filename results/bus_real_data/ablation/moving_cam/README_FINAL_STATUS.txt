MOVING-CAMERA ABLATION FINAL STATUS
===================================

res/
- Status: COMPLETE
- Final file:
  results/bus_real_data/ablation/moving_cam/res/final_results/MOVING_CAM_RES_CLEAN_FINAL_COMPARISON.txt
- Contains source-gated AP01/AP02/AP03 camera-to-GT comparison.
- AP03 is valid for 640x360, 960x540, and 1280x720.
- AP03 is excluded for 320x180 and 1920x1080 due to incomplete COLMAP static-camera registration.

fov/
- Status: CLEANED BUT INCOMPLETE
- Final file:
  results/bus_real_data/ablation/moving_cam/fov/final_results/MOVING_CAM_FOV_CLEAN_FINAL_COMPARISON.txt
- fov_40deg has partial method outputs only.
- AP01 needs final RES-style camera-map SE(3) re-evaluation.
- AP02 failed because cam_edge_5 estimated static-camera pose is missing.
- AP03 was not run / no final Multi-ArUco output found.

Finalization rule for future ablations:
- Keep exactly one final TXT in each ablation folder:
  <ablation>/final_results/<ABLATION>_CLEAN_FINAL_COMPARISON.txt
- Keep invalid method/variant combinations as explicit status rows.
- Reject copied hashes, intermediate outputs, and incomplete camera sets.
