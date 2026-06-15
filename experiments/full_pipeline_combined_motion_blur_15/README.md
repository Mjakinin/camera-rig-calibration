# Full Pipeline Ablation: Combined Static + Moving Motion Blur length=15

Branch: ablation-study
Commit: 9665926

Degradation:
- Static cameras: motion_blur length=15
- Moving camera: motion_blur length=15

Purpose:
This experiment evaluates the worst-case image degradation setting where both
the static bus camera anchor observations and the moving-camera relay sequence
are degraded simultaneously.

Pipeline:
- Static A4 marker detections are re-run on degraded static camera images.
- Moving A4 marker detections are re-run on degraded moving-camera frames.
- COLMAP is re-run on the degraded moving-camera sequence.
- ArUco-based metric scale is estimated from degraded moving-camera detections.
- Relay chains are evaluated using degraded static and degraded moving detections.
- Final extrinsics are exported relative to cam_edge_3 when valid chains exist.

Files:
- static_detection_report.txt
- static_detection_contact_sheet.png
- moving_detection_report.txt
- colmap_report.txt
- metric_scale.txt
- relay_chain_report.txt
- final_extrinsics_summary.csv

Interpretation:
This run tests whether the full bus interior calibration pipeline remains valid
when both the fixed bus cameras and the moving relay camera suffer from strong
motion blur. Missing marker IDs or no valid relay chains should be interpreted
as calibration failure under this stress condition.
