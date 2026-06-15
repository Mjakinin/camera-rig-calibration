# Full Pipeline Ablation: Static Camera Motion Blur length=15

Branch: ablation-study
Commit: d4d523f

Degradation:
- Type: static-camera-only degradation
- Mode: motion_blur
- Length: 15
- Degraded data: static bus camera images
- Clean data: moving camera sequence and COLMAP trajectory

Purpose:
This experiment isolates the sensitivity of the static-camera anchor detections.
The moving-camera relay capture remains unchanged.

Files:
- static_detection_report.txt: per-camera baseline-vs-degraded marker comparison
- static_detection_contact_sheet.png: visual overview of detected/missed IDs
- static_debug_images/: per-camera annotated images
- relay_chain_report.txt: relay-chain evaluation using degraded static detections
- final_extrinsics_summary.csv: final extrinsics relative to cam_edge_3

Interpretation:
This run tests whether the final bus camera calibration remains valid when
the fixed bus cameras observe the A4 ArUco markers with motion blur.
