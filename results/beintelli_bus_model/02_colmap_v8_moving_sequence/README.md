# 02 COLMAP V8 Moving Sequence

This folder contains only the latest V8 moving-camera sequence.

Contents:
- images/
  captured moving-camera images
- sparse_txt/
  COLMAP sparse reconstruction in TXT format
- route_commanded.csv
  commanded moving-camera pose for each image frame
- camera_intrinsics_used.txt
  camera intrinsics used for COLMAP/PnP
- aruco_no_gt_detections/
  offline ArUco detections in the moving-camera image sequence

Important:
route_commanded.csv is used only for evaluation.
The no-GT calibration method does not use route_commanded.csv.
