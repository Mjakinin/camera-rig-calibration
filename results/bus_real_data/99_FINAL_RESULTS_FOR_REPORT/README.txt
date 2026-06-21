FINAL RESULTS FOR REPORT / MEETING
==================================

Use these canonical result files:

AP01:
- AP01/AP01_FINAL_RESULT.txt
- AP01/AP01_FINAL_RESULT.csv
Meaning:
Ref14-origin static camera poses. Ref14->anchor camera is estimated from a real ArUco/PnP observation, not GT.

AP02:
- AP02/AP02_FINAL_RESULT.txt
- AP02/AP02_FINAL_RESULT.csv
Meaning:
GT-aligned full marker/camera map. Cameras and markers 0-14 are evaluated. Marker14 is held out from alignment.

AP03:
- AP03/AP03_FINAL_RESULT.txt
- AP03/AP03_FINAL_RESULT.csv
Meaning:
Targetless COLMAP + Ref14 Sim(3) scale/frame registration. Static cameras are evaluated relative to Ref14.

Main method comparison:
Compare static camera errors for cam_edge_0, cam_edge_1, cam_edge_3, cam_edge_5.
