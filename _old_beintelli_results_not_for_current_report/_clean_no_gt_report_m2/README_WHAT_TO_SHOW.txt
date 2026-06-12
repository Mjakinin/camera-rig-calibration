CLEAN NO-GT REPORT CONTENTS
===========================

Use these files for the milestone/progress presentation:

01_station_visibility_summary.txt
  Shows which ArUco station can be used by which camera.
  Important: valid front anchors are F3/F4. Valid rear anchors are R1/R3.

02_colmap_no_gt_coverage.txt
  Shows COLMAP registration quality for V8.

03_moving_board_detection_counts.csv
  Shows how many moving-camera frames detected each board.

04_pair_evaluation_table.csv / .md
  Main result table.
  Translation error and rotation error compare estimated T_front_rear against static-camera GT.
  GT is evaluation-only, not used in estimation.

debug_images/
  Use selected images from station_visibility and manual_anchor_debug for slides.

pair_summaries/
  Full no-GT result summaries and final matrices.

Suggested presentation result:
  Best current translation result: F3_R3.
  Explain F3_R3 as best current station pair by 3D translation error.
