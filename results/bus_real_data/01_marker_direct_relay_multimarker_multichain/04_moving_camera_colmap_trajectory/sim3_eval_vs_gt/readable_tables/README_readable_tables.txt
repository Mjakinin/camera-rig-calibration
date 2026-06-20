Readable Sim(3) evaluation tables
=================================

Important:
- Compare colmap_aligned_x/y/z with gt_x/y/z.
- Raw COLMAP coordinates are not directly comparable before Sim(3) alignment.
- Rotation is not evaluated here. gt_roll/pitch/yaw are only the commanded GT route angles.

Files:
01_side_by_side_xyz_errors.csv      Main readable table: aligned x next to gt x, etc.
02_x_axis_comparison.csv            X-axis only.
03_y_axis_comparison.csv            Y-axis only.
04_z_axis_comparison.csv            Z-axis only.
05_errors_sorted_worst_first.csv    Worst frames first.
06_raw_colmap_reference.csv         Raw COLMAP coordinates for reference only.
07_error_summary.csv                RMSE/mean/median/max summary.
