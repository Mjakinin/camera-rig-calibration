# 03 COLMAP Moving Pose vs Simulation GT

Purpose:
Evaluate how well COLMAP reconstructs the moving_calib_camera trajectory.

Files:
- summary_colmap_moving_pose_vs_gt.txt
- colmap_moving_pose_vs_gt.csv
- chain_used_frames.md
- chain_used_frames.csv

Main result:
- 346 registered/common frames
- 10.37 cm RMSE position error
- 8.88 cm mean position error

Important:
This is evaluation only.
The final no-GT front-rear chain does not use the simulation trajectory.
