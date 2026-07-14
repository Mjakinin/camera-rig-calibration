REAL-VEHICLE INTRINSIC CALIBRATION RESULTS
==========================================

This directory stores one readable TXT report per intrinsic-calibration video.
The matching CameraInfo JSON is archived beside the report when the canonical
wrapper is used.

Canonical command
-----------------

python3 run/real_vehicle_data/02_calibrate_intrinsics_and_archive.py \
  --video /path/to/intrinsics_video.mov \
  --out results/real_vehicle_data/<dataset>/00_shared_input/calibration/<run_name> \
  --result-name <camera_and_mode_label>

Examples
--------

0.5x 4K:

python3 run/real_vehicle_data/02_calibrate_intrinsics_and_archive.py \
  --video /path/to/0.5_Intrinsics.mov \
  --out results/real_vehicle_data/real_05x_4k_3hz_v1/00_shared_input/calibration/intrinsics_05x \
  --result-name iphone_05x_4k

Future 1x 4K:

python3 run/real_vehicle_data/02_calibrate_intrinsics_and_archive.py \
  --video /path/to/1x_Intrinsics.mov \
  --out results/real_vehicle_data/real_1x_4k_3hz_v1/00_shared_input/calibration/intrinsics_1x \
  --result-name iphone_1x_4k

Output naming
-------------

<result-name>_<width>x<height>_INTRINSICS_REPORT.txt
<result-name>_<width>x<height>_moving_calib_camera.json

The archive report includes:
- source video and resolution
- source frame rate
- checkerboard configuration
- number of detections and selected views
- removed outlier views
- model comparison and reprojection errors
- camera matrix K
- scalar fx, fy, cx and cy
- distortion vector D

Validity
--------

An intrinsic calibration is valid only for the same physical camera module,
resolution, sensor crop and capture mode.  A 0.5x result must not be reused as
a 1x calibration.
