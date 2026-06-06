# ChArUco Calibration Scripts

This folder contains the ChArUco-based calibration pipeline for the camera rig calibration project.

The goal is to use **ChArUco corner interpolation** with **ArUco marker IDs** and checkerboard-like corners, while keeping the same evaluation pipeline as the existing Checkerboard and ArUco methods.

ChArUco combines:

- ArUco marker IDs for robust target identification
- checkerboard-like interpolated corners for more precise pose estimation
- partial visibility support, because not all markers/corners need to be visible

---

## Scripts

Expected ChArUco scripts:

```txt
charuco_common.py
charuco_live_detector.py
charuco_pose_live.py
charuco_rig_estimator.py
charuco_rig_evaluator.py
```

Additional tool:

```txt
generate_charuco_target.py
```

---

## Expected Project Structure

```txt
project/
  src/
    calib_lab/
      config/
        ground_truth_minimal.yaml
        charuco_target.yaml

      models/
        charuco_target/
          model.config
          model.sdf
          materials/
            textures/
              charuco_board.png

      worlds/
        dynamic/
          charuco_res640x480.sdf
          scenario_poses.csv

      scripts/
        charuco/
          charuco_common.py
          charuco_live_detector.py
          charuco_pose_live.py
          charuco_rig_estimator.py
          charuco_rig_evaluator.py

        common/
          transform_utils.py

        tools/
          generate_charuco_target.py
          generate_dynamic_worlds.py
          set_gazebo_model_pose.py
          run_dynamic_sweep.sh
```

---

## 1. Generate the ChArUco Target

The ChArUco target model and texture must be generated before running Gazebo.

From the project root:

```bash
cd ~/cam/project

python3 src/calib_lab/scripts/tools/generate_charuco_target.py
```

This should create:

```txt
src/calib_lab/models/charuco_target/
src/calib_lab/models/charuco_target/materials/textures/charuco_board.png
src/calib_lab/config/charuco_target.yaml
```

Check the generated files:

```bash
ls src/calib_lab/models/charuco_target/materials/textures/
cat src/calib_lab/config/charuco_target.yaml
cat src/calib_lab/models/charuco_target/model.sdf
```

Expected texture:

```txt
charuco_board.png
```

Check for broken model names or wrong texture paths:

```bash
grep -R "chcharuco\|chchcharuco\|checkerboard_10x7\|charuco_10x7" -n src/calib_lab/models/charuco_target
```

Expected result: no output.

Expected `model.sdf` texture path:

```xml
<albedo_map>model://charuco_target/materials/textures/charuco_board.png</albedo_map>
```

---

## 2. Generate the Dynamic World

Generate the Gazebo world for ChArUco:

```bash
python3 src/calib_lab/scripts/tools/generate_dynamic_worlds.py \
  --method charuco \
  --resolution res640x480
```

This creates:

```txt
src/calib_lab/worlds/dynamic/charuco_res640x480.sdf
src/calib_lab/worlds/dynamic/scenario_poses.csv
```

`scenario_poses.csv` contains the target poses used for the dynamic sweeps, for example:

```txt
distance
yaw
shift
height
mixed
```

It is normal that `scenario_poses.csv` gets overwritten when `generate_dynamic_worlds.py` is run again.

---

## 3. Start Gazebo

Kill old Gazebo / bridge processes first:

```bash
pkill -9 -f "ign gazebo" || true
pkill -9 -f "ign-gazebo" || true
pkill -9 -f "ruby.*ign" || true
pkill -9 -f "ros2 run ros_gz_image image_bridge" || true
pkill -9 -f "ros2 run ros_gz_bridge parameter_bridge" || true
```

Set the model path and start Gazebo:

```bash
cd ~/cam/project

export IGN_GAZEBO_RESOURCE_PATH="$PWD/src/calib_lab/models:${IGN_GAZEBO_RESOURCE_PATH:-}"

ign gazebo -r src/calib_lab/worlds/dynamic/charuco_res640x480.sdf
```

The ChArUco target should be visible in the world.

A ChArUco target may visually look like ArUco markers placed inside a checkerboard-like grid. That is expected.

---

## 4. Start ROS2 Bridges

Open a new terminal:

```bash
cd ~/cam/project

ros2 run ros_gz_image image_bridge /camera_1/image /camera_2/image
```

Open another terminal:

```bash
cd ~/cam/project

ros2 run ros_gz_bridge parameter_bridge \
  /camera_1/camera_info@sensor_msgs/msg/CameraInfo[ignition.msgs.CameraInfo \
  /camera_2/camera_info@sensor_msgs/msg/CameraInfo[ignition.msgs.CameraInfo \
  /clock@rosgraph_msgs/msg/Clock[ignition.msgs.Clock
```

---

## 5. Test 1: Live Detection

Run:

```bash
cd ~/cam/project

python3 src/calib_lab/scripts/charuco/charuco_live_detector.py \
  --ros-args \
  -p show_gui:=false \
  -p save_debug:=true \
  -p save_every_n_frames:=1
```

Expected output:

```txt
camera_1 | markers > 0 | charuco_corners > 0
camera_2 | markers > 0 | charuco_corners > 0
```

Example successful output:

```txt
camera_1 | method=gray | markers=17 | charuco=24
camera_2 | method=gray | markers=17 | charuco=24
```

Debug images are saved to:

```txt
results/charuco/live_detector/debug_images/
```

This script only checks whether ArUco markers and interpolated ChArUco corners are detected.

---

## 6. Test 2: Pose Live

Run:

```bash
cd ~/cam/project

python3 src/calib_lab/scripts/charuco/charuco_pose_live.py \
  --ros-args \
  -p show_gui:=false \
  -p save_debug:=true \
  -p save_every_n_frames:=1
```

Expected output:

```txt
POSE FOUND | camera_1 | markers=... | charuco_corners=... | t=[...] m
POSE FOUND | camera_2 | markers=... | charuco_corners=... | t=[...] m
```

Example:

```txt
POSE FOUND | camera_1 | markers=17 | charuco_corners=24 | t=[-0.758, 0.295, 1.732] m
POSE FOUND | camera_2 | markers=17 | charuco_corners=24 | t=[-0.093, 0.291, 1.674] m
```

This script estimates the pose of the ChArUco target relative to each camera.

---

## 7. Test 3: Rig Estimator

Run:

```bash
cd ~/cam/project

python3 src/calib_lab/scripts/charuco/charuco_rig_estimator.py \
  --ros-args \
  -p show_gui:=false
```

Expected output format:

```txt
================ CHARUCO RIG ESTIMATE ================
valid_pair=...
camera_1: method=charuco/gray, markers=..., charuco_corners=..., t_cam_target=[...] m
camera_2: method=charuco/gray, markers=..., charuco_corners=..., t_cam_target=[...] m

Estimated relative transform from common ChArUco target:
T_camera_1_camera_2 translation = [...] m
full-transform baseline norm  = ... m
expected baseline norm        = 0.7000 m
full-transform baseline error = ... m
translation-only baseline     = ... m
translation-only error        = ... m
estimated relative rotation angle = ... deg
rotation angle error              = ... deg
======================================================
```

This script performs the full two-camera ChArUco rig estimation.

It computes:

- target pose in camera 1
- target pose in camera 2
- relative camera transform
- baseline error
- rotation error

The default target setup should ideally produce a small baseline and rotation error.

Suggested target validation goal:

```txt
baseline error < 1 cm
rotation error < 1 deg
```

Do not start large sweeps until the default/static rig estimation is validated.

---

## 8. Run Dynamic Sweep

After live detection, pose live and rig estimation work, run the automatic sweep.

Start with one group first:

```bash
cd ~/cam/project

./src/calib_lab/scripts/tools/run_dynamic_sweep.sh charuco res640x480 distance
```

If the sweep script is located directly in the project root, use:

```bash
./run_dynamic_sweep.sh charuco res640x480 distance
```

Available groups:

```txt
distance
yaw
shift
height
mixed
all
```

Examples:

```bash
./src/calib_lab/scripts/tools/run_dynamic_sweep.sh charuco res640x480 yaw
./src/calib_lab/scripts/tools/run_dynamic_sweep.sh charuco res640x480 shift
./src/calib_lab/scripts/tools/run_dynamic_sweep.sh charuco res640x480 height
./src/calib_lab/scripts/tools/run_dynamic_sweep.sh charuco res640x480 mixed
```

Run all groups:

```bash
./src/calib_lab/scripts/tools/run_dynamic_sweep.sh charuco res640x480 all
```

---

## 9. Sweep Result Structure

Sweep results are written to:

```txt
results/charuco/target_charuco_current/res640x480/
```

Example:

```txt
results/
  charuco/
    target_charuco_current/
      res640x480/
        distance/
          raw_results.csv
          summary.csv
          debug_images/
          evaluator_logs/

        yaw/
          raw_results.csv
          summary.csv
          debug_images/
          evaluator_logs/

        shift/
          raw_results.csv
          summary.csv
          debug_images/
          evaluator_logs/

        height/
          raw_results.csv
          summary.csv
          debug_images/
          evaluator_logs/

        mixed/
          raw_results.csv
          summary.csv
          debug_images/
          evaluator_logs/
```

Check results:

```bash
cat results/charuco/target_charuco_current/res640x480/distance/summary.csv
```

---

## 10. Important Notes

### Debug Images

Debug images are not created just by starting Gazebo.

They are created only when one of the ChArUco scripts runs:

```txt
charuco_live_detector.py
charuco_pose_live.py
charuco_rig_estimator.py
charuco_rig_evaluator.py
```

### `scenario_poses.csv`

`scenario_poses.csv` contains the target poses for the sweep.

It is generated by:

```bash
generate_dynamic_worlds.py
```

and used by:

```bash
run_dynamic_sweep.sh
```

It is normal that it gets overwritten when the dynamic world is regenerated.

### Current Debug Status

The ChArUco live detector works in the static/default case:

```txt
markers=17
charuco_corners=24
```

Pose live also works and returns stable target poses.

The rig estimator may still need geometric validation if the full-transform baseline error is larger than expected. This can indicate a remaining issue in:

- ChArUco board geometry
- target texture scaling
- target coordinate frame
- pose convention
- physical board size in `model.sdf`
- values in `charuco_target.yaml`

---

## 11. Recommended Test Order

Use this order when debugging:

```txt
1. generate_charuco_target.py
2. generate_dynamic_worlds.py
3. start Gazebo
4. start ROS2 bridges
5. charuco_live_detector.py
6. charuco_pose_live.py
7. charuco_rig_estimator.py
8. run_dynamic_sweep.sh charuco res640x480 distance
9. run remaining sweep groups
```

Do not start with the sweep directly. First validate detection and pose estimation.

---

## 12. Troubleshooting

### File not found: `charuco_live_detector.py`

Make sure you are in the project root:

```bash
cd ~/cam/project
```

Then run:

```bash
ls src/calib_lab/scripts/charuco/
```

Expected files:

```txt
charuco_common.py
charuco_live_detector.py
charuco_pose_live.py
charuco_rig_estimator.py
charuco_rig_evaluator.py
```

---

### Gazebo cannot find ChArUco texture

Error example:

```txt
Unable to find file with URI [model://chcharuco_target/...]
```

Check for broken paths:

```bash
grep -R "chcharuco\|chchcharuco\|checkerboard_10x7\|charuco_10x7" -n src/calib_lab/models/charuco_target
```

Expected model name:

```txt
charuco_target
```

Expected texture path:

```txt
model://charuco_target/materials/textures/charuco_board.png
```

---

### Markers are detected but no ChArUco corners

If output shows:

```txt
markers > 0
charuco_corners = 0
```

then the board geometry or YAML config may not match the generated texture.

Check:

```bash
cat src/calib_lab/config/charuco_target.yaml
cat src/calib_lab/models/charuco_target/model.sdf
```

The physical board size and YAML values must match.

---

### OpenCV ArUco module missing

If you see:

```txt
cv2.aruco is not available
```

then OpenCV was installed without the contrib modules. You need an OpenCV build that includes `cv2.aruco`.

---

## 13. Useful Validation Commands

Check the ChArUco model:

```bash
grep -R "charuco_target\|charuco_board.png\|size" -n src/calib_lab/models/charuco_target
```

Check for broken names:

```bash
grep -R "chcharuco\|chchcharuco\|checkerboard_10x7\|charuco_10x7" -n src/calib_lab/models/charuco_target
```

Check the config:

```bash
cat src/calib_lab/config/charuco_target.yaml
```

Check the dynamic world:

```bash
grep -R "charuco_target\|width\|height" -n src/calib_lab/worlds/dynamic/charuco_res640x480.sdf
```

---

## 14. Goal of ChArUco Experiments

The ChArUco method should be compared against Checkerboard and ArUco under the same sweep conditions.

The main comparison metrics are:

```txt
detection success
number of detected markers
number of interpolated ChArUco corners
estimated baseline
baseline error
rotation error
failure cases
debug images
```

Expected advantage of ChArUco:

```txt
more robust than Checkerboard because of marker IDs
more precise than ArUco because of interpolated checkerboard-like corners
better behavior under partial visibility and difficult target poses
```

---

## 15. Interpretation

ChArUco should help with limitations seen in pure Checkerboard detection.

Checkerboard can be accurate, but it has no marker IDs and may suffer from orientation ambiguity in difficult views.

ArUco is robust because of marker IDs, but can be less precise because it relies on marker corners.

ChArUco combines both:

```txt
ArUco IDs
+
checkerboard-like interpolated corners
```

The final goal is to evaluate whether ChArUco improves robustness and accuracy compared to Checkerboard and ArUco in the same Gazebo/ROS2 camera rig calibration pipeline.
