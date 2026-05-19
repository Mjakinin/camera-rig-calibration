# Camera Rig Calibration Lab

This project builds a minimal synthetic benchmark pipeline for arbitrary camera rig calibration using:

- ROS 2 Humble
- Gazebo Sim / Ignition Fortress
- ros_gz_image
- ros_gz_bridge
- OpenCV
- rosbag2

The current focus is a minimal two-camera setup with a checkerboard calibration target. The goal is to generate controlled synthetic camera images in Gazebo, process them with OpenCV-based calibration methods, and compare the estimated camera extrinsics against known ground truth from the simulation.

---

## 1. Project Goal

The main idea is:

```text
Gazebo defines the true camera poses.
These true poses are the ground truth.

The calibration algorithm only sees:
- camera images
- camera_info / intrinsics
- calibration target geometry

The algorithm estimates:
- target pose relative to camera_1
- target pose relative to camera_2
- camera_1 to camera_2 transformation

Then we compare:
estimated camera_1 -> camera_2
against
ground-truth camera_1 -> camera_2
```

This is the core of the camera rig calibration benchmark.

---

## 2. Current Minimal Setup

The current simulation contains:

```text
Gazebo world:
- ground plane
- light source
- camera_1
- camera_2
- checkerboard calibration target
```

Current ROS topics:

```text
/camera_1/image
/camera_2/image
/camera_1/camera_info
/camera_2/camera_info
/clock
```

Current camera resolution:

```text
width  = 320
height = 240
fps    ≈ 10 Hz
```

Current checkerboard:

```text
10 x 7 squares
9 x 6 inner corners
square_size = 0.12 m
```

OpenCV pattern size:

```python
pattern_size = (9, 6)
```

---

## 3. Repository Structure

The relevant files are inside the `project/` folder:

```text
project/
├── src/
│   └── calib_lab/
│       ├── worlds/
│       │   └── minimal_calib_world.sdf
│       ├── models/
│       │   └── checkerboard_target/
│       │       ├── model.config
│       │       ├── model.sdf
│       │       └── materials/
│       │           └── textures/
│       │               └── checkerboard_10x7.png
│       └── scripts/
│           ├── generate_checkerboard.py
│           ├── checkerboard_live_detector.py
│           └── checkerboard_pose_live.py
└── bags/                  # optional, generated rosbag2 recordings
```

---

## 4. Prerequisites

This project assumes that ROS 2 Humble, Gazebo Sim / Ignition Fortress, and the required ROS-Gazebo bridge packages are already installed.

Required packages include:

```text
ros-humble-ros-gz
ros-humble-ros-gz-image
ros-humble-ros-gz-bridge
ros-humble-rqt-image-view
ros-humble-cv-bridge
python3-opencv
python3-numpy
```

If `ros2`, `ign`, or one of the bridge commands is not found, check the local ROS/Gazebo installation first.

---

## 5. Important Gazebo Model Path

The checkerboard target is loaded as a local Gazebo model:

```xml
model://checkerboard_target
```

Therefore, Gazebo must be able to find the local model folder:

```text
project/src/calib_lab/models
```

If the checkerboard target appears black or is missing, run this once from inside the `project/` folder:

```bash
export IGN_GAZEBO_RESOURCE_PATH="$(pwd)/src/calib_lab/models:${IGN_GAZEBO_RESOURCE_PATH}"
```

This can also be added permanently to the local shell configuration, for example `.bashrc`.

---

## 6. Recommended Terminal Layout

Use four terminals while debugging:

```text
Terminal 1: Gazebo simulation
Terminal 2: Image bridge
Terminal 3: camera_info + clock bridge
Terminal 4: Tests, visualization, OpenCV scripts, rosbag2
```

This is the manual debug workflow. Later, this can be replaced by launch files or start scripts.

---

## 7. Terminal 1: Start Gazebo

From the repository root:

```bash
cd project
```

Start the minimal calibration world:

```bash
ign gazebo src/calib_lab/worlds/minimal_calib_world.sdf -r -v 4
```

Expected:

- Gazebo opens.
- Two cameras are visible.
- The checkerboard target is visible.
- The simulation is running because of `-r`.

If the checkerboard is black or missing, check the Gazebo model path:

```bash
export IGN_GAZEBO_RESOURCE_PATH="$(pwd)/src/calib_lab/models:${IGN_GAZEBO_RESOURCE_PATH}"
```

Then restart Gazebo.

---

## 8. Terminal 2: Start Image Bridge

From the repository root:

```bash
cd project
```

Start the image bridge:

```bash
ros2 run ros_gz_image image_bridge /camera_1/image /camera_2/image
```

Keep this terminal open.

This provides the ROS image topics:

```text
/camera_1/image
/camera_2/image
```

---

## 9. Terminal 3: Start camera_info and clock Bridge

From the repository root:

```bash
cd project
```

Start the bridge for camera intrinsics and simulation time:

```bash
ros2 run ros_gz_bridge parameter_bridge \
  /camera_1/camera_info@sensor_msgs/msg/CameraInfo[ignition.msgs.CameraInfo \
  /camera_2/camera_info@sensor_msgs/msg/CameraInfo[ignition.msgs.CameraInfo \
  /clock@rosgraph_msgs/msg/Clock[ignition.msgs.Clock
```

Keep this terminal open.

This provides:

```text
/camera_1/camera_info
/camera_2/camera_info
/clock
```

---

## 10. Terminal 4: Check the Pipeline

From the repository root:

```bash
cd project
```

List relevant topics:

```bash
ros2 topic list | grep -E "camera|clock"
```

Expected topics:

```text
/camera_1/camera_info
/camera_1/image
/camera_1/image/compressed
/camera_1/image/compressedDepth
/camera_1/image/theora
/camera_2/camera_info
/camera_2/image
/camera_2/image/compressed
/camera_2/image/compressedDepth
/camera_2/image/theora
/clock
```

If `/camera_1/image` or `/camera_2/image` is missing, check Terminal 2.

If `/camera_1/camera_info` or `/camera_2/camera_info` is missing, check Terminal 3.

---

## 11. Check Image Frequency

Check camera 1:

```bash
ros2 topic hz /camera_1/image
```

Expected:

```text
average rate: about 9.5 to 10.0 Hz
```

Check camera 2:

```bash
ros2 topic hz /camera_2/image
```

Expected:

```text
average rate: about 9.5 to 10.0 Hz
```

Stop with:

```text
Ctrl+C
```

---

## 12. Check Image Resolution

Check image width:

```bash
timeout 5 ros2 topic echo /camera_1/image --field width --once
```

Expected:

```text
320
---
```

Check image height:

```bash
timeout 5 ros2 topic echo /camera_1/image --field height --once
```

Expected:

```text
240
---
```

---

## 13. Check camera_info

Check camera 1:

```bash
timeout 5 ros2 topic echo /camera_1/camera_info --once
```

Expected output contains:

```text
height: 240
width: 320
distortion_model: plumb_bob
k:
- ...
```

Short check:

```bash
timeout 5 ros2 topic echo /camera_1/camera_info --field width --once
```

Expected:

```text
320
---
```

Check camera 2:

```bash
timeout 5 ros2 topic echo /camera_2/camera_info --field width --once
```

Expected:

```text
320
---
```

---

## 14. Check /clock

```bash
timeout 5 ros2 topic echo /clock --once
```

Expected:

```text
clock:
  sec: ...
  nanosec: ...
---
```

`/clock` is the simulation time from Gazebo. It is useful for synchronized multi-camera data and reproducible rosbag2 replay.

---

## 15. Visualize Images with rqt_image_view

Start:

```bash
ros2 run rqt_image_view rqt_image_view
```

Then select:

```text
/camera_1/image
```

or:

```text
/camera_2/image
```

Expected:

- Both cameras show the checkerboard target.
- The checkerboard has a white margin.
- The target is not completely black.

Important: Moving the free Gazebo GUI camera does not change `/camera_1/image`. `rqt_image_view` shows the simulated sensor camera image, not the Gazebo editor viewport.

---

## 16. Record a rosbag2 Dataset

Make sure Gazebo and both bridges are running.

From inside `project/`:

```bash
mkdir -p bags
```

Record:

```bash
ros2 bag record \
  /camera_1/image \
  /camera_2/image \
  /camera_1/camera_info \
  /camera_2/camera_info \
  /clock \
  -o bags/checkerboard_static_01
```

Let it run for 10 to 30 seconds.

Stop with:

```text
Ctrl+C
```

Check the bag:

```bash
ros2 bag info bags/checkerboard_static_01
```

Expected topics:

```text
/camera_1/image
/camera_2/image
/camera_1/camera_info
/camera_2/camera_info
/clock
```

---

## 17. Replay a rosbag2 Dataset

Stop Gazebo and all bridges first for a clean replay test.

From inside `project/`:

```bash
ros2 bag play bags/checkerboard_static_01 --clock
```

In another terminal:

```bash
ros2 topic list | grep camera
ros2 topic hz /camera_1/image
```

Expected:

- Image topics appear.
- `/camera_1/image` publishes at approximately the recorded rate.

---

## 18. Checkerboard Texture Generation

The checkerboard image is generated by:

```bash
python3 src/calib_lab/scripts/generate_checkerboard.py
```

Current board design:

```text
10 x 7 squares
9 x 6 inner corners
white margin around the board
```

The white margin is important. Without it, OpenCV may fail to detect the checkerboard even if it looks correct to humans.

Texture path:

```text
src/calib_lab/models/checkerboard_target/materials/textures/checkerboard_10x7.png
```

If Gazebo shows a black target, check:

```bash
ls -lh src/calib_lab/models/checkerboard_target/materials/textures/
```

and:

```bash
echo $IGN_GAZEBO_RESOURCE_PATH
```

Gazebo may cache models and textures. If the texture changes, fully restart Gazebo.

---

## 19. Checkerboard Detection Script

Script:

```text
src/calib_lab/scripts/checkerboard_live_detector.py
```

Purpose:

```text
Checks whether OpenCV can detect the checkerboard in a live ROS image topic.
```

Run for camera 1:

```bash
python3 src/calib_lab/scripts/checkerboard_live_detector.py \
  --ros-args \
  -p image_topic:=/camera_1/image \
  -p corners_x:=9 \
  -p corners_y:=6
```

Expected:

```text
FOUND checkerboard | frame=... | corners=54 | success=...
```

Run for camera 2:

```bash
python3 src/calib_lab/scripts/checkerboard_live_detector.py \
  --ros-args \
  -p image_topic:=/camera_2/image \
  -p corners_x:=9 \
  -p corners_y:=6
```

Expected:

```text
FOUND checkerboard | frame=... | corners=54 | success=...
```

Important:

```text
54 corners = 9 x 6 inner corners
```

If detection fails:

- check that `/camera_X/image` is running
- check `rqt_image_view`
- check that the checkerboard has a white margin
- check that `corners_x=9`, `corners_y=6`
- move target closer
- increase target size
- increase resolution later if needed

---

## 20. Checkerboard Pose Script

Script:

```text
src/calib_lab/scripts/checkerboard_pose_live.py
```

Purpose:

```text
Estimates the pose of the checkerboard relative to one camera using OpenCV solvePnP.
```

Inputs:

```text
/camera_X/image
/camera_X/camera_info
```

Run for camera 1:

```bash
python3 src/calib_lab/scripts/checkerboard_pose_live.py \
  --ros-args \
  -p image_topic:=/camera_1/image \
  -p camera_info_topic:=/camera_1/camera_info \
  -p corners_x:=9 \
  -p corners_y:=6 \
  -p square_size:=0.12
```

Expected:

```text
POSE FOUND | method=SB | frame=... | t=[..., ..., ...] m | dist=... m
```

Example:

```text
POSE FOUND | method=SB | frame=1 | t=[0.130, 0.298, 1.785] m | dist=1.814 m
```

Run for camera 2:

```bash
python3 src/calib_lab/scripts/checkerboard_pose_live.py \
  --ros-args \
  -p image_topic:=/camera_2/image \
  -p camera_info_topic:=/camera_2/camera_info \
  -p corners_x:=9 \
  -p corners_y:=6 \
  -p square_size:=0.12
```

Important:

If the script only prints:

```text
Image topic: ...
CameraInfo topic: ...
Pattern: ...
```

and then nothing else, then it is probably not receiving `camera_info`.

Check:

```bash
ros2 topic list | grep camera_info
```

and:

```bash
timeout 5 ros2 topic echo /camera_1/camera_info --field width --once
```

If this fails, restart Terminal 3.

---

## 21. Meaning of solvePnP Output

OpenCV `solvePnP` estimates the pose of the checkerboard in the camera coordinate system.

OpenCV camera coordinates are typically:

```text
x = right in the image
y = down in the image
z = forward from the camera
```

Example:

```text
t=[0.130, 0.298, 1.785] m
```

roughly means:

```text
The checkerboard is about 1.785 m in front of the camera,
0.130 m sideways,
and 0.298 m vertically in image/camera coordinates.
```

The distance:

```text
dist=1.814 m
```

is the Euclidean distance from the camera to the checkerboard origin.

---

## 22. Ground Truth Concept

In Gazebo, we know the actual poses because we define them in the SDF world file.

Example current setup:

```text
camera_1 pose: x=0, y=-0.35, z=1.0
camera_2 pose: x=0, y=+0.35, z=1.0
target pose:   x≈1.8, y=0.0,   z=1.0
```

Therefore, the true camera baseline is approximately:

```text
camera_1 to camera_2 = 0.70 m lateral offset
```

The algorithm should estimate this from images.

The final evaluation will compare:

```text
estimated T_camera1_camera2
against
ground-truth T_camera1_camera2
```

Metrics:

```text
translation error [m or cm]
rotation error [deg]
reprojection error [px]
success rate [%]
runtime [s]
```

---

## 23. Current Calibration Pipeline

Current checkerboard pipeline:

```text
Gazebo renders checkerboard target
↓
ros_gz_image publishes camera images to ROS
↓
ros_gz_bridge publishes camera_info and clock
↓
OpenCV detects checkerboard corners
↓
OpenCV solvePnP estimates target pose per camera
↓
Next step: combine camera_1 and camera_2 target poses
↓
Estimate camera_1 -> camera_2 extrinsic transform
↓
Compare against Gazebo ground truth
```

---

## 24. Next Development Step

Next script to implement:

```text
checkerboard_rig_estimator.py
```

It should subscribe to:

```text
/camera_1/image
/camera_1/camera_info
/camera_2/image
/camera_2/camera_info
```

Then:

```text
1. detect checkerboard in camera_1
2. detect checkerboard in camera_2
3. run solvePnP for both cameras
4. compute T_cam1_target
5. compute T_cam2_target
6. compute estimated T_cam1_cam2
7. compare to known Gazebo ground truth
8. print translation and rotation errors
```

---

## 25. Common Problems and Fixes

### Problem: Checkerboard appears black or missing in Gazebo

Check the texture:

```bash
ls -lh src/calib_lab/models/checkerboard_target/materials/textures/
```

Check the Gazebo model path:

```bash
echo $IGN_GAZEBO_RESOURCE_PATH
```

If needed, set it from inside the `project/` folder:

```bash
export IGN_GAZEBO_RESOURCE_PATH="$(pwd)/src/calib_lab/models:${IGN_GAZEBO_RESOURCE_PATH}"
```

Restart Gazebo fully.

---

### Problem: ROS image topics are missing

Check that the image bridge is running:

```bash
ros2 run ros_gz_image image_bridge /camera_1/image /camera_2/image
```

Then check:

```bash
ros2 topic list | grep image
```

---

### Problem: camera_info topics are missing

Check that the camera_info bridge is running:

```bash
ros2 run ros_gz_bridge parameter_bridge \
  /camera_1/camera_info@sensor_msgs/msg/CameraInfo[ignition.msgs.CameraInfo \
  /camera_2/camera_info@sensor_msgs/msg/CameraInfo[ignition.msgs.CameraInfo \
  /clock@rosgraph_msgs/msg/Clock[ignition.msgs.Clock
```

Then check:

```bash
ros2 topic list | grep camera_info
```

---

### Problem: Checkerboard visible but OpenCV does not detect it

Likely causes:

```text
no white margin around board
board too small
board too far away
wrong pattern size
low image resolution
texture not sharp enough
```

Fixes:

```text
add white margin
move target closer
increase target size
use pattern_size=(9, 6)
increase resolution later if needed
```

---

### Problem: Pose script prints only startup logs

Example:

```text
Image topic: /camera_1/image
CameraInfo topic: /camera_1/camera_info
Pattern: (9, 6), square_size=0.12 m
```

but no pose output.

Likely cause:

```text
/camera_1/camera_info is not running
```

Check:

```bash
ros2 topic list | grep camera_info
```

Restart Terminal 3.

---

## 26. Current Status

Working:

```text
Gazebo minimal world
two cameras
checkerboard target
ROS image topics
camera_info topics
/clock topic
rqt_image_view
rosbag2 recording
rosbag2 replay
OpenCV checkerboard detection
OpenCV solvePnP pose estimation for camera_1
```

Next:

```text
Run solvePnP for camera_2
Implement camera_1 -> camera_2 estimation
Add ground truth YAML
Compute translation and rotation error
Then add ArUco
Then add ChArUco
Then compare methods
```

---

## 27. Quick Start Summary

From the repository root, enter the project folder:

```bash
cd project
```

If the checkerboard model is not found by Gazebo, set the model path:

```bash
export IGN_GAZEBO_RESOURCE_PATH="$(pwd)/src/calib_lab/models:${IGN_GAZEBO_RESOURCE_PATH}"
```

Start Gazebo:

```bash
ign gazebo src/calib_lab/worlds/minimal_calib_world.sdf -r -v 4
```

Start image bridge:

```bash
ros2 run ros_gz_image image_bridge /camera_1/image /camera_2/image
```

Start camera_info and clock bridge:

```bash
ros2 run ros_gz_bridge parameter_bridge \
  /camera_1/camera_info@sensor_msgs/msg/CameraInfo[ignition.msgs.CameraInfo \
  /camera_2/camera_info@sensor_msgs/msg/CameraInfo[ignition.msgs.CameraInfo \
  /clock@rosgraph_msgs/msg/Clock[ignition.msgs.Clock
```

Check image:

```bash
ros2 topic hz /camera_1/image
```

Check camera_info:

```bash
timeout 5 ros2 topic echo /camera_1/camera_info --field width --once
```

Run checkerboard detector:

```bash
python3 src/calib_lab/scripts/checkerboard_live_detector.py \
  --ros-args \
  -p image_topic:=/camera_1/image \
  -p corners_x:=9 \
  -p corners_y:=6
```

Run pose estimation:

```bash
python3 src/calib_lab/scripts/checkerboard_pose_live.py \
  --ros-args \
  -p image_topic:=/camera_1/image \
  -p camera_info_topic:=/camera_1/camera_info \
  -p corners_x:=9 \
  -p corners_y:=6 \
  -p square_size:=0.12
```
