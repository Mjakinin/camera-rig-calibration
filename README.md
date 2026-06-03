# Camera Rig Calibration Lab

Synthetic benchmark and evaluation pipeline for arbitrary camera-rig calibration using **ROS 2 Humble**, **Gazebo Sim / Ignition Fortress**, **OpenCV**, and **ros_gz** bridges.

The project goal is to build a reproducible simulation benchmark for camera-rig calibration methods. Gazebo provides known ground-truth camera poses. The calibration methods only receive camera images, `camera_info`, and known target geometry. The estimated camera-to-camera transform is then compared against the Gazebo ground truth.

Current method status:

- **Checkerboard**: baseline implemented and evaluated.
- **ArUco**: target generation, live detector, pose-live, rig estimator, and evaluator integration in progress.
- **ChArUco**: planned next.
- **Targetless calibration**: optional later.

---

## 1. Core Pipeline

```text
Gazebo simulation
  -> camera images + camera_info
  -> OpenCV target detection
  -> solvePnP per camera
  -> target pose in camera_1 and camera_2
  -> relative camera_1 -> camera_2 transform
  -> comparison against Gazebo ground truth
```

Input available to the calibration method:

```text
- /camera_1/image
- /camera_2/image
- /camera_1/camera_info
- /camera_2/camera_info
- known calibration target geometry
```

Ground truth is used only for evaluation:

```text
estimated camera_1 -> camera_2
vs.
true Gazebo camera_1 -> camera_2
```

Main metrics:

```text
- detection success / failure
- which camera failed
- estimated camera baseline
- baseline error in cm
- relative rotation error in degrees
- debug images for visual inspection
```

---

## 2. Current Minimal Setup

The current setup contains two simulated cameras and one calibration target.

```text
camera_1 pose: y = -0.35 m
camera_2 pose: y = +0.35 m
expected baseline: 0.70 m
expected relative rotation: 0 deg
```

Main ROS topics:

```text
/camera_1/image
/camera_2/image
/camera_1/camera_info
/camera_2/camera_info
/clock
```

Currently used resolutions:

```text
res320x240
res640x480
```

Dynamic Gazebo worlds are generated for each method and resolution, for example:

```text
src/calib_lab/worlds/dynamic/checkerboard_res320x240.sdf
src/calib_lab/worlds/dynamic/checkerboard_res640x480.sdf
src/calib_lab/worlds/dynamic/aruco_res320x240.sdf
src/calib_lab/worlds/dynamic/aruco_res640x480.sdf
```

---

## 3. Repository Structure

Important files and folders inside `project/`:

```text
project/
├── run_dynamic_sweep.sh
├── README.md
├── results/
│   ├── checkerboard/
│   ├── aruco/
│   └── charuco/
└── src/calib_lab/
    ├── config/
    │   ├── ground_truth_minimal.yaml
    │   └── aruco_target.yaml
    ├── models/
    │   ├── checkerboard_target/
    │   └── aruco_target/
    ├── worlds/
    │   ├── minimal_calib_world.sdf
    │   └── dynamic/
    └── scripts/
        ├── common/
        │   └── transform_utils.py
        ├── checkerboard/
        │   ├── checkerboard_live_detector.py
        │   ├── checkerboard_pose_live.py
        │   ├── checkerboard_rig_estimator.py
        │   └── checkerboard_rig_evaluator.py
        ├── aruco/
        │   ├── aruco_live_detector.py
        │   ├── aruco_pose_live.py
        │   ├── aruco_rig_estimator.py
        │   └── aruco_rig_evaluator.py
        ├── charuco/
        │   └── README.md
        └── tools/
            ├── aggregate_target_results.py
            ├── analyze_checkerboard_results.py
            ├── compare_resolution_sweeps.py
            ├── generate_checkerboard.py
            ├── generate_aruco_target.py
            ├── generate_dynamic_worlds.py
            └── set_gazebo_model_pose.py
```

---

## 4. Estimator vs Evaluator

There are two script types.

### Rig Estimator

Examples:

```text
checkerboard_rig_estimator.py
aruco_rig_estimator.py
```

Purpose: live sanity check. It continuously reads the current image pair, estimates the relative camera transform, and prints the result.

Use it to answer:

```text
Does detection work right now?
Does solvePnP work?
Is the estimated baseline roughly correct?
Do both cameras see the target?
```

### Rig Evaluator

Examples:

```text
checkerboard_rig_evaluator.py
aruco_rig_evaluator.py
```

Purpose: benchmark and ablation. It processes one scenario, writes CSV output, saves debug images, and exits. The dynamic sweep runner calls it repeatedly for all scenarios.

Simple difference:

```text
estimator = does it work live?
evaluator = how well does it work systematically?
```

---

## 5. Calibration Math

For each camera:

```text
known 3D target points
+ detected 2D image points
+ camera intrinsics from camera_info
  -> OpenCV solvePnP
  -> T_camera_target
```

For two cameras observing the same target:

```text
T_camera1_camera2 = T_camera1_target * inverse(T_camera2_target)
```

Then the estimated baseline and rotation are compared against Gazebo ground truth:

```text
estimated baseline = norm(translation part of T_camera1_camera2)
baseline error = abs(estimated baseline - ground truth baseline)
rotation error = abs(estimated relative rotation - ground truth rotation)
```

---

## 6. Checkerboard Pipeline

Current checkerboard target:

```text
target name: target_9x6_square0_12
inner corners: 9 x 6
square size: 0.12 m
```

Detector:

```text
cv2.findChessboardCornersSB
```

The checkerboard pipeline uses **SB-only** detection. The classic fallback is intentionally not used in the main evaluation pipeline because mixed detectors can produce inconsistent corner ordering and near-180-degree pose outliers in difficult views.

Checkerboard scripts:

```text
checkerboard_live_detector.py
    Live corner detector.

checkerboard_pose_live.py
    Per-camera checkerboard target pose using solvePnP.

checkerboard_rig_estimator.py
    Live two-camera checkerboard rig estimate.

checkerboard_rig_evaluator.py
    Scenario evaluator used by the dynamic sweep runner.
```

---

## 7. ArUco Pipeline

Current ArUco target:

```text
target name: target_aruco_6x4_marker0_15_sep0_06
dictionary: DICT_4X4_50
markers: 6 x 4
marker length: 0.15 m
marker separation: 0.06 m
target plane: about 1.20 m x 0.84 m
```

This target has a similar outer size to the checkerboard target, making the comparison more fair than comparing a large checkerboard against a single marker.

ArUco scripts:

```text
generate_aruco_target.py
    Generates the ArUco target texture, model, and config.

aruco_live_detector.py
    Detects markers in both cameras, optionally opens two GUI windows, logs marker IDs, and saves debug images.

aruco_pose_live.py
    Estimates ArUco target pose per camera using marker IDs and solvePnP.

aruco_rig_estimator.py
    Live two-camera ArUco rig estimation.

aruco_rig_evaluator.py
    Scenario evaluator for dynamic ArUco sweeps.
```

---

## 8. Dynamic Sweep System

Older sweep scripts restarted Gazebo for each scenario. The current dynamic runner starts Gazebo and the bridges once, then moves the target inside the running simulation using Gazebo's `set_pose` service.

Main command:

```bash
./run_dynamic_sweep.sh METHOD RESOLUTION [GROUP]
```

Examples:

```bash
./run_dynamic_sweep.sh checkerboard res640x480
./run_dynamic_sweep.sh checkerboard res320x240
./run_dynamic_sweep.sh checkerboard res640x480 yaw
./run_dynamic_sweep.sh aruco res640x480
./run_dynamic_sweep.sh aruco res320x240
```

If no group is given, all groups run:

```text
distance
yaw
shift
height
mixed
```

Each group is saved into its own result folder.

---

## 9. Scenario Groups

### Distance

```text
dist_1_2m
dist_1_4m
dist_1_6m
dist_1_8m
dist_2_0m
dist_2_2m
dist_2_4m
dist_2_6m
dist_2_8m
```

### Yaw

```text
yaw_0deg
yaw_10deg
yaw_20deg
yaw_30deg
yaw_35deg
yaw_40deg
yaw_45deg
yaw_50deg
```

### Shift

```text
shift_left_0_2m
shift_left_0_4m
shift_left_0_6m
shift_right_0_2m
shift_right_0_4m
shift_right_0_6m
```

### Height

```text
height_0_6m
height_0_8m
height_1_0m
height_1_2m
height_1_4m
```

### Mixed

```text
close_1_4m_yaw_10deg
far_2_4m_yaw_20deg
far_2_4m_yaw_30deg
shift_left_0_2m_yaw_20deg
shift_right_0_2m_yaw_20deg
```

---

## 10. Results Structure

Checkerboard results:

```text
results/checkerboard/target_9x6_square0_12/
├── res320x240/
│   ├── distance/
│   ├── yaw/
│   ├── shift/
│   ├── height/
│   └── mixed/
├── res640x480/
│   ├── distance/
│   ├── yaw/
│   ├── shift/
│   ├── height/
│   └── mixed/
└── comparison/
```

ArUco results:

```text
results/aruco/target_aruco_6x4_marker0_15_sep0_06/
├── res320x240/
│   ├── distance/
│   ├── yaw/
│   ├── shift/
│   ├── height/
│   └── mixed/
├── res640x480/
│   ├── distance/
│   ├── yaw/
│   ├── shift/
│   ├── height/
│   └── mixed/
└── comparison/
```

Each group folder contains:

```text
raw_results.csv
summary.csv
analysis_printout.txt
debug_images/
evaluator_logs/
```

---

## 11. Environment Setup

Use the ROS 2 Humble environment. In the devcontainer, the project root is usually:

```bash
cd /workspaces/project
```

Set the Gazebo model path:

```bash
export IGN_GAZEBO_RESOURCE_PATH="$PWD/src/calib_lab/models:${IGN_GAZEBO_RESOURCE_PATH:-}"
```

If Gazebo, the bridges, or evaluator scripts are already running in other terminals, stop them before starting a new run:

```bash
pkill -9 -f "ign gazebo" || true
pkill -9 -f "ign-gazebo" || true
pkill -9 -f "ruby.*ign" || true
pkill -9 -f "ros2 run ros_gz_image" || true
pkill -9 -f "ros2 run ros_gz_bridge" || true
pkill -9 -f "checkerboard_rig_evaluator.py" || true
pkill -9 -f "aruco_rig_evaluator.py" || true
```

---

## 12. Generate Targets

Generate Checkerboard:

```bash
cd /workspaces/project
python3 src/calib_lab/scripts/tools/generate_checkerboard.py
```

Generate ArUco:

```bash
cd /workspaces/project
python3 src/calib_lab/scripts/tools/generate_aruco_target.py
```

This creates:

```text
src/calib_lab/models/aruco_target/
src/calib_lab/config/aruco_target.yaml
```

---

## 13. Generate Dynamic Worlds

Checkerboard:

```bash
cd /workspaces/project

python3 src/calib_lab/scripts/tools/generate_dynamic_worlds.py \
  --method checkerboard \
  --resolution res320x240 \
  --target_uri model://checkerboard_target

python3 src/calib_lab/scripts/tools/generate_dynamic_worlds.py \
  --method checkerboard \
  --resolution res640x480 \
  --target_uri model://checkerboard_target
```

ArUco:

```bash
cd /workspaces/project

python3 src/calib_lab/scripts/tools/generate_dynamic_worlds.py \
  --method aruco \
  --resolution res320x240 \
  --target_uri model://aruco_target

python3 src/calib_lab/scripts/tools/generate_dynamic_worlds.py \
  --method aruco \
  --resolution res640x480 \
  --target_uri model://aruco_target
```

Check a generated world:

```bash
grep "<world name" src/calib_lab/worlds/dynamic/aruco_res640x480.sdf
grep -n "aruco_target\|UserCommands" src/calib_lab/worlds/dynamic/aruco_res640x480.sdf | head -30
```

---

## 14. Manual Debug Workflow

Use this when testing live detectors, pose scripts, or estimators manually.

### Terminal 1: Gazebo

Checkerboard:

```bash
cd /workspaces/project

export DISPLAY=:0
export QT_X11_NO_MITSHM=1
export IGN_GAZEBO_RESOURCE_PATH="$PWD/src/calib_lab/models:${IGN_GAZEBO_RESOURCE_PATH:-}"

ign gazebo src/calib_lab/worlds/dynamic/checkerboard_res640x480.sdf -r -v 4
```

ArUco:

```bash
cd /workspaces/project

export DISPLAY=:0
export QT_X11_NO_MITSHM=1
export IGN_GAZEBO_RESOURCE_PATH="$PWD/src/calib_lab/models:${IGN_GAZEBO_RESOURCE_PATH:-}"

ign gazebo src/calib_lab/worlds/dynamic/aruco_res640x480.sdf -r -v 4
```

Headless alternative:

```bash
ign gazebo -s src/calib_lab/worlds/dynamic/aruco_res640x480.sdf -r -v 2
```

### Terminal 2: Image Bridge

```bash
cd /workspaces/project
ros2 run ros_gz_image image_bridge /camera_1/image /camera_2/image
```

### Terminal 3: Camera Info + Clock Bridge

```bash
cd /workspaces/project

ros2 run ros_gz_bridge parameter_bridge \
  /camera_1/camera_info@sensor_msgs/msg/CameraInfo[ignition.msgs.CameraInfo \
  /camera_2/camera_info@sensor_msgs/msg/CameraInfo[ignition.msgs.CameraInfo \
  /clock@rosgraph_msgs/msg/Clock[ignition.msgs.Clock
```

### Terminal 4: Detector / Pose / Estimator

Run one of the scripts listed below.

---

## 15. Run Checkerboard Scripts Manually

Live detector:

```bash
cd /workspaces/project
python3 src/calib_lab/scripts/checkerboard/checkerboard_live_detector.py
```

Pose live:

```bash
cd /workspaces/project
python3 src/calib_lab/scripts/checkerboard/checkerboard_pose_live.py
```

Rig estimator:

```bash
cd /workspaces/project
python3 src/calib_lab/scripts/checkerboard/checkerboard_rig_estimator.py
```

Manual evaluator test:

```bash
cd /workspaces/project

python3 src/calib_lab/scripts/checkerboard/checkerboard_rig_evaluator.py \
  --ros-args \
  -p scenario_name:=manual_static \
  -p output_csv:=results/checkerboard/manual_evaluator_test/raw_results.csv \
  -p debug_dir:=results/checkerboard/manual_evaluator_test/debug_images \
  -p max_valid_samples:=1 \
  -p max_attempts:=1 \
  -p ready_timeout_sec:=20.0
```

---

## 16. Run ArUco Scripts Manually

Live detector:

```bash
cd /workspaces/project

python3 src/calib_lab/scripts/aruco/aruco_live_detector.py \
  --ros-args \
  -p show_gui:=true \
  -p save_debug:=true \
  -p save_every_n_frames:=30
```

Pose live:

```bash
cd /workspaces/project

python3 src/calib_lab/scripts/aruco/aruco_pose_live.py \
  --ros-args \
  -p show_gui:=true \
  -p save_debug:=true \
  -p save_every_n_frames:=30
```

Rig estimator:

```bash
cd /workspaces/project

python3 src/calib_lab/scripts/aruco/aruco_rig_estimator.py \
  --ros-args \
  -p show_gui:=true \
  -p save_debug:=true \
  -p save_every_n_successes:=10
```

Manual evaluator test:

```bash
cd /workspaces/project

python3 src/calib_lab/scripts/aruco/aruco_rig_evaluator.py \
  --ros-args \
  -p scenario_name:=manual_static \
  -p output_csv:=results/aruco/manual_evaluator_test/raw_results.csv \
  -p debug_dir:=results/aruco/manual_evaluator_test/debug_images \
  -p max_valid_samples:=1 \
  -p max_attempts:=1 \
  -p ready_timeout_sec:=20.0
```

---

## 17. Run Dynamic Sweeps

Checkerboard, all groups:

```bash
cd /workspaces/project

./run_dynamic_sweep.sh checkerboard res320x240
./run_dynamic_sweep.sh checkerboard res640x480
```

Checkerboard, one group:

```bash
./run_dynamic_sweep.sh checkerboard res640x480 yaw
./run_dynamic_sweep.sh checkerboard res640x480 distance
./run_dynamic_sweep.sh checkerboard res640x480 shift
./run_dynamic_sweep.sh checkerboard res640x480 height
./run_dynamic_sweep.sh checkerboard res640x480 mixed
```

ArUco, all groups:

```bash
cd /workspaces/project

./run_dynamic_sweep.sh aruco res320x240
./run_dynamic_sweep.sh aruco res640x480
```

ArUco, one group:

```bash
./run_dynamic_sweep.sh aruco res640x480 yaw
./run_dynamic_sweep.sh aruco res640x480 distance
./run_dynamic_sweep.sh aruco res640x480 shift
./run_dynamic_sweep.sh aruco res640x480 height
./run_dynamic_sweep.sh aruco res640x480 mixed
```

---

## 18. Aggregate Results

Checkerboard:

```bash
cd /workspaces/project

python3 src/calib_lab/scripts/tools/aggregate_target_results.py \
  --target_dir results/checkerboard/target_9x6_square0_12
```

ArUco:

```bash
cd /workspaces/project

python3 src/calib_lab/scripts/tools/aggregate_target_results.py \
  --target_dir results/aruco/target_aruco_6x4_marker0_15_sep0_06
```

Aggregation output:

```text
comparison/all_results_long.csv
comparison/resolution_comparison_wide.csv
comparison/counts_by_resolution.csv
```

---

## 19. Quick Result Printout

Checkerboard compact comparison:

```bash
cd /workspaces/project

python3 - <<'PY'
import csv
from pathlib import Path

path = Path("results/checkerboard/target_9x6_square0_12/comparison/resolution_comparison_wide.csv")
with path.open(newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

for r in rows:
    print(r["group"], r["scenario"],
          "| 320:", r.get("res320x240_pose_class", ""), r.get("res320x240_baseline_error_cm", ""),
          "| 640:", r.get("res640x480_pose_class", ""), r.get("res640x480_baseline_error_cm", ""),
          "| best:", r.get("best_resolution_by_error", ""))
PY
```

ArUco compact comparison:

```bash
cd /workspaces/project

python3 - <<'PY'
import csv
from pathlib import Path

path = Path("results/aruco/target_aruco_6x4_marker0_15_sep0_06/comparison/resolution_comparison_wide.csv")
with path.open(newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

for r in rows:
    print(r["group"], r["scenario"],
          "| 320:", r.get("res320x240_pose_class", ""), r.get("res320x240_baseline_error_cm", ""),
          "| 640:", r.get("res640x480_pose_class", ""), r.get("res640x480_baseline_error_cm", ""),
          "| best:", r.get("best_resolution_by_error", ""))
PY
```

---

## 20. Current Checkerboard Findings

Current checkerboard results showed:

```text
Useful distance range:
approximately 1.6 m to 2.0 m

Too close:
1.2 m and 1.4 m failed

Too far:
2.4 m and beyond usually failed

Yaw:
0° to 30° works
35° works only at 640x480
40° and above fails

Shift:
pure lateral shifts mostly fail

Resolution:
640x480 often improves accuracy and can rescue some borderline cases,
but it does not solve visibility / field-of-view / overlap problems.
```

Interpretation:

```text
Higher resolution improves corner localization and can improve robustness near detection limits. However, higher resolution cannot compensate for missing common field of view or severe target visibility limitations.
```

---

## 21. Troubleshooting

### Target appears black or missing

Set the model path and restart Gazebo:

```bash
export IGN_GAZEBO_RESOURCE_PATH="$PWD/src/calib_lab/models:${IGN_GAZEBO_RESOURCE_PATH:-}"
```

### set_pose service missing

Check:

```bash
ign service -l | grep set_pose
```

Expected examples:

```text
/world/dynamic_checkerboard_res640x480/set_pose
/world/dynamic_aruco_res640x480/set_pose
```

If missing, regenerate the dynamic world using `generate_dynamic_worlds.py`.

### Wrong result folder named `0`

This means the scenario CSV is wrong or old. Regenerate dynamic worlds and check:

```bash
head -n 5 src/calib_lab/worlds/dynamic/scenario_poses.csv | cat -A
```

Expected:

```text
scenario,group,x,y,z,roll,pitch,yaw$
```

### High memory usage

Use `run_dynamic_sweep.sh`. Avoid older restart-based sweep scripts for large sweeps.

---

## 22. Next Steps

Planned next steps:

```text
1. Finish and validate the ArUco evaluator.
2. Run ArUco dynamic sweeps for res320x240 and res640x480.
3. Aggregate ArUco results.
4. Compare Checkerboard and ArUco.
5. Implement ChArUco target and scripts.
6. Run the same dynamic sweeps for ChArUco.
7. Compare Checkerboard, ArUco, and ChArUco.
8. Later extend the setup from two cameras to a 3+ camera rig and a bus-interior-inspired layout.
```

Longer-term evaluation questions:

```text
- Which method is most accurate under favorable views?
- Which method is most robust under yaw?
- Which method handles partial visibility better?
- How much does resolution matter?
- How much does target size / geometry matter?
- How much common field of view is required?
- When do methods fail and why?
```

---

## 23. Presentation Summary

The current project story:

```text
We built a reproducible ROS 2 + Gazebo benchmark for camera rig calibration.
Gazebo provides ground truth.
OpenCV methods only receive camera images, camera_info, and target geometry.
The pipeline estimates camera-to-camera extrinsics and compares them against ground truth.
Checkerboard is the first completed baseline.
ArUco is being integrated with the same structure.
The dynamic sweep runner allows systematic ablation over distance, yaw, shift, height, mixed poses, and resolution.
```

The main contribution so far is a reusable evaluation framework:

```text
method-specific detector/evaluator
+ common Gazebo scenario generation
+ common dynamic runner
+ common result structure
+ common CSV aggregation
+ debug images
+ ground-truth metrics
```
