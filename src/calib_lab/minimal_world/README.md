# Minimal World

Controlled synthetic benchmark world for target-based camera-rig calibration.

This experiment is the clean baseline environment of the project. It uses two simulated cameras, one calibration target and known Gazebo ground truth. It is intended for repeatable method comparison and ablation studies.

---

## Purpose

The minimal world answers questions such as:

- How accurately can a target-based method estimate the relative transform between two cameras?
- Which method is more robust under distance, yaw, shift and height changes?
- At which viewpoint or resolution does detection fail?
- How do Checkerboard, ArUco and later ChArUco compare under the same scenario set?

---

## Folder Structure

```text
minimal_world/
├── config/
│   ├── ground_truth_minimal.yaml
│   └── aruco_target.yaml
├── models/
│   ├── checkerboard_target/
│   └── aruco_target/
├── scripts/
│   ├── checkerboard/
│   ├── aruco/
│   ├── charuco/
│   └── tools/
└── worlds/
    ├── minimal_calib_world.sdf
    └── dynamic/
```

---

## Main Runner

Run from the `project/` directory:

```bash
source /opt/ros/humble/setup.bash
./run/minimal_world/run_dynamic_sweep.sh checkerboard res640x480 yaw gui
```

General form:

```bash
./run/minimal_world/run_dynamic_sweep.sh METHOD RESOLUTION [GROUP] [headless|gui]
```

Supported values:

```text
METHOD:      checkerboard | aruco | charuco
RESOLUTION:  res320x240 | res640x480
GROUP:       distance | yaw | shift | height | mixed | all
MODE:        headless | gui
```

Examples:

```bash
./run/minimal_world/run_dynamic_sweep.sh checkerboard res640x480 all headless
./run/minimal_world/run_dynamic_sweep.sh checkerboard res640x480 yaw gui
./run/minimal_world/run_dynamic_sweep.sh aruco res640x480 distance headless
```

---

## Dynamic Scenario Groups

The dynamic sweep runner starts Gazebo once and moves the target with Gazebo's `set_pose` service.

Scenario definition file:

```text
worlds/dynamic/scenario_poses.csv
```

Groups:

```text
distance: target distance sweep
yaw:      target yaw-angle sweep
shift:    lateral target shift sweep
height:   target height sweep
mixed:    combined difficult cases
all:      all groups sequentially
```

---

## Outputs

Results are written to:

```text
project/results/minimal_world/<method>/<target>/<resolution>/<group>/
```

Typical files:

```text
raw_results.csv
summary.csv
analysis_printout.txt
debug_images/
evaluator_logs/
```

---

## Core Scripts

### Checkerboard

```text
scripts/checkerboard/checkerboard_live_detector.py
scripts/checkerboard/checkerboard_pose_live.py
scripts/checkerboard/checkerboard_rig_estimator.py
scripts/checkerboard/checkerboard_rig_evaluator.py
```

### ArUco

```text
scripts/aruco/aruco_live_detector.py
scripts/aruco/aruco_pose_live.py
scripts/aruco/aruco_rig_estimator.py
scripts/aruco/aruco_rig_evaluator.py
```

### Tools

```text
scripts/tools/generate_checkerboard.py
scripts/tools/generate_aruco_target.py
scripts/tools/generate_dynamic_worlds.py
scripts/tools/set_gazebo_model_pose.py
scripts/tools/analyze_checkerboard_results.py
scripts/tools/compare_resolution_sweeps.py
scripts/tools/aggregate_target_results.py
```

---

## Notes

- `ground_truth_minimal.yaml` contains the expected camera rig transform used for evaluation.
- `scenario_poses.csv` stores yaw in radians because Gazebo uses radians internally.
- Checkerboard evaluation currently uses OpenCV's SB detector for stable corner ordering.
- Result folders can be deleted and regenerated with the runner.
