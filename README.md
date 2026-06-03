# Camera Rig Calibration Lab

Synthetic benchmark and simulation lab for **arbitrary camera-rig calibration** using **ROS 2 Humble**, **Gazebo Sim / Ignition Fortress**, **OpenCV**, and `ros_gz` bridges.

The project investigates when different target-based calibration methods work, where they fail, and which experimental factors influence camera-rig calibration accuracy. The current repository is organized around three experiment worlds:

1. **Minimal World**: controlled two-camera benchmark for Checkerboard, ArUco and ChArUco calibration pipelines.
2. **Bus Corridor Relay**: placeholder for the simplified moving-camera relay calibration setup.
3. **BeIntelli Bus Model**: full imported 3D bus model with static front/rear cameras and distributed ArUco landmarks.

The repository root contains this overview. All executable project content is inside `project/`.

---

## 1. Project Goal

The goal is to build a reproducible simulation pipeline for camera-rig calibration research.

Gazebo provides:

- synthetic camera images,
- `camera_info` topics,
- known ground-truth camera poses,
- known target poses and object geometry.

The calibration algorithms receive only:

- camera images,
- camera intrinsics from `camera_info`,
- known target geometry.

The estimated relative camera transform is then compared against the Gazebo ground truth.

Core evaluation questions:

- Which calibration method works best under which conditions?
- How do distance, yaw, lateral shift, height, field of view and resolution affect calibration?
- Where do the methods fail and why?
- How can calibration be transferred to a bus-like camera layout with limited camera overlap?

---

## 2. Repository Layout

```text
camera-rig-calibration/
├── README.md
└── project/
    ├── run/
    │   ├── minimal_world/
    │   │   └── run_dynamic_sweep.sh
    │   ├── bus_corridor_relay/
    │   └── beintelli_bus_model/
    │       └── run_aruco_visibility_detector.sh
    │
    ├── results/
    │   ├── minimal_world/
    │   ├── bus_corridor_relay/
    │   └── beintelli_bus_model/
    │
    └── src/
        └── calib_lab/
            ├── package.xml
            ├── CMakeLists.txt
            ├── common/
            ├── minimal_world/
            ├── bus_corridor_relay/
            └── beintelli_bus_model/
```

Important convention:

```text
src/        = ROS workspace source folder
calib_lab/  = ROS 2 package
```

Therefore `package.xml` and `CMakeLists.txt` must stay directly inside `project/src/calib_lab/`.

---

## 3. Experiment Worlds

### 3.1 Minimal World

Path:

```text
project/src/calib_lab/minimal_world/
```

Purpose:

- controlled synthetic two-camera benchmark,
- target-based calibration method comparison,
- dynamic sweeps over distance, yaw, shift, height and mixed scenarios,
- repeatable ablation study.

Current method status:

```text
Checkerboard: implemented and evaluated
ArUco: implemented for target generation, detection, pose and rig evaluation
ChArUco: structure prepared / planned extension
Targetless: optional future extension
```

Main runner:

```bash
cd project
./run/minimal_world/run_dynamic_sweep.sh checkerboard res640x480 yaw gui
```

General syntax:

```bash
./run/minimal_world/run_dynamic_sweep.sh METHOD RESOLUTION [GROUP] [headless|gui]
```

Arguments:

```text
METHOD:      checkerboard | aruco | charuco
RESOLUTION:  res320x240 | res640x480
GROUP:       distance | yaw | shift | height | mixed | all
MODE:        headless | gui
```

Examples:

```bash
./run/minimal_world/run_dynamic_sweep.sh checkerboard res640x480 yaw gui
./run/minimal_world/run_dynamic_sweep.sh checkerboard res640x480 all headless
./run/minimal_world/run_dynamic_sweep.sh aruco res640x480 distance headless
```

Results are written to:

```text
project/results/minimal_world/<method>/<target>/<resolution>/<group>/
```

Each group folder usually contains:

```text
raw_results.csv
summary.csv
analysis_printout.txt
debug_images/
evaluator_logs/
```

---

### 3.2 Bus Corridor Relay

Path:

```text
project/src/calib_lab/bus_corridor_relay/
```

Purpose:

- future simplified bus-corridor world,
- two static cameras with limited or no direct overlap,
- moving virtual camera traveling through the corridor,
- moving camera observes intermediate targets,
- static cameras are connected through a relay / pose-graph style calibration structure.

This is currently a placeholder experiment area. It is intentionally separate from the full BeIntelli bus mesh so the relay idea can be developed in a clean and simple environment first.

---

### 3.3 BeIntelli Bus Model

Path:

```text
project/src/calib_lab/beintelli_bus_model/
```

Purpose:

- full imported BeIntelli 3D bus model,
- static front/rear camera setup,
- distributed individual ArUco landmarks,
- visibility analysis for realistic bus-like camera placement,
- later extension toward full bus calibration experiments.

Main runner:

```bash
cd project
./run/beintelli_bus_model/run_aruco_visibility_detector.sh gui results/beintelli_bus_model/aruco_visibility/current
```

Headless variant:

```bash
./run/beintelli_bus_model/run_aruco_visibility_detector.sh headless results/beintelli_bus_model/aruco_visibility/current
```

Main world:

```text
project/src/calib_lab/beintelli_bus_model/worlds/bus_individual_marker_visibility_test.sdf
```

Main detector:

```text
project/src/calib_lab/beintelli_bus_model/scripts/bus_aruco_visibility_detector.py
```

Results are written to:

```text
project/results/beintelli_bus_model/aruco_visibility/
```

---

## 4. Calibration Pipeline

For target-based calibration, the general pipeline is:

```text
Gazebo world
  -> camera image topics
  -> camera_info topics
  -> OpenCV target detection
  -> solvePnP per camera
  -> target pose in each camera frame
  -> relative camera-to-camera transform
  -> comparison against Gazebo ground truth
```

For two cameras observing the same target:

```text
known target geometry
+ detected 2D target points
+ camera intrinsics from camera_info
  -> OpenCV solvePnP
  -> T_camera_target for each camera

T_camera1_camera2 = T_camera1_target * inverse(T_camera2_target)
```

Main metrics:

```text
detection status
valid sample count
estimated baseline
baseline error in cm
estimated relative rotation
rotation error in degrees
visible marker IDs / target points
debug images
```

---

## 5. Environment Setup

The project was developed for:

```text
Ubuntu 22.04
ROS 2 Humble
Gazebo Sim / Ignition Fortress
OpenCV with aruco module
ros_gz image and parameter bridges
Python 3
Git LFS
```

Before running experiments:

```bash
source /opt/ros/humble/setup.bash
cd /path/to/camera-rig-calibration/project
export PROJECT_DIR="$(pwd)"
```

For WSLg / GUI Gazebo runs, `DISPLAY` is usually set automatically. If not:

```bash
export DISPLAY=:0
export QT_X11_NO_MITSHM=1
```

---

## 6. Git LFS Requirement

The BeIntelli mesh files are large and are stored through **Git LFS**.

Before cloning or pulling the repository, install Git LFS:

```bash
sudo apt update
sudo apt install git-lfs -y
git lfs install
```

After cloning:

```bash
git lfs pull
```

Check that LFS files are present:

```bash
git lfs ls-files
```

Expected large LFS-managed files include:

```text
beintelli_erklarbus.glb
beintelli_erklarbus.dae
scene.bin
beintelli_erklarbus.obj
beintelli_erklarbus.stl
```

If Gazebo cannot load the bus mesh, first verify that these files were downloaded correctly through LFS.

---

## 7. Sanity Tests

Run these commands from `project/`.

### 7.1 Validate Python scripts

```bash
find src/calib_lab -name "*.py" -print0 | xargs -0 -n1 python3 -m py_compile
```

### 7.2 Validate SDF/XML syntax

```bash
python3 - <<'PY'
from pathlib import Path
import xml.etree.ElementTree as ET

ok = True
for path in Path("src/calib_lab").rglob("*.sdf"):
    try:
        ET.parse(path)
        print("[OK] XML:", path)
    except Exception as e:
        ok = False
        print("[ERROR] XML:", path, e)

raise SystemExit(0 if ok else 1)
PY
```

### 7.3 Minimal world smoke test

```bash
source /opt/ros/humble/setup.bash
./run/minimal_world/run_dynamic_sweep.sh checkerboard res640x480 yaw gui
```

Expected behavior:

- Gazebo starts,
- the target is moved through the yaw scenarios,
- result folders are created under `results/minimal_world/`,
- `raw_results.csv` and `summary.csv` are written.

### 7.4 BeIntelli bus model smoke test

```bash
source /opt/ros/humble/setup.bash
./run/beintelli_bus_model/run_aruco_visibility_detector.sh gui results/beintelli_bus_model/aruco_visibility/current
```

Expected behavior:

- Gazebo starts with the BeIntelli bus model,
- front and rear static camera topics are bridged,
- ArUco marker visibility is evaluated,
- CSV and summary text files are written.

---

## 8. Notes on Results

`results/` contains generated experiment outputs. These files are useful for inspection, but the source of truth is the reproducible pipeline in `src/` and `run/`.

For clean experiments, remove old result folders before rerunning large sweeps:

```bash
rm -rf results/minimal_world/checkerboard/target_9x6_square0_12/res640x480/yaw
```

The runners also clean the requested result group before writing new outputs.

---

## 9. Development Notes

Recommended workflow:

1. Keep experiment-specific files inside their world folder.
2. Keep shared utilities inside `src/calib_lab/common/`.
3. Keep large mesh files under Git LFS.
4. Avoid committing temporary logs, caches and large debug result folders.
5. Add new experiment worlds as separate folders instead of mixing scripts and models globally.

Experiment folders:

```text
minimal_world       = controlled calibration benchmark
bus_corridor_relay  = future moving-camera relay setup
beintelli_bus_model = full imported bus model setup
common              = shared utilities only
```

---

## 10. License / Attribution

The BeIntelli bus model is stored with attribution information in:

```text
project/src/calib_lab/beintelli_bus_model/models/beintelli_bus/ATTRIBUTION.txt
```

Check this file before redistributing the mesh assets.
