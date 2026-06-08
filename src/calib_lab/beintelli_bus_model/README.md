# BeIntelli Bus Model

Full imported BeIntelli 3D bus model setup for bus-like camera-rig calibration experiments.

This experiment is separate from the minimal benchmark world. It is intended to move the project toward the real-world use case: multiple static cameras installed in a bus interior with limited overlap and distributed calibration landmarks.

---

## Purpose

The current BeIntelli setup is used to answer:

- Can the imported 3D bus model be loaded reliably in Gazebo?
- Where should static front/rear cameras be placed?
- Which ArUco landmarks are visible from each static camera?
- How much overlap exists between the static cameras?
- Which landmarks are occluded by seats, windows or other bus geometry?

Later this setup can be extended toward full rig calibration and moving-camera relay calibration.

---

## Folder Structure

```text
beintelli_bus_model/
├── config/
│   └── bus_static_cameras.yaml
├── models/
│   ├── beintelli_bus/
│   └── aruco_individual/
├── scripts/
│   ├── bus_aruco_visibility_detector.py
│   └── tools/
└── worlds/
    ├── bus_visual_test.sdf
    ├── bus_static_camera_test.sdf
    └── bus_individual_marker_visibility_test.sdf
```

---

## Main World

```text
worlds/bus_individual_marker_visibility_test.sdf
```

This is the current main test world. It contains:

- the BeIntelli bus model,
- static front and rear cameras,
- individual ArUco marker models,
- idealized stable lighting for reproducible visibility tests.

---

## Support Worlds

```text
worlds/bus_visual_test.sdf
```

Used to verify that the bus mesh loads, has the correct scale, and has the expected orientation.

```text
worlds/bus_static_camera_test.sdf
```

Used to tune front/rear static camera placement before adding distributed ArUco landmarks.

---

## Main Runner

Run from the `project/` directory:

```bash
source /opt/ros/humble/setup.bash
./run/beintelli_bus_model/run_aruco_visibility_detector.sh gui results/beintelli_bus_model/aruco_visibility/current
```

Headless variant:

```bash
./run/beintelli_bus_model/run_aruco_visibility_detector.sh headless results/beintelli_bus_model/aruco_visibility/current
```

The runner:

1. starts Gazebo,
2. sets the correct Gazebo resource paths,
3. bridges the front and rear camera image topics,
4. runs the ArUco visibility detector,
5. writes CSV and text summary output.

---

## Important Topics

```text
/front_static_camera/image
/front_static_camera/camera_info
/rear_static_camera/image
/rear_static_camera/camera_info
```

---

## Outputs

Default output location:

```text
project/results/beintelli_bus_model/aruco_visibility/current/
```

Typical files:

```text
bus_aruco_visibility.csv
bus_aruco_visibility_summary.txt
debug_images/
```

The summary reports:

```text
front_static_camera detected IDs
rear_static_camera detected IDs
overlap IDs between front and rear
```

---

## Tools

```text
scripts/tools/convert_bus_gltf_for_gazebo.py
```

Converts the downloaded glTF/GLB bus asset with Blender into Gazebo-friendly mesh files such as OBJ/STL.

```text
scripts/tools/generate_individual_aruco_markers.py
```

Generates individual ArUco marker models, for example `aruco_marker_00` to `aruco_marker_09`.

```text
scripts/tools/set_bus_camera_pose_live.py
```

Moves a static camera live in a running Gazebo simulation and helps tune camera placement.

```text
scripts/tools/set_bus_pose_in_world.py
```

Patches the bus pose/orientation/scale in a world file.

```text
scripts/tools/set_camera_intrinsics_in_world.py
```

Patches camera resolution, horizontal field of view and related camera parameters in the SDF world.

---

## Mesh Assets and Git LFS

Large bus mesh files are stored through Git LFS. After cloning the repository:

```bash
git lfs pull
```

Expected LFS-managed files include:

```text
beintelli_erklarbus.glb
beintelli_erklarbus.dae
scene.bin
beintelli_erklarbus.obj
beintelli_erklarbus.stl
```

The current `model.sdf` loads the OBJ mesh:

```text
model://beintelli_bus/meshes/obj/beintelli_erklarbus.obj
```

If Gazebo cannot load the bus, first check whether the OBJ file was downloaded through Git LFS.

---

## Attribution

The bus model attribution file is located at:

```text
models/beintelli_bus/ATTRIBUTION.txt
```

Check this file before redistributing the mesh assets.
