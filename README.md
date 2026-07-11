# Arbitrary Camera Rig Calibration

Final fixed methods:

- AP01 baseline: direct multimarker with moving-COLMAP relay where required
- AP02 baseline: distortion-aware reference-marker graph bundle adjustment
- AP03: grouped calibrated targetless COLMAP with ArUco marker-size scale

Canonical shared inputs:

```text
results/bus_real_data/00_shared_baseline/bus_real_data_ref_marker_v1/
├── raw_images/
├── aruco_observations/
└── metadata/

results/real_vehicle_data/real_05x_4k_3hz_v1/00_shared_input/
├── raw_images/
├── aruco_observations/
├── calibration/
└── metadata/
```

AP01, AP02 and AP03 read the same immutable input dataset. Their outputs are
stored separately and are overwritten by the corresponding rerun pipeline.

Simulation development baseline: Route 2 with the extended ArUco layout.
