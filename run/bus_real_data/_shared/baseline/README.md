# Shared Baseline

Method-independent shared preprocessing for the bus real-data experiment.

Main scripts:

```text
run_shared_preprocessing.sh
02_detect_shared_aruco_observations.py
03_export_ap1_observations_from_shared.py
```

Canonical output root:

```text
results/bus_real_data/00_shared_baseline/bus_real_data_ref_marker_v1/
```

Expected structure:

```text
raw_images/
  static/
  moving/
  camera_info/
aruco_observations/
  shared_static_aruco_observations.csv
  shared_moving_aruco_observations.csv
  shared_all_aruco_observations.csv
metadata/
  route_commanded.csv
```

Important: `raw_images/` must contain real image files or valid links. If legacy `results/bus_real_data/00_raw_images/` is removed, verify that shared raw image paths still exist.
