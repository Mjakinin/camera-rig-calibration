# Shared Baseline Results

Canonical shared baseline result folder.

Expected dataset:

```text
bus_real_data_ref_marker_v1/
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

These files are shared across AP01, AP02, and AP03. They are method-independent and should not contain approach-specific estimation outputs.
