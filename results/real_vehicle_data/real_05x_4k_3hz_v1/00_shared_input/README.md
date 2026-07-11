# Shared real-data input

This directory is the single canonical input consumed by AP01, AP02 and AP03.

```text
00_shared_input/
├── raw_images/             # original canonical extracted dataset
├── aruco_observations/     # shared ArUco detections and metadata
├── DATASET_MANIFEST.json
└── README.md
```

The three methods must not own or silently modify their own copies of these
inputs. Method-specific outputs remain in their separate result directories.

The original local compatibility path under `data_local/.../datasets/` is
created as a symlink by
`run/real_vehicle_data/00_validate_and_prepare_shared_input.py`.
