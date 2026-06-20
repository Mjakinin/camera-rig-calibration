# Shared raw image datasets

This folder stores approach-independent raw image datasets.

Current canonical dataset:

```text
results/bus_real_data/00_raw_images/bus_real_data_ref_marker_v1/
```

Contract:

- Raw images and camera_info only.
- No approach-specific detections.
- No approach-specific estimates.
- AP02/AP03 should read from this folder.
- Each approach writes its own detections/results into its own result folder.
- AP01 remains untouched as the frozen baseline; its real moving sequence is used as the canonical moving raw sequence for this dataset.
