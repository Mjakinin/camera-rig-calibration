# Bus Real-Data Results

Generated results for the bus real-data calibration benchmark.

Recommended high-level structure:

```text
00_shared_baseline/
01_marker_direct_relay_multimarker_multichain/
02_ref_marker_graph_ba/
03_targetless_colmap_aruco_scale/
90_approach_comparison_ref_aruco/
```

`00_shared_baseline/` is the canonical shared input/result baseline. It should contain raw images, camera info, route metadata, and neutral ArUco observations. Approach directories should contain approach-specific outputs only.
