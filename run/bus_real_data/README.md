# Bus Real-Data Calibration Pipelines

This directory contains executable scripts for the bus real-data camera-rig calibration experiments.

```text
run/bus_real_data/
  _shared/
    baseline/                 # shared preprocessing and neutral ArUco observation export
    common/                   # reusable Python utilities
    tools/
      capture/                # dataset acquisition / preview / capture helpers
      live_sim/               # Gazebo/live marker pose helpers
      migration/              # one-off migration helpers
  approach1_marker_direct_relay/
  approach2_ref_marker_graph_ba/
  approach3_targetless_colmap_aruco_scale/
  approach_comparison_ref_aruco/
```

## Approaches

| ID | Directory | Method |
|---|---|---|
| AP01 | `approach1_marker_direct_relay/` | Direct marker relay + COLMAP moving-camera trajectory + multichain aggregation |
| AP02 | `approach2_ref_marker_graph_ba/` | Reference-marker pose graph and bundle adjustment |
| AP03 | `approach3_targetless_colmap_aruco_scale/` | Targetless COLMAP/SfM plus ArUco Ref14 metric scale registration |

## Shared baseline

Shared/raw data and method-independent ArUco observations are generated under:

```text
results/bus_real_data/00_shared_baseline/bus_real_data_ref_marker_v1/
```

Approach-specific outputs remain in their own result directories.

## Quick smoke tests

```bash
cd /workspaces/project
PYTHONPATH=run/bus_real_data find run/bus_real_data -name "*.py" -print0 | xargs -0 python3 -m py_compile
bash run/bus_real_data/approach3_targetless_colmap_aruco_scale/run_approach3_full_pipeline.sh --reuse-existing
```
