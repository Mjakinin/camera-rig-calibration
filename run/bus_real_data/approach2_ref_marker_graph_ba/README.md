# AP02: Reference-Marker Graph Bundle Adjustment

AP02 estimates a reference-marker pose graph and optimizes it with bundle adjustment.

## Method summary

- Reference marker: ArUco marker 14.
- Single-reference-marker PnP is used as a baseline.
- Static-only graph initialization shows limited connectivity.
- With-moving graph initialization connects all static cameras and marker poses.
- Bundle adjustment optimizes reprojection error.
- Ground truth is used only for evaluation.

## Main runner

```bash
bash run/bus_real_data/approach2_ref_marker_graph_ba/run_approach2_full_pipeline.sh --skip-shared-baseline
```

## Final result

```text
results/bus_real_data/02_ref_marker_graph_ba/08_final_results/
results/bus_real_data/90_approach_comparison_ref_aruco/02_ref_marker_graph_ba/
```

Validated final camera-level result:

```text
mean translation error: 10.099 cm
mean rotation error:     1.607 deg
```
