# AP03: Targetless COLMAP with ArUco Metric Registration

AP03 reconstructs camera poses with targetless COLMAP/SfM and then establishes
metric scale and registration using ArUco geometry.

Full execution:

```bash
bash run/bus_real_data/approach3_targetless_colmap_aruco_scale/run_approach3_full_pipeline.sh
```

Reuse existing reconstruction:

```bash
bash run/bus_real_data/approach3_targetless_colmap_aruco_scale/run_approach3_full_pipeline.sh --reuse-existing
```

Canonical result root:

```text
results/bus_real_data/03_targetless_colmap_aruco_scale/
```

Future AP03 optimization must preserve primary and secondary metrics, coverage
states, ablation contracts, deterministic reporting and directory structure.
