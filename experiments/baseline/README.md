# Baseline Run

Branch: ablation-study
Commit: a0a05ba

Commands:

```bash
python3 run/bus_real_data/12_estimate_colmap_scale_from_aruco.py
python3 run/bus_real_data/13_eval_moving_relay_chains.py
python3 run/bus_real_data/14_export_final_extrinsics_cam3_reference.py
```

Key baseline values:
- COLMAP ArUco metric scale: 0.77794587944
- COLMAP registered poses: 129
- Raw scale pairs: 981
- Kept scale pairs after MAD trim: 850

Best COLMAP relay results:
- cam_edge_3 -> cam_edge_0: 20.91 cm, 2.24 deg
- cam_edge_3 -> cam_edge_5: 9.72 cm, 4.08 deg
