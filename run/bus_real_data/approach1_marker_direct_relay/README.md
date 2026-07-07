# AP01: Marker Direct Relay, Multimarker and Multichain

AP01 estimates static-camera extrinsics from ArUco observations, direct
static-camera overlap and moving-camera relay chains.

Reference gauge: `cam_edge_3`.

- `cam_edge_1`: direct static-camera multimarker overlap.
- `cam_edge_0` and `cam_edge_5`: moving-camera relay chains.
- COLMAP provides moving-camera motion.
- ArUco geometry establishes metric scale.
- Ground truth is evaluation-only.

Entry point:

```bash
bash run/bus_real_data/approach1_marker_direct_relay/run_approach1_full_pipeline.sh
```

Canonical result root:

```text
results/bus_real_data/01_marker_direct_relay_multimarker_multichain/
```

A clean AP01 run requires COLMAP and must not reuse stale relay outputs.
