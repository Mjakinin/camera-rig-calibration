# AP01 — marker-direct relay

AP01 estimates the moving-camera reconstruction scale from ArUco observations,
builds candidate relay transforms and solves the static-camera extrinsics.
The canonical `baseline_v1` is the reconstructed Legacy Main method. Ordinary
execution uses fresh SfM and scale; exact historical SfM reuse requires the
explicit, fail-closed `historical_reproduction` option.

Execution order:

1. `reconstruct_moving.py` — run/reuse the moving-camera COLMAP reconstruction
2. `estimate_scale.py` — estimate its metric scale
3. `build_candidates.py` — build marker/direct relay transform candidates
4. `solve_extrinsics.py` — choose and combine candidates into rig extrinsics
5. `report.py` — validate and normalize the method result

Supporting files:

- `pipeline.py` declares the order above and passes resolved configuration.
- `core.py` contains the AP01 numerical and COLMAP implementation shared by the
  stages.
- `_shared.py` contains only AP01 stage CLI and serialization helpers.

The authoritative output is the static-camera extrinsics emitted by
`solve_extrinsics.py` and normalized by `report.py`.
