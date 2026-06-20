# AP03: Targetless COLMAP + ArUco Ref14 Scale Registration

AP03 reconstructs the scene with COLMAP/SfM and then registers the reconstruction to metric scale using triangulated Ref14 ArUco marker corners.

## Method summary

- Targetless COLMAP/SfM is used as the frontend.
- ArUco marker 14 is used after reconstruction for metric Sim(3) registration.
- The marker side length is used for post-COLMAP scale/registration, not as a PnP constraint during reconstruction.
- Ground truth is used only for final evaluation.

## Main runner

Full reconstruction:

```bash
bash run/bus_real_data/approach3_targetless_colmap_aruco_scale/run_approach3_full_pipeline.sh
```

Fast smoke test using existing outputs:

```bash
bash run/bus_real_data/approach3_targetless_colmap_aruco_scale/run_approach3_full_pipeline.sh --reuse-existing
```

## Final result

```text
results/bus_real_data/03_targetless_colmap_aruco_scale/06_triangulated_ref_aruco_registration/
results/bus_real_data/90_approach_comparison_ref_aruco/03_targetless_colmap_aruco_scale/
```

Validated final camera-level result:

```text
registered images: 208
sparse 3D points: 9401
mean translation error: 7.501966 cm
mean rotation error:    0.367828 deg
```
