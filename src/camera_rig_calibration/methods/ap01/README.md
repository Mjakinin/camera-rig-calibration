# AP01 — marker-direct relay

AP01 estimates the moving-camera reconstruction scale from ArUco observations,
builds candidate Direct and Relay transforms and solves the static-camera
extrinsics. The canonical `baseline_v1` uses a fixed Direct target and Relay
support for the remaining cameras. Every run computes fresh SfM and metric
scale from the selected input dataset. `recommended_wizard_v1` exposes the
configurable multi-marker consensus strategy.

## Method semantics

AP01 first reconstructs the moving calibration camera with COLMAP. The
reconstruction provides relative moving-camera geometry but has no metric
scale. Repeated observations of known-size ArUco markers provide metric
inter-frame motion estimates. These are compared with the corresponding COLMAP
motion and robustly aggregated to obtain a scale for the moving-camera
trajectory.

For two registered moving frames `i` and `j` that observe the same marker, the
implementation forms a metric translation magnitude from their marker-based
PnP poses and the corresponding translation magnitude in the COLMAP
reconstruction. One scale candidate is

```math
s_{ij}
=
\frac{\left\lVert \mathbf{t}^{\mathrm{metric}}_{ij}\right\rVert_2}
     {\left\lVert \mathbf{t}^{\mathrm{COLMAP}}_{ij}\right\rVert_2}
```

with units of metres per COLMAP unit. Candidate construction and robust
filtering are defined by the selected AP01 method contract. After filtering,
the final moving-trajectory scale is the median of the retained scale
candidates. Ground Truth is not involved in this scale estimate.

Static-camera poses are expressed relative to a selected root camera. The
notation `T_A_B` below denotes a transform that maps coordinates from frame `B`
into frame `A`. Two kinds of transform candidates can provide the root-to-target
relation:

- **Direct:** the root camera and a target static camera observe the same marker.
  The implemented relation is

  ```math
  \mathbf{T}_{\mathrm{root}\leftarrow\mathrm{target}}
  =
  \mathbf{T}_{\mathrm{root}\leftarrow\mathrm{marker}}
  \mathbf{T}_{\mathrm{target}\leftarrow\mathrm{marker}}^{-1}
  ```

- **Relay:** a marker observation links the root camera to one registered moving
  frame `i` and another marker observation links the target camera to another
  registered moving frame `j`. The scaled COLMAP motion between those frames
  bridges the two static-camera observations:

  ```math
  \mathbf{T}_{\mathrm{root}\leftarrow\mathrm{target}}
  =
  \mathbf{T}_{\mathrm{root}\leftarrow\mathrm{moving},i}
  \mathbf{T}_{\mathrm{moving},i\leftarrow\mathrm{moving},j}
  \mathbf{T}_{\mathrm{target}\leftarrow\mathrm{moving},j}^{-1}
  ```

  The markers at the two ends of the relay do not have to be the same marker.

Candidate construction uses only the selected observations, the reconstructed
moving-camera poses and the recovered metric scale. Candidate filtering,
aggregation and Direct/Relay priority are defined by the selected AP01 method
contract. Ground Truth is not used to choose or combine calibration candidates.

Relay estimates compose several locally estimated transformations. Errors in
marker pose estimation, moving-camera SfM or metric scale can therefore
propagate through the relay into the final static-camera pose.

## Execution order

1. `reconstruct_moving.py` — run/reuse the moving-camera COLMAP reconstruction
2. `estimate_scale.py` — estimate its metric scale
3. `build_candidates.py` — build marker-supported Direct and Relay transform candidates
4. `solve_extrinsics.py` — choose and combine candidates into rig extrinsics
5. `report.py` — validate and normalize the method result

## Supporting files

- `pipeline.py` declares the order above and passes resolved configuration.
- `core.py` exposes the AP01 numerical and COLMAP implementation shared by the
  stages.
- `core_scale.py` contains the moving-trajectory metric-scale estimation.
- `core_candidates.py` contains Direct/Relay candidate construction and robust
  aggregation helpers.
- `_shared.py` contains AP01 stage CLI and serialization helpers.

The authoritative output is the static-camera extrinsics emitted by
`solve_extrinsics.py` and normalized by `report.py`.
