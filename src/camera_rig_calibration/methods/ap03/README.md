# AP03 — Joint SfM with marker-based metric scale

AP03 jointly registers the selected static-camera images and the moving-camera frames in one COLMAP Structure-from-Motion reconstruction. Image registration uses natural image features; ArUco detections are not used to establish the SfM reconstruction. Calibrated camera intrinsics remain fixed.

The resulting COLMAP reconstruction has an arbitrary global metric scale. AP03 therefore uses known-size ArUco markers only after image registration to recover metric scale. Ground-Truth camera poses, Ground-Truth marker poses, simulator marker maps and known marker-map coordinates are not calibration inputs.

The default `baseline_v1` contract keeps the established scientific baseline behavior. Multi-marker scale is the primary AP03 result; the Single-marker result is retained as an additional diagnostic. More configurable feature, observation-selection and reconstruction policies remain explicit advanced settings rather than silently changing the baseline contract.

## Execution order

1. `prepare_colmap.py` — prepare the grouped COLMAP input and calibrated camera definitions.
2. `reconstruct_stage.py` / `reconstruct.py` — run the shared static + moving COLMAP reconstruction.
3. `inspect_stage.py` / `inspect.py` — inspect candidate sparse models and select the reconstruction used downstream.
4. `estimate_scale.py` / `scale_core.py` — estimate Single- and Multi-marker metric scale from reconstructed marker geometry.
5. `report.py` — normalize and publish the AP03 outputs, with Multi as the primary result.

Single and Multi share the same COLMAP reconstruction; they differ only in the marker set used for metric-scale estimation.

## Metric scale recovery

For the scale stage, AP03 detects configured ArUco markers in registered images and triangulates each marker corner from multiple registered views. A marker contributes scale candidates only when all four corners are successfully reconstructed.

For each complete marker, six geometric segments are evaluated:

- four edges with known physical length equal to the marker side length;
- two diagonals with known physical length `sqrt(2)` times the marker side length.

For segment `k`, with known metric length `l_k` and reconstructed corner positions `X_(k,a)` and `X_(k,b)` in COLMAP units, the scale candidate is

```text
s_k = l_k / ||X_(k,a) - X_(k,b)||_2
```

with units of metres per COLMAP unit.

### Robust aggregation

The implementation first computes the median and median absolute deviation (MAD) of the finite positive scale candidates. A candidate is retained when its absolute deviation from the raw median is no larger than

```text
max(3 * MAD, 0.10 * median)
```

If fewer than four candidates survive this filter, the implementation falls back to all finite positive candidates rather than returning a scale from fewer than four observations.

The recovered global metric scale is the median of the retained candidates:

```text
s_hat = median(retained s_k)
```

The recovered scale is applied to COLMAP camera translations only. Camera orientations are unchanged, and the reconstruction remains in the native COLMAP gauge apart from the metric translation scale.

## Scale-dispersion diagnostic

AP03 reports the dispersion of the retained scale candidates relative to the recovered median scale:

```text
scale_RStd[%] = 100 * std(retained s_k, ddof=0) / s_hat
```

NumPy's population standard deviation (`ddof=0`) is used. A low value means that the retained marker measurements support a common metric scale; a large value indicates inconsistent reconstructed scale.

This quantity is a consistency diagnostic, not an independent camera-pose accuracy measurement. In particular, a low scale dispersion does not by itself prove that the recovered camera geometry is correct.

The software may use configurable warning or quality-policy thresholds around scale dispersion. These are operational diagnostics and are not universal calibration-accuracy limits established by the experiments.

## Baseline marker semantics

Under the reviewed baseline configuration, Multi uses markers `0`–`14` as its scale set and is the primary AP03 output. Single uses one configured marker (marker `14` in the simulation baseline) and remains a diagnostic result. Marker sets, minimum detection area, observation-selection policy and related scale settings can be changed explicitly through the configuration system.

The baseline scale stage re-detects markers in the registered images, requires a minimum marker area of `100 px^2`, and derives scale from the reconstructed marker segments described above. Exact requested and resolved settings for an execution are recorded by the normal configuration/provenance pipeline.

See [`../../../docs/evaluation.md`](../../../docs/evaluation.md) for the evaluation-level interpretation of AP03 scale consistency together with the other reported metrics.
