# Evaluation metrics

This document defines the evaluation quantities used by `rigcal` and the scientific comparison. It is intentionally separate from the calibration methods: simulation Ground Truth is used only after calibration, while real-data evaluation relies on internal consistency because independently measured static-camera poses are not available.

## Simulation: pairwise static-camera pose error

For every pair of recovered static cameras `(i, j)`, the evaluator forms the relative estimated and Ground-Truth transforms in the same pair direction. With relative translation vectors `t_est` and `t_gt`, the absolute translation error is

```math
e_t^{\mathrm{cm}}
=
100\,\left\lVert
\mathbf{t}_{\mathrm{est}}-\mathbf{t}_{\mathrm{GT}}
\right\rVert_2
```

where the transforms are represented in metres and the factor `100` converts the result to centimetres.

The implementation writes this value as `translation_error_cm`. It also stores the Ground-Truth pair baseline

```math
b_{\mathrm{GT}}
=
\left\lVert \mathbf{t}_{\mathrm{GT}} \right\rVert_2
```

as `gt_baseline_m`, together with the corresponding rotation error `rotation_error_deg`.

### Relative translation error used for the nominal comparison

For the nominal comparison, each pairwise translation error is normalized by that pair's Ground-Truth baseline before averaging:

```math
e_t^{\mathrm{rel}}[\%]
=
100\,
\frac{
\left\lVert
\mathbf{t}_{\mathrm{est}}-\mathbf{t}_{\mathrm{GT}}
\right\rVert_2
}{
\left\lVert \mathbf{t}_{\mathrm{GT}} \right\rVert_2
}
```

Because `translation_error_cm = 100 * ||t_est - t_gt||_2`, the same quantity can be computed numerically from the stored reporter fields as

```math
e_t^{\mathrm{rel}}[\%]
=
\frac{\mathrm{translation\_error\_cm}}
     {\mathrm{gt\_baseline\_m}}
```

The reported nominal value is the arithmetic mean of the per-pair relative errors. With four recovered static cameras, six unordered camera pairs are evaluated.

This is deliberately different from dividing the mean translation error by the mean baseline length. Normalization is performed per pair first and the normalized values are then averaged.

The robustness plots use the mean unnormalized pairwise translation error in centimetres instead.

## Simulation: pairwise rotation error

For relative rotations `R_est` and `R_gt`, the evaluator uses the angular distance on SO(3):

```math
\Delta \mathbf{R}
=
\mathbf{R}_{\mathrm{GT}}^{\mathsf T}\mathbf{R}_{\mathrm{est}}
```

```math
e_R[\mathrm{deg}]
=
\frac{180}{\pi}
\arccos\!\left(
\frac{\mathrm{tr}(\Delta\mathbf{R})-1}{2}
\right)
```

The reported simulation rotation value is the mean over all evaluable static-camera pairs.

## Real data: cross-camera reprojection consistency

Real recordings do not provide independent Ground-Truth static-camera poses. The real-data evaluator therefore measures internal geometric consistency instead of absolute pose accuracy.

Marker corners are triangulated from accepted moving-camera observations using the recovered moving-camera poses. The reconstructed 3D corners are then projected into static-camera images for which the same marker is observed. For each evaluated corner, the image-space reprojection distance is

```math
e_n
=
\left\lVert
\hat{\mathbf{u}}_n-\mathbf{u}_n
\right\rVert_2
```

and the cross-camera RMSE is

```math
\mathrm{RMSE}_{\mathrm{cross}}[\mathrm{px}]
=
\sqrt{
\frac{1}{N}\sum_{n=1}^{N} e_n^2
}
```

where `u_n` is the observed static-image corner and `u_hat_n` is the corresponding projection from the reconstructed geometry.

The standalone marker-consistency evaluator uses non-anchor markers for the aggregate moving-to-static validation statistic. The scale anchor establishes metric scale and is not reused as an independent global validation marker.

A low cross-camera RMSE indicates better agreement between the recovered moving/static camera geometry and the observed marker corners. It is an internal-consistency diagnostic and must not be interpreted as independent Ground-Truth camera-pose accuracy.

## Camera coverage

Camera coverage is the number of static cameras for which a method provides a usable pose. Coverage alone is not an accuracy measure: a method can recover all cameras while still producing a poor geometric estimate.

## AP02 observation-graph connectivity

AP02 additionally reports observation-graph connectivity. Static cameras, retained moving-camera frames, and markers can only share one observable coordinate frame when the accepted observations connect them through the graph. Disconnected graph components have independent gauges, so the implementation does not synthesize a relative transformation between them.

## AP03 metric-scale consistency

AP03 reconstructs camera geometry with COLMAP up to an unknown global scale and subsequently recovers metric scale from known-size ArUco markers. Its scale-dispersion diagnostic is based on the retained marker-derived scale candidates after the method's robust filtering.

If the retained candidates are `s_k`, the recovered global scale is

```math
\hat{s}
=
\mathrm{median}\!\left(\{s_k\}_{k\in\mathcal{I}}\right)
```

and the implementation reports

```math
\mathrm{RStd}[\%]
=
100\,
\frac{
\mathrm{std}\!\left(\{s_k\}_{k\in\mathcal{I}},\,\mathrm{ddof}=0\right)
}{\hat{s}}
```

Only retained scale candidates are included. NumPy's population standard deviation (`ddof=0`) is used.

A low value means that the retained marker measurements agree on a common metric scale. A large value indicates inconsistent reconstructed scale. This diagnostic alone does not establish correct camera geometry, which is why it is interpreted together with coverage and reprojection consistency.

For the exact construction and filtering of AP03 scale candidates, see `src/camera_rig_calibration/methods/ap03/README.md`.
