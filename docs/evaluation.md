# Evaluation metrics

This document defines the evaluation quantities used by `rigcal` and the scientific comparison. It is intentionally separate from the calibration methods: simulation Ground Truth is used only after calibration, while real-data evaluation relies on internal consistency because independently measured static-camera poses are not available.

## Simulation: pairwise static-camera pose error

For every pair of recovered static cameras $(i,j)$, the evaluator forms the relative estimated and Ground-Truth transforms in the same pair direction. With relative translation vectors $\mathbf{t}_{\mathrm{est}}$ and $\mathbf{t}_{\mathrm{GT}}$, the absolute translation error is

$$
e_t^{\mathrm{cm}}
=
100\,\left\lVert
\mathbf{t}_{\mathrm{est}}-\mathbf{t}_{\mathrm{GT}}
\right\rVert_2.
$$

where the transforms are represented in metres and the factor $100$ converts the result to centimetres.

The implementation writes this value as `translation_error_cm`. It also stores the Ground-Truth pair baseline

$$
b_{\mathrm{GT}}
=
\left\lVert \mathbf{t}_{\mathrm{GT}} \right\rVert_2,
$$

as `gt_baseline_m`, together with the corresponding rotation error `rotation_error_deg`.

### Relative translation error used for the nominal comparison

For the nominal comparison, each pairwise translation error is normalized by that pair's Ground-Truth baseline before averaging:

$$
e_t^{\mathrm{rel}}[\%]
=
100\,
\frac{
\left\lVert
\mathbf{t}_{\mathrm{est}}-\mathbf{t}_{\mathrm{GT}}
\right\rVert_2
}{
\left\lVert \mathbf{t}_{\mathrm{GT}} \right\rVert_2
}.
$$

Because `translation_error_cm` already contains the metre-to-centimetre factor, the same numerical percentage can be computed directly from the stored reporter fields as

$$
e_t^{\mathrm{rel}}[\%]
=
\frac{\texttt{translation\_error\_cm}}
     {\texttt{gt\_baseline\_m}}.
$$

The reported nominal value is the arithmetic mean of the per-pair relative errors. With four recovered static cameras, six unordered camera pairs are evaluated.

This is deliberately different from dividing the mean translation error by the mean baseline length. Normalization is performed per pair first and the normalized values are then averaged.

The robustness plots use the mean unnormalized pairwise translation error in centimetres instead.

## Simulation: pairwise rotation error

For relative rotations $\mathbf{R}_{\mathrm{est}}$ and $\mathbf{R}_{\mathrm{GT}}$, the evaluator uses the angular distance on $\mathrm{SO}(3)$. First,

$$
\Delta\mathbf{R}
=
\mathbf{R}_{\mathrm{GT}}^{\mathsf T}\mathbf{R}_{\mathrm{est}},
$$

then

$$
e_R[\mathrm{deg}]
=
\frac{180}{\pi}
\arccos\!\left(
\frac{\operatorname{tr}(\Delta\mathbf{R})-1}{2}
\right).
$$

The implementation clamps the arccos argument to the valid interval $[-1,1]$ for numerical robustness. The reported simulation rotation value is the mean over all evaluable static-camera pairs.

## Real data: cross-camera reprojection consistency

Real recordings do not provide independent Ground-Truth static-camera poses. The real-data evaluator therefore measures internal geometric consistency instead of absolute pose accuracy.

Marker corners are triangulated from accepted moving-camera observations using the recovered moving-camera poses. The reconstructed 3D corners are then projected into static-camera images for which the same marker is observed. For each evaluated corner, the image-space reprojection distance is

$$
e_n
=
\left\lVert
\hat{\mathbf{u}}_n-\mathbf{u}_n
\right\rVert_2,
$$

and the cross-camera RMSE is

$$
\operatorname{RMSE}_{\mathrm{cross}}[\mathrm{px}]
=
\sqrt{
\frac{1}{N}\sum_{n=1}^{N} e_n^2
}.
$$

Here $\mathbf{u}_n$ is the observed static-image corner and $\hat{\mathbf{u}}_n$ is the corresponding projection from the reconstructed geometry.

The standalone marker-consistency evaluator uses non-anchor markers for the aggregate moving-to-static validation statistic. The scale anchor establishes metric scale and is not reused as an independent global validation marker.

A low cross-camera RMSE indicates better agreement between the recovered moving/static camera geometry and the observed marker corners. It is an internal-consistency diagnostic and must not be interpreted as independent Ground-Truth camera-pose accuracy.

## Camera coverage

Camera coverage is the number of static cameras for which a method provides a usable pose. Coverage alone is not an accuracy measure: a method can recover all cameras while still producing a poor geometric estimate.

## AP02 observation-graph connectivity

AP02 additionally reports observation-graph connectivity. Static cameras, retained moving-camera frames, and markers can only share one observable coordinate frame when the accepted observations connect them through the graph. Disconnected graph components have independent gauges, so the implementation does not synthesize a relative transformation between them.

## AP03 metric-scale consistency

AP03 reconstructs camera geometry with COLMAP up to an unknown global scale and subsequently recovers metric scale from known-size ArUco markers. Its scale-dispersion diagnostic is based on the retained marker-derived scale candidates after the method's robust filtering.

For retained candidates $s_k$, $k\in\mathcal{I}$, the recovered global scale is

$$
\hat{s}
=
\operatorname{median}\!\left(\{s_k\}_{k\in\mathcal{I}}\right),
$$

and the implementation reports

$$
\mathrm{RStd}[\%]
=
100\,
\frac{
\operatorname{std}\!\left(\{s_k\}_{k\in\mathcal{I}},\,\mathrm{ddof}=0\right)
}{
\hat{s}
}.
$$

Only retained scale candidates are included. NumPy's population standard deviation (`ddof=0`) is used.

A low value means that the retained marker measurements agree on a common metric scale. A large value indicates inconsistent reconstructed scale. This diagnostic alone does not establish correct camera geometry, which is why it is interpreted together with coverage and reprojection consistency.

For the exact construction and filtering of AP03 scale candidates, see `src/camera_rig_calibration/methods/ap03/README.md`.
