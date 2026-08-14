# Product policies

These modules install cross-cutting product behavior around stable facades;
they never contain AP01/AP02/AP03 solver implementations.

- selection and authority: marker preference, queue/common anchors, AP01
  anchor export, and real-vehicle marker defaults;
- reporting: convergence, partial-result, real-marker, quality, and final
  report front doors;
- presentation: result view/output, RViz selection/manifests, and UI display;
- composition: product/submission defaults and their late-bound bindings.

`camera_rig_calibration.bootstrap` is the single ordered installer. A new
calibration method normally uses the Method SDK and does not need a new policy
module.
