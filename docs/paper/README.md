# Scientific reproducibility notes

This directory collects study-facing reproducibility notes for `rigcal`. It does not duplicate large raw recordings, generated COLMAP reconstructions, or other bulky experiment data.

> Generated per-run experiment outputs and resolved configurations are not tracked in this public source repository. They are produced locally by `rigcal`; the tracked repository documents their schema, provenance contract, and output layout.

The repository separates the scientific definitions from implementation details so that each can be inspected at the appropriate level:

- [`../evaluation.md`](../evaluation.md) defines the reported simulation and real-data evaluation metrics.
- [`../configuration.md`](../configuration.md) documents requested/resolved configuration handling and advanced settings.
- [`../../results/README.md`](../../results/README.md) documents the generated result layout and provenance files.
- [`../../src/camera_rig_calibration/methods/ap01/README.md`](../../src/camera_rig_calibration/methods/ap01/README.md) documents AP01.
- [`../../src/camera_rig_calibration/methods/ap02/README.md`](../../src/camera_rig_calibration/methods/ap02/README.md) documents AP02.
- [`../../src/camera_rig_calibration/methods/ap03/README.md`](../../src/camera_rig_calibration/methods/ap03/README.md) documents AP03.

## Reproducibility principle

A scientific result should be traceable to the calibration method, the resolved configuration used for that execution, the machine-readable result files, and the corresponding diagnostics/provenance. Ground Truth used for simulation evaluation remains post-hoc evaluation data and is not an input to the calibration methods.

Large raw recordings and reconstruction intermediates are intentionally kept separate from this documentation. When study outputs are archived for review, their resolved configurations and numerical outputs should be kept together so reported tables and plots can be audited without requiring the full raw-data footprint.
