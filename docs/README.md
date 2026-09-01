# Documentation

This directory contains the detailed documentation behind the shorter root [`README.md`](../README.md).

Use the root README for the normal user workflow. Come here when you need configuration contracts, architecture details, repository layout, evaluation definitions, or extension information.

## Documentation map

- [`architecture.md`](architecture.md) — package boundaries, execution flow, publication and major design decisions.
- [`configuration.md`](configuration.md) — schema-v5 configuration, queues, simulation batches, observation quality, anchors and advanced method settings.
- [`evaluation.md`](evaluation.md) — simulation and real-data evaluation metrics, including pairwise pose error, cross-camera reprojection consistency, coverage, graph connectivity and AP03 scale dispersion.
- [`repository_structure.md`](repository_structure.md) — purpose and ownership of repository directories.
- [`method_sdk.md`](method_sdk.md) — contract for implementing/connecting calibration methods.
- [`extensions.md`](extensions.md) — extension points and integration guidance.
- [`paper/README.md`](paper/README.md) — study-facing scientific reproducibility notes and pointers to the relevant method/result documentation.
- `images/overview/` — presentation images used by the root README.

## Where should new documentation go?

Keep the root README focused on:

- what the project does;
- installation;
- simulation and real-data quick starts;
- the three calibration methods;
- results and visualization;
- links to deeper material.

Put implementation contracts, evaluation definitions and long technical explanations here in `docs/` instead of continuously expanding the root README.
