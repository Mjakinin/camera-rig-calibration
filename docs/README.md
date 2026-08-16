# Documentation

This directory contains the detailed documentation behind the shorter root [`README.md`](../README.md).

Use the root README for the normal user workflow. Come here when you need configuration contracts, architecture details, repository layout, or extension information.

## Documentation map

- [`architecture.md`](architecture.md) — package boundaries, execution flow, publication and major design decisions.
- [`configuration.md`](configuration.md) — schema-v5 configuration, queues, simulation batches, observation quality, anchors and advanced method settings.
- [`repository_structure.md`](repository_structure.md) — purpose and ownership of repository directories.
- [`method_sdk.md`](method_sdk.md) — contract for implementing/connecting calibration methods.
- [`extensions.md`](extensions.md) — extension points and integration guidance.
- [`paper/README.md`](paper/README.md) — placeholder/status page for the accompanying project paper/report.
- `images/overview/` — presentation images used by the root README.

## Where should new documentation go?

Keep the root README focused on:

- what the project does;
- installation;
- simulation and real-data quick starts;
- the three calibration methods;
- results and visualization;
- links to deeper material.

Put implementation contracts and long technical explanations here in `docs/` instead of continuously expanding the root README.

## Paper/report status

The accompanying paper/report is still **work in progress**. The repository therefore links to a status page rather than presenting an unfinished PDF as a final publication. See [`paper/README.md`](paper/README.md).
