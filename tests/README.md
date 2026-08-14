# Tests

This directory contains the versioned acceptance and regression tests for
`rigcal`. They validate configuration, storage, preflight, queueing, all three
method contracts, simulation handling and scientific reporting.

Tests are grouped by responsibility instead of keeping implementation-facing
regressions in one flat directory:

- `methods/ap01/`, `methods/ap02/`, `methods/ap03/`: method-specific contracts,
  diagnostics and regressions.
- `application/`: CLI, Wizard and product-facing application behavior.
- `architecture/`: bootstrap, registry, compatibility and source-boundary checks.
- `observations/`: marker detection, quality filtering and selection behavior.
- `pipeline/`: shared method contracts, stages, repairs and integration behavior.
- `publication/`: anchor exports, result views, reporting and publication policy.
- `runtime/`: queueing, rerun guards and runtime lifecycle behavior.
- `simulation/`: simulation capture, routes, variants and Wizard integration.
- `storage/`: configuration, datasets, intrinsics, inventory and preparation.
- `visualization/`: common visualization and RViz fallback behavior.
- `sdk/`: method extension SDK contracts.

`conftest.py` remains at the test root so fixtures and the stable repository-root
constant are available to the recursively discovered suite. The root itself is
kept free of subsystem test modules.

The test source belongs in Git and is run by CI. Generated files such as
`.pytest_cache`, `__pycache__`, coverage reports and temporary workspaces are
ignored through the repository `.gitignore`.

Run the normal suite from the repository root:

```bash
pytest -q
```
