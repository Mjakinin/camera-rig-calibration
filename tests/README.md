# Tests

This directory contains the versioned acceptance and regression tests for
`rigcal`. They validate configuration, storage, preflight, queueing, all three
method contracts, simulation handling and scientific reporting.

Tests are grouped by responsibility instead of keeping every module in one
flat directory:

- `methods/ap01/`, `methods/ap02/`, `methods/ap03/`: method-specific contracts,
  diagnostics and regressions.
- `application/`: CLI and product-facing application behavior.
- `sdk/`: method extension SDK contracts.
- `architecture/`: compatibility and source-boundary regressions.
- Tests that span several subsystems remain at the test root until a single
  ownership boundary is clear.

`conftest.py` remains at the test root so its fixtures are available to the
entire recursively discovered suite.

The test source belongs in Git and is run by CI. Generated files such as
`.pytest_cache`, `__pycache__`, coverage reports and temporary workspaces are
ignored through the repository `.gitignore`.

Run the normal suite from the repository root:

```bash
pytest -q
```
