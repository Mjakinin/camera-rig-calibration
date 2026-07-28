# Tests

This directory contains the versioned acceptance and regression tests for
`rigcal`. They validate configuration, storage, preflight, queueing, all three
method contracts, simulation handling and scientific reporting.

The test source belongs in Git and is run by CI. Generated files such as
`.pytest_cache`, `__pycache__`, coverage reports and temporary workspaces are
ignored through the repository `.gitignore`.

Run the normal suite from the repository root:

```bash
pytest -q
```
