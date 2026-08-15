# Python package layout

The package root is intentionally structural: productive Python code lives in
focused packages, not in loose root modules. Only `__init__.py` and
`__main__.py` are allowed as root-level Python files.

Canonical ownership:

- product entry points and startup checks: `application/`;
- shared contracts and registries: `core/`;
- dataset discovery, inventory and identity: `dataset/`;
- input preparation and intrinsics profiles: `input/`;
- calibration algorithms: `methods/<method-id>/`;
- observation selection/detection/quality: `observation_services/`;
- queue, runtime and preflight orchestration: their `*_services/` packages;
- storage layout, cleanup, filesystem and packaged assets: `storage_services/`;
- publication and result indexing: `publication_services/`;
- terminal interaction: `ui/`;
- visualization: `visualization/`;
- cross-cutting product composition: `policies/`.

Public service surfaces use `api.py` where a subsystem needs a composed API.
Historical imports such as `camera_rig_calibration.runtime` are supported from
the single `compat/` layer; compatibility must not require physical facade
files in the package root.

`tools/check_source_layout.py` enforces both the module-size budget and the
root-module rule in CI.
