# Python package layout

Files at this package root are stable commands, public facades, or small shared
contracts. New implementation code should not be added here by default.

- calibration algorithms: `methods/<method-id>/`;
- new external methods: `method_sdk/` contract plus their own method package;
- terminal UI: `ui/`;
- input/capture: `input/` and `dataset/`;
- observation pipeline: `observation_services/`;
- queue/runtime/preflight: their corresponding `*_services/` package;
- evaluation/publication/visualization: their named packages;
- cross-cutting product composition: `policies/`.

The small root facades preserve installed CLI and import paths while delegating
to these packages. Python modules are never numbered: their dependency and
execution order is explicit in coordinators, not their filenames. See
`docs/repository_structure.md` and `docs/architecture.md` for the full map.
