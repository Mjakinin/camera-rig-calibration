# Configuration assets (`config`)

`config/` contains **managed, reusable configuration assets** that belong to the repository or are intentionally published by `rigcal`.

The main user-facing content is:

```text
config/intrinsics/
```

Managed intrinsics profiles are immutable and stored by profile identity/hash, for example:

```text
config/intrinsics/<profile-id>/<profile-hash>/
  intrinsics.json
  profile.yaml
  reports / source records
```

Use the application to create, select, and manage these profiles:

```text
rigcal
  → Manage intrinsics profiles
```

or the intrinsics step inside a real-data calibration workflow.

## Runtime experiment configuration

Normal experiment queues and resolved run configurations are **not** authored here. They are created under `workspace/` and their requested/resolved copies are preserved with published method provenance under `results/`.

For the full configuration schema and advanced parameters, see [`../docs/configuration.md`](../docs/configuration.md).

Avoid manually editing hashed managed profiles after publication; create/recalculate a new profile instead so provenance remains explicit.
