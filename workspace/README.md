# Local workspace (`workspace`)

`workspace/` is managed by `rigcal` and contains **temporary or regenerable runtime state**, not the final scientific result.

Typical contents include:

- preparation/method caches;
- saved queue/configuration snapshots;
- resumable temporary transactions;
- simulation batches;
- visualization sessions.

Generated workspace contents are ignored by Git and are not part of the submission artifact.

## Important distinction

```text
data_local/   → user input
workspace/    → temporary execution state / caches
results/      → published scientific results
```

Interrupted, selection-waiting, or publication-failed work may remain under `workspace/temporary_runs/` so it can be inspected or resumed.

Do not manually remove files from an active temporary transaction. Use **Manage incomplete runs** or **Cleanup storage** from `rigcal` when possible.

A successful calibration is published under `results/`; temporary workspace paths are not the authoritative result.
