# AP03 — shared COLMAP with ArUco scale

AP03 reconstructs static and moving images together once, then evaluates a
single-marker diagnostic scale and a multi-marker primary scale.

Execution order:

1. `prepare_colmap.py` — create grouped COLMAP input
2. `reconstruct_stage.py` / `reconstruct.py` — run the shared reconstruction
3. `inspect_stage.py` / `inspect.py` — select and validate the sparse model
4. `estimate_scale.py` / `scale_core.py` — estimate single and multi scale
5. `report.py` — normalize both results, with multi-marker scale as primary

Supporting files:

- `pipeline.py` declares the stage order and resolved settings.
- `scale_common.py` contains AP03-specific reconstruction/scale helpers.

The COLMAP reconstruction is shared; single and multi are not independent
reconstructions.
