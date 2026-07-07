# Reporting

Transforms evaluated data into
`results/bus_real_data/99_FINAL_RESULTS_FOR_REPORT/`.

Canonical files:

```text
build_final_results.py
run_refresh_final_results.sh
visualization/
```

Validation only:

```bash
bash run/bus_real_data/reporting/run_refresh_final_results.sh --reuse-baseline
```

Promotion:

```bash
bash run/bus_real_data/reporting/run_refresh_final_results.sh   --reuse-baseline --promote
```

The validated workflow discovers 16 variants, preserves scientific diagnostics,
builds deterministically, avoids duplicate rows and starts no method pipeline.
