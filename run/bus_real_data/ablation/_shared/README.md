# Shared Ablation Orchestration

Reusable variant preparation, execution, evaluation, packaging and canonical
report refresh logic.

A clean workflow prepares one controlled variant, runs all applicable methods,
evaluates explicit coverage, writes one `FINAL_RESULTS` package and refreshes
central reports only after successful packaging.

Fresh and stale method outputs must never be mixed.
