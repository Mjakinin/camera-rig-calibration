from __future__ import annotations


_INSTALLED = False


def install_real_ap02_budget_policy() -> None:
    """Set the final AP02 BA safety budget for newly created real-vehicle jobs.

    This changes only the product default used when the Wizard creates a new
    Real Vehicle method job. It does not rewrite explicit user configs and it
    does not change the frozen Simulation baseline/ablation contract.

    Static-only BA keeps the proven 80-evaluation ceiling. Combined BA gets a
    160-evaluation ceiling so ftol/xtol/gtol can terminate normally instead of
    the 80-evaluation cap becoming the stopping criterion on larger real-data
    problems.
    """
    global _INSTALLED
    if _INSTALLED:
        return

    from . import wizard
    from .product_policy import _DATASET_CONTEXT

    original = wizard._new_method_job
    if getattr(original, "_rigcal_real_ap02_budget_policy", False):
        _INSTALLED = True
        return

    def new_method_job(*args, **kwargs):
        job = original(*args, **kwargs)
        if _DATASET_CONTEXT.get() != "simulation":
            ap02 = job.methods.ap02.model_copy(
                update={
                    "static_only_ba_max_function_evaluations": 80,
                    "combined_ba_max_function_evaluations": 160,
                }
            )
            job.methods = job.methods.model_copy(
                update={"ap02": ap02},
                deep=True,
            )
            wizard._refresh_method_job_label(job)
        return job

    new_method_job._rigcal_real_ap02_budget_policy = True  # type: ignore[attr-defined]
    wizard._new_method_job = new_method_job
    _INSTALLED = True


__all__ = ["install_real_ap02_budget_policy"]
