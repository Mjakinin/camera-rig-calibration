from __future__ import annotations


def install_submission_bindings() -> None:
    """Rebind already-imported consumers to the submission selection policy.

    ``product_policy`` imports the wizard before the public CLI starts, and the
    wizard imports queue/preflight/runtime modules that bind observation helpers
    with ``from ... import ...``.  The submission policy deliberately wraps those
    helpers afterwards, so every already-loaded consumer must point at the wrapped
    functions as well.  This keeps interactive wizard runs, saved-config CLI runs,
    queue preflight, selection previews and queue freezing behavior identical.
    """

    from . import observations, preflight, queueing, runtime, submission_policy, wizard
    from .ap01_auto_direct import automatic_ap01_direct_target

    # The resolver wrapper defined in submission_policy looks up this module
    # global at call time. Replace its bootstrap selector with the exact AP01
    # baseline quality/medoid-MAD selector before any preflight can run.
    submission_policy._automatic_ap01_direct_target = automatic_ap01_direct_target

    preflight.resolve_selections = observations.resolve_selections
    wizard.resolve_selections = observations.resolve_selections
    runtime.resolve_selections = observations.resolve_selections

    queueing.freeze_selections = observations.freeze_selections
    runtime.freeze_selections = observations.freeze_selections

    # The robustness strategy remains readable for old schema-v5 configs and
    # historical diagnostics, but it is not a supported submission baseline and
    # must not look like a normal operator-tunable AP01 mode.
    current_rows = wizard._setting_rows
    if not getattr(current_rows, "_rigcal_submission_ap01_rows", False):
        def setting_rows(job, groups=None):
            rows = current_rows(job, groups)
            if job.method_id == "ap01":
                rows = [
                    row
                    for row in rows
                    if row[0] not in {"ap01_advanced_strategy", "ap01_direct_target"}
                ]
            return rows

        setting_rows._rigcal_submission_ap01_rows = True  # type: ignore[attr-defined]
        wizard._setting_rows = setting_rows
