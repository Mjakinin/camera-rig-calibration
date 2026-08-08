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
