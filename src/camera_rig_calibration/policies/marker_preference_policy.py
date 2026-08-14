from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any


_INSTALLED = False


def _category_name(config: Any) -> str:
    value = getattr(config.dataset, "category", "real_vehicle")
    return str(getattr(value, "value", value))


def _preferred_marker_for_category(category: str) -> int:
    return 14 if category == "simulation" else 0


def _candidate_by_id(payload: dict[str, Any], section: str, marker_id: int) -> dict[str, Any] | None:
    for item in payload.get(section, {}).get("candidates", []):
        try:
            if int(item.get("id")) == int(marker_id):
                return item
        except (TypeError, ValueError):
            continue
    return None


def _ap02_preference_usable(payload: dict[str, Any], marker_id: int) -> bool:
    item = _candidate_by_id(payload, "ap02_reference_marker", marker_id)
    return bool(
        item
        and item.get("compatible", False)
        and item.get("automatic_candidate", False)
    )


def _ap03_preference_usable(payload: dict[str, Any], marker_id: int) -> bool:
    item = _candidate_by_id(payload, "ap03_single_scale_marker", marker_id)
    return bool(
        item
        and item.get("ap03_compatible", False)
        and item.get("automatic_candidate", False)
    )


def _probe_config(
    config: Any,
    *,
    ap02_reference: int | str,
    ap02_mode: str,
    ap03_single: int | str,
    evaluation_anchor: int | str,
) -> Any:
    """Build an auto-resolvable preflight config without mutating the saved request."""

    ap02 = config.methods.ap02.model_copy(
        update={
            "reference_marker_selection_mode": ap02_mode,
            "reference_marker_id": ap02_reference,
        }
    )
    ap03 = config.methods.ap03.model_copy(
        update={
            "single": config.methods.ap03.single.model_copy(
                update={"scale_marker_id": ap03_single}
            )
        },
        deep=True,
    )
    methods = config.methods.model_copy(
        update={"ap02": ap02, "ap03": ap03}, deep=True
    )
    evaluation = config.evaluation.model_copy(
        update={
            "anchor_marker_id": evaluation_anchor,
            "anchor_selection_mode": "auto",
        }
    )
    selection = config.selection.model_copy(update={"mode": "auto"})
    return config.__class__.model_validate(
        config.model_copy(
            update={
                "methods": methods,
                "evaluation": evaluation,
                "selection": selection,
            },
            deep=True,
        ).model_dump(mode="python")
    )


def _install_selection_preference_policy() -> None:
    from .. import observations

    original = observations.resolve_selections
    if getattr(original, "_rigcal_marker_preference", False):
        return

    def resolve_selections(config, observations_root):
        category = _category_name(config)
        category_preferred = _preferred_marker_for_category(category)

        ap02_setting = config.methods.ap02
        ap02_preferred = (
            int(ap02_setting.reference_marker_id)
            if (
                ap02_setting.reference_marker_selection_mode == "auto"
                and isinstance(ap02_setting.reference_marker_id, int)
            )
            else None
        )
        ap03_value = config.methods.ap03.single.scale_marker_id
        # Category defaults are preferences. Other explicit integer IDs remain
        # strict selections and are deliberately allowed to fail if unavailable.
        ap03_preferred = (
            int(ap03_value)
            if isinstance(ap03_value, int)
            and int(ap03_value) == category_preferred
            else None
        )
        evaluation_preferred = (
            int(config.evaluation.anchor_marker_id)
            if (
                config.evaluation.anchor_selection_mode == "auto"
                and isinstance(config.evaluation.anchor_marker_id, int)
            )
            else None
        )

        if not any(
            value is not None
            for value in (ap02_preferred, ap03_preferred, evaluation_preferred)
        ):
            return original(config, observations_root)

        # First resolve the pure automatic recommendations. This guarantees that
        # a missing preferred marker never blocks the queue before fallback.
        probe = _probe_config(
            config,
            ap02_reference="auto" if ap02_preferred is not None else ap02_setting.reference_marker_id,
            ap02_mode="auto" if ap02_preferred is not None else ap02_setting.reference_marker_selection_mode,
            ap03_single="auto" if ap03_preferred is not None else ap03_value,
            evaluation_anchor="auto" if evaluation_preferred is not None else config.evaluation.anchor_marker_id,
        )
        resolved = original(probe, observations_root)

        # AP02 changes the connected marker component used by common evaluation,
        # so re-resolve once with the preferred reference if it is genuinely an
        # automatic-quality candidate. A missing/incompatible preference stays on
        # the already resolved automatic recommendation.
        ap02_used_preference = bool(
            ap02_preferred is not None
            and _ap02_preference_usable(resolved.payload, ap02_preferred)
        )
        if ap02_used_preference:
            probe = _probe_config(
                config,
                ap02_reference=int(ap02_preferred),
                ap02_mode="explicit",
                ap03_single="auto" if ap03_preferred is not None else ap03_value,
                evaluation_anchor="auto" if evaluation_preferred is not None else config.evaluation.anchor_marker_id,
            )
            resolved = original(probe, observations_root)

        payload = copy.deepcopy(resolved.payload)

        if ap02_preferred is not None:
            selected = int(resolved.ap02_reference_marker_id)
            section = payload.setdefault("ap02_reference_marker", {})
            section.update(
                {
                    "configured": int(ap02_preferred),
                    "selection_mode": "preferred_with_auto_fallback",
                    "selected": selected,
                    "preferred_marker_id": int(ap02_preferred),
                    "fallback_used": not ap02_used_preference,
                    "reason": (
                        f"preferred marker {ap02_preferred} passed automatic AP02 support; preference frozen"
                        if ap02_used_preference
                        else f"preferred marker {ap02_preferred} was unavailable or incompatible; deterministic AP02 auto fallback selected marker {selected}"
                    ),
                }
            )

        single_selected = int(resolved.ap03_single_scale_marker_id)
        ap03_used_preference = False
        if ap03_preferred is not None:
            ap03_used_preference = _ap03_preference_usable(
                payload, ap03_preferred
            )
            if ap03_used_preference:
                single_selected = int(ap03_preferred)
            section = payload.setdefault("ap03_single_scale_marker", {})
            section.update(
                {
                    "configured": int(ap03_preferred),
                    "selection_mode": "preferred_with_auto_fallback",
                    "selected": single_selected,
                    "preferred_marker_id": int(ap03_preferred),
                    "fallback_used": not ap03_used_preference,
                    "reason": (
                        f"preferred marker {ap03_preferred} passed automatic AP03 Single support; preference frozen"
                        if ap03_used_preference
                        else f"preferred marker {ap03_preferred} was unavailable or incompatible; deterministic AP03 Single auto fallback selected marker {single_selected}"
                    ),
                }
            )

        evaluation_selected = resolved.evaluation_anchor_marker_id
        evaluation_used_preference = False
        if evaluation_preferred is not None and config.evaluation.enabled:
            automatic_ids = {
                int(value)
                for value in payload.get("evaluation_anchor", {}).get(
                    "automatic_observation_candidates", []
                )
            }
            evaluation_used_preference = int(evaluation_preferred) in automatic_ids
            if evaluation_used_preference:
                evaluation_selected = int(evaluation_preferred)
            section = payload.setdefault("evaluation_anchor", {})
            section.update(
                {
                    "configured": int(evaluation_preferred),
                    "selection_mode": "preferred_with_auto_fallback",
                    "selected": evaluation_selected,
                    "preferred_marker_id": int(evaluation_preferred),
                    "fallback_used": not evaluation_used_preference,
                    "reason": (
                        f"preferred common anchor {evaluation_preferred} passed common automatic support; preference frozen"
                        if evaluation_used_preference
                        else f"preferred common anchor {evaluation_preferred} was unavailable or incompatible; deterministic common-anchor auto fallback selected marker {evaluation_selected}"
                    ),
                }
            )

        payload["category_marker_preference"] = {
            "dataset_category": category,
            "category_default_marker_id": category_preferred,
            "ap02": {
                "preferred": ap02_preferred,
                "selected": int(resolved.ap02_reference_marker_id),
                "fallback_used": (
                    not ap02_used_preference if ap02_preferred is not None else None
                ),
            },
            "ap03_single": {
                "preferred": ap03_preferred,
                "selected": single_selected,
                "fallback_used": (
                    not ap03_used_preference if ap03_preferred is not None else None
                ),
            },
            "evaluation_anchor": {
                "preferred": evaluation_preferred,
                "selected": evaluation_selected,
                "fallback_used": (
                    not evaluation_used_preference
                    if evaluation_preferred is not None
                    else None
                ),
            },
            "ground_truth_used": False,
        }

        final = observations.ResolvedSelections(
            root_camera=resolved.root_camera,
            ap02_reference_marker_id=int(resolved.ap02_reference_marker_id),
            ap03_single_scale_marker_id=single_selected,
            ap03_multi_marker_ids=resolved.ap03_multi_marker_ids,
            evaluation_anchor_marker_id=evaluation_selected,
            marker_ids=resolved.marker_ids,
            payload=payload,
        )

        root = Path(observations_root)
        text = json.dumps(payload, indent=2) + "\n"
        for name in ("SELECTION_CANDIDATES.json", "REFERENCE_SELECTIONS.json"):
            (root / name).write_text(text, encoding="utf-8")
        (root / "REFERENCE_MARKER_ID.txt").write_text(
            f"{final.ap02_reference_marker_id}\n", encoding="utf-8"
        )
        observations.write_selection_candidates_csv(root, payload)
        return final

    resolve_selections._rigcal_marker_preference = True  # type: ignore[attr-defined]
    observations.resolve_selections = resolve_selections


def _install_wizard_marker_defaults() -> None:
    from .. import wizard
    from .product_policy import _DATASET_CONTEXT

    if getattr(wizard, "_MARKER_PREFERENCE_POLICY_INSTALLED", False):
        return

    original_new_method_job = wizard._new_method_job

    def new_method_job(*args, **kwargs):
        job = original_new_method_job(*args, **kwargs)
        category = _DATASET_CONTEXT.get()
        preferred = _preferred_marker_for_category(category)

        ap02 = job.methods.ap02.model_copy(
            update={
                "reference_marker_selection_mode": "auto",
                "reference_marker_id": preferred,
            }
        )
        ap03_update: dict[str, Any] = {
            "single": job.methods.ap03.single.model_copy(
                update={"scale_marker_id": preferred}
            )
        }
        if category != "simulation":
            # Real recordings contain an arbitrary detected marker inventory.
            ap03_update["multi"] = job.methods.ap03.multi.model_copy(
                update={"marker_ids": "auto"}
            )
        ap03 = job.methods.ap03.model_copy(update=ap03_update, deep=True)
        job.methods = job.methods.model_copy(
            update={"ap02": ap02, "ap03": ap03}, deep=True
        )
        job.evaluation = job.evaluation.model_copy(
            update={
                "anchor_marker_id": preferred,
                "anchor_selection_mode": "auto",
            }
        )
        # The integer values above are category preferences, not strict explicit
        # selections. Keep preflight free to fall back automatically.
        job.selection = job.selection.model_copy(update={"mode": "auto"})
        wizard._refresh_method_job_label(job)
        return job

    wizard._new_method_job = new_method_job

    original_method_job_label = wizard._method_job_label

    def method_job_label(job, context_key=None):
        # Simulation auto+14 is the canonical baseline contract even though its
        # fallback semantics are safer than the old hard failure. Normalize only
        # the label calculation; the saved configuration remains fully explicit.
        if _DATASET_CONTEXT.get() != "simulation" or job.method_id != "ap02":
            return original_method_job_label(job, context_key)
        normalized = copy.deepcopy(job)
        ap02 = normalized.methods.ap02
        if (
            ap02.reference_marker_selection_mode == "auto"
            and ap02.reference_marker_id == 14
        ):
            normalized.methods = normalized.methods.model_copy(
                update={
                    "ap02": ap02.model_copy(
                        update={
                            "reference_marker_selection_mode": "baseline",
                            "reference_marker_id": 14,
                        }
                    )
                },
                deep=True,
            )
        return original_method_job_label(normalized, context_key)

    wizard._method_job_label = method_job_label

    original_setting_rows = wizard._setting_rows

    def setting_rows(job, groups=None):
        rows = original_setting_rows(job, groups)
        category = _DATASET_CONTEXT.get()
        preferred = _preferred_marker_for_category(category)
        rendered = []
        for key, group, label, current, baseline, description in rows:
            if key == "evaluation_anchor":
                if (
                    job.evaluation.anchor_selection_mode == "auto"
                    and job.evaluation.anchor_marker_id == preferred
                ):
                    current = f"preferred marker {preferred} -> auto fallback"
                baseline = f"preferred marker {preferred} -> auto fallback"
                description = (
                    f"Category default prefers marker {preferred}. If it lacks common repeat-supported evidence, preflight falls back to the deterministic automatic marker. Editing this row to Auto requests pure automatic ranking; Manual shows detected candidates after preflight."
                )
            elif key == "ap02_reference_mode":
                baseline = "auto (category preference with fallback)"
                description = (
                    f"Default prefers marker {preferred} and falls back automatically if it is unavailable or incompatible. Auto requests pure deterministic ranking; Manual lets you choose a detected marker after preflight. Baseline keeps the strict built-in simulation contract."
                )
            elif key == "ap02_reference_display":
                if (
                    job.methods.ap02.reference_marker_selection_mode == "auto"
                    and job.methods.ap02.reference_marker_id == preferred
                ):
                    current = f"marker {preferred} preferred; auto fallback"
                baseline = f"marker {preferred} preferred; auto fallback"
                description = (
                    "Read-only requested preference. The resolved marker and whether fallback was used are frozen in SELECTION_CANDIDATES.json."
                )
            elif key == "single_marker":
                if job.methods.ap03.single.scale_marker_id == preferred:
                    current = f"marker {preferred} preferred; auto fallback"
                baseline = f"marker {preferred} preferred; auto fallback"
                description = (
                    f"AP03 Single prefers marker {preferred}. If it lacks compatible repeated moving support, preflight falls back to the deterministic automatic scale marker. Edit this row for pure Auto or Manual selection."
                )
            rendered.append((key, group, label, current, baseline, description))
        return rendered

    wizard._setting_rows = setting_rows

    original_configure_guided_selection = wizard._configure_guided_selection

    def configure_guided_selection(
        console,
        job,
        *,
        key,
        label,
        contexts,
        requested_mode=None,
    ):
        if key != "single_marker" or requested_mode is not None:
            return original_configure_guided_selection(
                console,
                job,
                key=key,
                label=label,
                contexts=contexts,
                requested_mode=requested_mode,
            )
        preferred = _preferred_marker_for_category(_DATASET_CONTEXT.get())
        pending = wizard._pending_selection_keys(job)
        current_value = job.methods.ap03.single.scale_marker_id
        current_mode = (
            "preferred"
            if current_value == preferred and key not in pending
            else "manual"
            if key in pending or current_value != "auto"
            else "auto"
        )
        mode = wizard._prompt_enum_choice(
            label,
            current_mode,
            (
                (
                    "preferred",
                    f"prefer marker {preferred}; fall back automatically if it is not a compatible repeated observation",
                ),
                (
                    "auto",
                    "use only the deterministic automatic recommendation",
                ),
                (
                    "manual",
                    "choose from compatible detected markers now or after preflight",
                ),
            ),
        )
        if mode == "preferred":
            job.methods = wizard._methods_with_selection(
                job.methods, key, preferred
            )
            job.deferred_selection_keys.discard(key)
            job.selection = job.selection.model_copy(update={"mode": "auto"})
            for context in contexts:
                job.context_methods[context.key] = wizard._methods_with_selection(
                    wizard._job_methods(job, context.key), key, preferred
                )
                job.context_deferred_selection_keys.setdefault(
                    context.key, set()
                ).discard(key)
                job.context_selections[context.key] = job.selection.model_copy(
                    update={"mode": "auto"}
                )
            return None
        return original_configure_guided_selection(
            console,
            job,
            key=key,
            label=label,
            contexts=contexts,
            requested_mode=mode,
        )

    wizard._configure_guided_selection = configure_guided_selection
    wizard._MARKER_PREFERENCE_POLICY_INSTALLED = True


def _install_reporting_preference_contract() -> None:
    from ..evaluation import reporting

    original = reporting._baseline_contract
    if getattr(original, "_rigcal_marker_preference", False):
        return

    def baseline_contract(*, category, method_payloads, evaluation_anchor):
        contract = original(
            category=category,
            method_payloads=method_payloads,
            evaluation_anchor=evaluation_anchor,
        )
        payloads = {
            (str(item.get("method", "")), str(item.get("label", ""))): item
            for item in method_payloads
        }
        for variant in contract.get("variants", []):
            if str(variant.get("method")) != "ap02":
                continue
            payload = payloads.get(
                (str(variant.get("method", "")), str(variant.get("label", ""))),
                {},
            )
            summary = payload.get("config_summary", {})
            checks = variant.get("checks", {})
            mode_ok = (
                summary.get("reference_marker_selection_mode") == "baseline"
                or (
                    summary.get("reference_marker_selection_mode") == "auto"
                    and summary.get("reference_marker_id") == 14
                )
            )
            resolved_ok = str(
                summary.get("resolved_reference_marker_id")
                or summary.get("reference_marker_id")
            ) == "14"
            checks.pop("reference_mode_baseline", None)
            checks["reference_preference_14_with_auto_fallback"] = bool(mode_ok)
            checks["reference_marker_14"] = bool(resolved_ok)
            variant["passes"] = all(bool(value) for value in checks.values())
        contract["passes"] = bool(contract.get("variants")) and all(
            bool(item.get("passes")) for item in contract.get("variants", [])
        )
        return contract

    baseline_contract._rigcal_marker_preference = True  # type: ignore[attr-defined]
    reporting._baseline_contract = baseline_contract


def install_marker_preference_policy() -> None:
    """Install category marker preferences while preserving explicit/manual choices."""

    global _INSTALLED
    if _INSTALLED:
        return
    _install_selection_preference_policy()
    _install_wizard_marker_defaults()
    _install_reporting_preference_contract()
    _INSTALLED = True


__all__ = ["install_marker_preference_policy"]
