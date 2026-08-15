from __future__ import annotations

from ..policies.ap01_common_anchor_policy import install_ap01_common_anchor_policy
from ..policies.ap02_convergence_frontdoor_policy import install_ap02_convergence_frontdoor_policy
from ..policies.ap02_convergence_reporting_policy import install_ap02_convergence_reporting_policy
from ..policies.ap02_partial_reference_reporting_policy import (
    install_ap02_partial_reference_reporting_policy,
)
from ..policies.ap03_camera_model_sensitivity_policy import (
    install_ap03_camera_model_sensitivity_policy,
)
from ..policies.common_anchor_authority_policy import install_common_anchor_authority_policy
from ..policies.dataset_context_policy import install_dataset_context_policy
from ..policies.final_reporting_frontdoor_policy import install_final_reporting_frontdoor_policy
from ..policies.marker_preference_policy import install_marker_preference_policy
from ..policies.product_policy import install_product_policy
from ..policies.queue_anchor_preference_policy import install_queue_anchor_preference_policy
from ..policies.real_ap02_budget_policy import install_real_ap02_budget_policy
from ..policies.real_marker_reporting_policy import install_real_marker_reporting_policy
from ..policies.real_partial_evaluation_policy import install_real_partial_evaluation_policy
from ..policies.real_vehicle_marker_zero_policy import install_real_vehicle_marker_zero_policy
from ..policies.reporting_authority_policy import install_reporting_authority_policy
from ..policies.result_output_policy import install_result_output_policy
from ..policies.result_view_policy import install_result_view_policy
from ..policies.rviz_manifest_policy import install_rviz_manifest_policy
from ..policies.rviz_method_selection_policy import install_rviz_method_selection_policy
from ..policies.submission_bindings import install_submission_bindings
from ..policies.submission_policy import install_submission_policy
from ..policies.submission_quality_policy import install_submission_quality_policy
from ..policies.ui_display_policy import install_ui_display_policy


_INSTALLED = False


def _resolve_wizard_policy_target() -> None:
    """Resolve the legacy wizard alias before private hook policies are applied.

    The compatibility proxy forwards public attribute assignments to its target,
    but private hook names such as ``_new_method_job`` are proxy-local.  Product
    policies intentionally wrap those private hooks, while the modular UI reads
    them from the canonical ``ui.wizard`` module through late bindings.  Touching
    one private hook forces the alias to resolve and rebinds the package-level
    historical name to the canonical module before any policy writes occur.
    """

    from .. import wizard

    _ = wizard._new_method_job


def install_product_stack() -> None:
    """Install the maintained product-facing policy stack exactly once."""

    global _INSTALLED
    if _INSTALLED:
        return

    _resolve_wizard_policy_target()

    install_product_policy()
    install_real_ap02_budget_policy()
    install_reporting_authority_policy()
    install_submission_policy()
    install_marker_preference_policy()
    install_common_anchor_authority_policy()
    install_real_vehicle_marker_zero_policy()
    install_queue_anchor_preference_policy()
    install_ap01_common_anchor_policy()
    install_ap03_camera_model_sensitivity_policy()
    install_ap02_convergence_reporting_policy()
    install_result_output_policy()
    install_submission_quality_policy()
    install_real_partial_evaluation_policy()
    install_ap02_partial_reference_reporting_policy()
    install_real_marker_reporting_policy()
    install_final_reporting_frontdoor_policy()
    install_ap02_convergence_frontdoor_policy()
    install_rviz_manifest_policy()
    install_rviz_method_selection_policy()
    install_submission_bindings()
    install_ui_display_policy()
    install_dataset_context_policy()
    install_result_view_policy()

    _INSTALLED = True


__all__ = ["install_product_stack"]
