from __future__ import annotations

from .ap01_common_anchor_policy import install_ap01_common_anchor_policy
from .ap02_convergence_frontdoor_policy import install_ap02_convergence_frontdoor_policy
from .ap02_convergence_reporting_policy import install_ap02_convergence_reporting_policy
from .ap02_partial_reference_reporting_policy import (
    install_ap02_partial_reference_reporting_policy,
)
from .ap03_camera_model_sensitivity_policy import (
    install_ap03_camera_model_sensitivity_policy,
)
from .common_anchor_authority_policy import install_common_anchor_authority_policy
from .final_reporting_frontdoor_policy import install_final_reporting_frontdoor_policy
from .marker_preference_policy import install_marker_preference_policy
from .product_policy import install_product_policy
from .queue_anchor_preference_policy import install_queue_anchor_preference_policy
from .real_ap02_budget_policy import install_real_ap02_budget_policy
from .real_marker_reporting_policy import install_real_marker_reporting_policy
from .real_partial_evaluation_policy import install_real_partial_evaluation_policy
from .real_vehicle_marker_zero_policy import install_real_vehicle_marker_zero_policy
from .reporting_authority_policy import install_reporting_authority_policy
from .result_output_policy import install_result_output_policy
from .result_view_policy import install_result_view_policy
from .rviz_manifest_policy import install_rviz_manifest_policy
from .rviz_method_selection_policy import install_rviz_method_selection_policy
from .submission_bindings import install_submission_bindings
from .submission_policy import install_submission_policy
from .ui_display_policy import install_ui_display_policy


_INSTALLED = False


def install_product_stack() -> None:
    """Install the maintained product-facing policy stack exactly once.

    Installation order is part of the compatibility contract because several
    legacy policy modules wrap the same public entry points. Keeping that order
    in one explicit bootstrap function makes startup deterministic while the
    wrappers are progressively replaced by normal composition during the
    submission refactor.
    """

    global _INSTALLED
    if _INSTALLED:
        return

    # Preserve the exact historical installation order from product_cli.py.
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
    install_real_partial_evaluation_policy()
    install_ap02_partial_reference_reporting_policy()
    install_real_marker_reporting_policy()
    install_final_reporting_frontdoor_policy()
    install_ap02_convergence_frontdoor_policy()
    install_rviz_manifest_policy()
    install_rviz_method_selection_policy()
    install_submission_bindings()
    install_ui_display_policy()
    install_result_view_policy()

    _INSTALLED = True


__all__ = ["install_product_stack"]
