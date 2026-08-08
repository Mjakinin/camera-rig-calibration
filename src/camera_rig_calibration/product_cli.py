from __future__ import annotations

from .common_anchor_authority_policy import install_common_anchor_authority_policy
from .marker_preference_policy import install_marker_preference_policy
from .product_policy import install_product_policy
from .queue_anchor_preference_policy import install_queue_anchor_preference_policy
from .reanchor_existing_results_policy import install_reanchor_existing_results_policy
from .reporting_authority_policy import install_reporting_authority_policy
from .result_output_policy import install_result_output_policy
from .submission_bindings import install_submission_bindings
from .submission_policy import install_submission_policy
from .ui_display_policy import install_ui_display_policy


# Install product defaults before the public CLI imports the wizard and queue
# modules. Scientific method implementations remain unchanged; these layers own
# category-specific defaults, data-driven pre-method selections, product wording,
# reporting authority, derived-publication policy and visualization defaults.
install_product_policy()
install_reporting_authority_policy()
install_submission_policy()
install_marker_preference_policy()
# Match common-anchor eligibility to the actual method anchor exporters before
# the queue intersects per-method automatic candidates.
install_common_anchor_authority_policy()
install_queue_anchor_preference_policy()
# Existing method estimates may be expressed in a newly selected common anchor
# without rerunning native calibration or COLMAP when the saved geometry supports it.
install_reanchor_existing_results_policy()
install_result_output_policy()
# Bind every already-imported preflight/runtime consumer only after all
# selection wrappers are installed, so Wizard and --config use the same path.
install_submission_bindings()
install_ui_display_policy()

from .cli import main  # noqa: E402


__all__ = ["main"]
