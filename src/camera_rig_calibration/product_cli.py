from __future__ import annotations

from .common_anchor_authority_policy import install_common_anchor_authority_policy
from .marker_preference_policy import install_marker_preference_policy
from .product_policy import install_product_policy
from .queue_anchor_preference_policy import install_queue_anchor_preference_policy
from .real_marker_reporting_policy import install_real_marker_reporting_policy
from .real_vehicle_marker_zero_policy import install_real_vehicle_marker_zero_policy
from .reporting_authority_policy import install_reporting_authority_policy
from .result_output_policy import install_result_output_policy
from .rviz_manifest_policy import install_rviz_manifest_policy
from .submission_bindings import install_submission_bindings
from .submission_policy import install_submission_policy
from .ui_display_policy import install_ui_display_policy


# Install product defaults before the public CLI imports the wizard and queue.
# Scientific method implementations remain unchanged; these layers own the
# submission-facing defaults, selection contract, reporting and visualization.
install_product_policy()
install_reporting_authority_policy()
install_submission_policy()
install_marker_preference_policy()
install_common_anchor_authority_policy()
# Real Vehicle has one canonical marker contract: marker 0 is retained whenever
# observed. A different marker is allowed only when marker 0 is absent.
install_real_vehicle_marker_zero_policy()
install_queue_anchor_preference_policy()
install_result_output_policy()
install_real_marker_reporting_policy()
install_rviz_manifest_policy()
# Bind already-imported preflight/runtime consumers only after all selection
# wrappers are installed, so Wizard and --config use the identical path.
install_submission_bindings()
install_ui_display_policy()

from .cli import main  # noqa: E402


__all__ = ["main"]
