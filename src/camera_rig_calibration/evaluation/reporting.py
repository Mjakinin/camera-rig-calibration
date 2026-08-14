from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml

from ..anchor_export import ensure_experiment_anchor_exports
from ..anchor_export.geometry import rotation_to_quaternion
from ..visualization.scene import ensure_visualization_artifacts
from .ap03_derived import ensure_ap03_derived_results
from .simulation_ground_truth import (
    ensure_simulation_ground_truth,
    resolve_simulation_ground_truth,
)

from ..methods.common.geometry import (
    R_to_rpy_deg,
    R_to_rvec,
    invT,
    make_T,
    rot_error_deg,
    rpy_to_R,
    rvec_to_R,
)

from .reporting_core import (
    PoseRecord,
    _now,
    _read_json,
    _write_json,
    _write_csv,
    _write_text,
    _sha256,
    _float,
    _finite,
    _fmt,
    _mean,
    _median,
    _maximum,
    _text_table,
    _pose_from_row,
    load_pose_records,
    _pose_columns,
    _direction,
    _angle_between,
    pairwise_rows,
)
from .reporting_configuration import (
    _configuration_summary,
    _config_text,
    _baseline_contract,
    _baseline_contract_text,
)
from .reporting_quality import (
    _quality_details,
)
from .reporting_diagnostics import (
    _method_diagnostics,
    _scale_comparison_rows,
    _scale_comparison_text,
)
from .reporting_method import (
    _method_report_text,
    refresh_method_reports,
    complete_existing_dataset,
    run_real_marker_consistency,
)
from .reporting_simulation_geometry import (
    _repository_root,
    _matrix,
    _simulation_gt_maps,
    _simulation_pairwise,
    _anchor_camera_gt_rows,
    _anchor_pose_records,
    _ground_truth_anchor_records,
    _pose_alignment,
    _apply_alignment,
    _camera_map_rows,
    _point_alignment,
    _summary,
    _simulation_primary_text,
    _camera_map_text,
    _ap02_marker_map,
    _latest_marker_report,
    _real_variant_disagreement,
)
from .reporting_real import (
    _real_results_text,
)
from .reporting_simulation import (
    _simulation_results,
    _factor_report,
    path_name,
    _refresh_factor_reports,
    _write_route2_baseline_comparison,
)
from .reporting_orchestration import (
    write_scientific_experiment_reports,
)

__all__ = [
    'PoseRecord',
    '_now',
    '_read_json',
    '_write_json',
    '_write_csv',
    '_write_text',
    '_sha256',
    '_float',
    '_finite',
    '_fmt',
    '_mean',
    '_median',
    '_maximum',
    '_text_table',
    '_pose_from_row',
    'load_pose_records',
    '_pose_columns',
    '_direction',
    '_angle_between',
    'pairwise_rows',
    '_configuration_summary',
    '_config_text',
    '_baseline_contract',
    '_baseline_contract_text',
    '_quality_details',
    '_method_diagnostics',
    '_scale_comparison_rows',
    '_scale_comparison_text',
    '_method_report_text',
    'refresh_method_reports',
    'complete_existing_dataset',
    'run_real_marker_consistency',
    '_repository_root',
    '_matrix',
    '_simulation_gt_maps',
    '_simulation_pairwise',
    '_anchor_camera_gt_rows',
    '_anchor_pose_records',
    '_ground_truth_anchor_records',
    '_pose_alignment',
    '_apply_alignment',
    '_camera_map_rows',
    '_point_alignment',
    '_summary',
    '_simulation_primary_text',
    '_camera_map_text',
    '_ap02_marker_map',
    '_latest_marker_report',
    '_real_variant_disagreement',
    '_real_results_text',
    '_simulation_results',
    '_factor_report',
    'path_name',
    '_refresh_factor_reports',
    '_write_route2_baseline_comparison',
    'write_scientific_experiment_reports',
]
