"""Compatibility facade for dataset preparation and finalization."""
from __future__ import annotations

from .preparation_files import (
    IMAGE_SUFFIXES,
    PreparationPlan,
    _configured_source_files,
    _copy_file,
    _copy_image_as_png,
    _copy_moving_frames,
    _copy_static_camera,
    _extract_static_video_frame,
    _hash_sources,
    _image_files,
    _link_or_copy_file,
    _materialize_tree,
    _normalize_intrinsics,
    _prepared_source_files,
    _sha256,
)
from .preparation_finalization import finalize_dataset
from .preparation_planning import (
    _build_real_preparation_plan,
    _fingerprint_sources,
    _preparation_fingerprint,
    _real_acquisition_payload,
    _real_acquisition_sources,
    _resolved_moving_intrinsics,
    build_preparation_plan,
)
