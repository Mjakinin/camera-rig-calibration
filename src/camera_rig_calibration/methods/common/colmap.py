from __future__ import annotations

from pathlib import Path

from camera_rig_calibration.pipeline.artifacts import require_directory


def sparse_text_model(root: Path) -> Path:
    directory = require_directory(root, label="COLMAP sparse-text model")
    models = sorted(
        candidate.parent
        for candidate in directory.rglob("images.txt")
        if (candidate.parent / "cameras.txt").is_file()
    )
    if not models:
        raise RuntimeError(f"No readable COLMAP text model under {directory}")
    return models[0]
