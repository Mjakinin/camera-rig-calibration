from __future__ import annotations

import json
from pathlib import Path
from typing import Any


_INSTALLED = False


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _visible_keys(variants: list[dict[str, Any]]) -> set[tuple[str, str]]:
    families: dict[str, list[dict[str, Any]]] = {}
    for item in variants:
        method = str(item.get("method", ""))
        family = "ap03" if method in {"ap03_single", "ap03_multi"} else method
        families.setdefault(family, []).append(item)
    selected: set[tuple[str, str]] = set()
    for family, items in families.items():
        if family == "ap03":
            multi = [item for item in items if item.get("method") == "ap03_multi"]
            pool = multi or items
        else:
            pool = items
        baseline = [item for item in pool if item.get("label") == "baseline"]
        chosen = sorted(
            baseline or pool,
            key=lambda item: (str(item.get("method")), str(item.get("label"))),
        )[0]
        selected.add((str(chosen.get("method")), str(chosen.get("label"))))
    return selected


def _synchronize_manifest(experiment_root: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    if not manifest.get("available"):
        return manifest
    variants = [
        dict(item)
        for item in manifest.get("variants", [])
        if isinstance(item, dict)
    ]
    visible = _visible_keys(variants)
    changed = False
    for item in variants:
        key = (str(item.get("method")), str(item.get("label")))
        enabled = key in visible
        if item.get("default_visible") != enabled:
            item["default_visible"] = enabled
            changed = True
        if item.get("anchor_edges_default_visible") != enabled:
            item["anchor_edges_default_visible"] = enabled
            changed = True
    if not changed:
        return manifest
    updated = {**manifest, "variants": variants}
    path = experiment_root / "visualization" / "visualization_manifest.json"
    _write_json(path, updated)
    return updated


def install_rviz_manifest_policy() -> None:
    """Keep visualization manifest defaults identical to the generated RViz config."""

    global _INSTALLED
    if _INSTALLED:
        return

    from .visualization import scene
    from .evaluation import reporting

    original = scene.ensure_visualization_artifacts
    if getattr(original, "_rigcal_manifest_primary_defaults", False):
        patched = original
    else:
        def ensure_visualization_artifacts(experiment_root):
            root = Path(experiment_root).resolve()
            manifest = original(root)
            return _synchronize_manifest(root, manifest)

        ensure_visualization_artifacts._rigcal_manifest_primary_defaults = True  # type: ignore[attr-defined]
        scene.ensure_visualization_artifacts = ensure_visualization_artifacts
        patched = ensure_visualization_artifacts

    # reporting imported the function directly at module load time, so update its
    # already-bound symbol as well.
    reporting.ensure_visualization_artifacts = patched
    _INSTALLED = True
