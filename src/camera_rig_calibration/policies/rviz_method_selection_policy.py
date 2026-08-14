from __future__ import annotations

from pathlib import Path


_INSTALLED = False


def install_rviz_method_selection_policy() -> None:
    """Let result option 5 choose a scientifically valid RViz frame.

    Common-anchor views keep the existing overlay behavior. Partial AP02 results
    whose evaluation anchor is not reconstructable are offered as separate native
    component scenes. Disconnected AP02 components are never overlaid because no
    AP02 cross-component transform is observable.
    """

    global _INSTALLED
    if _INSTALLED:
        return

    from .. import wizard
    from ..visualization.ap02_native import (
        discover_ap02_native_scenes,
        ensure_ap02_native_scene,
    )
    from ..visualization.session import launch_visualization_directory

    original = wizard.launch_isolated_rviz
    if getattr(original, "_rigcal_method_scene_selection", False):
        _INSTALLED = True
        return

    def launch_selected(experiment_root: Path, repository_root: Path):
        root = Path(experiment_root).resolve()
        native = discover_ap02_native_scenes(root)
        if not native:
            return original(root, repository_root)

        wizard.typer.echo("\nRViz view:")
        wizard.typer.echo(
            "  1. common evaluation/export frame — only methods with a valid common anchor"
        )
        for index, descriptor in enumerate(native, 2):
            wizard.typer.echo(f"  {index}. {descriptor['display_name']}")
        wizard.typer.echo(
            "Disconnected AP02 components use different local marker frames and are shown separately."
        )
        while True:
            raw = str(
                wizard.typer.prompt("Selection", default="1")
            ).strip()
            try:
                selected = int(raw)
            except ValueError:
                wizard.typer.echo("Choose one listed RViz view.")
                continue
            if selected == 1:
                return original(root, repository_root)
            native_index = selected - 2
            if 0 <= native_index < len(native):
                descriptor = native[native_index]
                visualization = ensure_ap02_native_scene(root, descriptor)
                return launch_visualization_directory(
                    visualization,
                    root,
                    repository_root,
                    scene_label=str(descriptor["scene_id"]),
                )
            wizard.typer.echo("Choose one listed RViz view.")

    launch_selected._rigcal_method_scene_selection = True  # type: ignore[attr-defined]
    wizard.launch_isolated_rviz = launch_selected
    _INSTALLED = True


__all__ = ["install_rviz_method_selection_policy"]
