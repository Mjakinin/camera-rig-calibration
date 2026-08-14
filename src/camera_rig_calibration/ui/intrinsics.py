"""Interactive management of immutable moving-camera intrinsics profiles."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import typer
from rich.console import Console
from rich.table import Table

from ..intrinsics_profiles import (
    delete_profile,
    discover_intrinsic_profiles,
    intrinsic_profile_references,
    update_profile_alias,
)
from .storage_cleanup import human_size


def run_intrinsics_manager(
    repository_root: Path,
    console: Console,
    *,
    choose: Callable[[str, dict[str, str], str], str],
    create_profile: Callable[[Path, Console], None],
    relative_display: Callable[[Path, Path], str],
    show_error: Callable[[str], None],
) -> None:
    """Manage profiles using navigation callbacks owned by the wizard facade."""
    while True:
        profiles = discover_intrinsic_profiles(repository_root)
        table = Table(title="Moving-camera intrinsics profiles")
        table.add_column("#", justify="right")
        table.add_column("Name / stable key", overflow="fold")
        table.add_column("Resolution")
        table.add_column("Model")
        table.add_column("Size")
        table.add_column("References")
        table.add_column("Storage", overflow="fold")
        references: dict[str, tuple[Path, ...]] = {}
        for index, profile in enumerate(profiles, 1):
            refs = intrinsic_profile_references(repository_root, profile)
            references[profile.key] = refs
            table.add_row(
                str(index),
                f"{profile.label}\n{profile.key}",
                f"{profile.width}x{profile.height}",
                profile.distortion_model,
                human_size(profile.size_bytes),
                str(len(refs)),
                relative_display(profile.root, repository_root),
            )
        if profiles:
            console.print(table)
        else:
            console.print("No managed intrinsics profiles were found.")
        action = choose(
            "Intrinsics profiles",
            {
                "1": "create or recalculate a new immutable profile version",
                "2": "rename a profile display alias",
                "3": "delete a profile",
                "0": "back to main menu",
            },
            "1",
        )
        if action == "0":
            return
        if action == "1":
            create_profile(repository_root, console)
            continue
        if not profiles:
            continue
        number = typer.prompt("Profile number", type=int)
        if number < 1 or number > len(profiles):
            show_error("Invalid profile number.")
            continue
        profile = profiles[number - 1]
        if action == "2":
            alias = typer.prompt(
                "New display name", default=profile.label
            ).strip()
            updated = update_profile_alias(profile, alias)
            console.print(
                f"[green]Display name updated to '{updated.label}'. Stable "
                f"key remains {updated.key}.[/green]"
            )
            continue
        refs = references[profile.key]
        if refs:
            console.print(
                f"Profile is referenced by {len(refs)} saved configuration(s). "
                "Completed scientific results will remain, but a configuration "
                "without a published intrinsics snapshot may no longer be rerunnable."
            )
            for reference in refs[:10]:
                console.print(f"  - {reference}")
            if len(refs) > 10:
                console.print(f"  ... and {len(refs) - 10} more")
        if typer.confirm(
            f"Permanently delete profile {profile.key}?",
            default=False,
        ):
            try:
                delete_profile(repository_root, profile)
            except RuntimeError as exc:
                show_error(str(exc))
                continue
            console.print(
                "[green]Profile deleted. Completed method results were kept.[/green]"
            )


__all__ = ["run_intrinsics_manager"]
