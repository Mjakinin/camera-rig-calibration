from __future__ import annotations

import json
from pathlib import Path

import yaml
import pytest

from camera_rig_calibration.intrinsics_profiles import (
    delete_profile,
    discover_intrinsic_profiles,
    update_profile_alias,
)


def _profile(repository: Path) -> Path:
    root = (
        repository
        / "config/intrinsics/iphone_05x/abcdef123456"
    )
    root.mkdir(parents=True)
    (root / "intrinsics.json").write_text(
        json.dumps(
            {
                "width": 3840,
                "height": 2160,
                "distortion_model": "plumb_bob",
                "K": [1.0] * 9,
            }
        )
    )
    (root / "profile.yaml").write_text(
        yaml.safe_dump(
            {
                "profile_id": "iphone_05x",
                "fingerprint": "abcdef1234567890",
                "intrinsics": "intrinsics.json",
            }
        )
    )
    return root


def test_alias_does_not_change_stable_profile_key(
    tmp_path: Path,
) -> None:
    _profile(tmp_path)
    profile = discover_intrinsic_profiles(tmp_path)[0]
    key = profile.key

    renamed = update_profile_alias(profile, "iPhone 0.5x 4K")

    assert renamed.label == "iPhone 0.5x 4K"
    assert renamed.key == key
    assert discover_intrinsic_profiles(tmp_path)[0].label == "iPhone 0.5x 4K"


def test_unreferenced_profile_can_be_deleted(tmp_path: Path) -> None:
    root = _profile(tmp_path)
    profile = discover_intrinsic_profiles(tmp_path)[0]

    delete_profile(tmp_path, profile)

    assert not root.exists()


def test_profile_used_by_active_temporary_queue_cannot_be_deleted(
    tmp_path: Path,
) -> None:
    root = _profile(tmp_path)
    profile = discover_intrinsic_profiles(tmp_path)[0]
    config = (
        tmp_path
        / "workspace/temporary_runs/active_queue/resolved/ap03.yaml"
    )
    config.parent.mkdir(parents=True)
    config.write_text(
        yaml.safe_dump(
            {"moving_camera": {"intrinsics_profile": profile.key}}
        )
    )

    with pytest.raises(RuntimeError, match="active temporary queue"):
        delete_profile(tmp_path, profile)
    assert root.is_dir()


def test_completed_run_reference_is_reported_but_does_not_block_delete(
    tmp_path: Path,
) -> None:
    root = _profile(tmp_path)
    profile = discover_intrinsic_profiles(tmp_path)[0]
    config = (
        tmp_path
        / "results/real_vehicle/3Hz/example/methods/ap02/baseline/"
        "provenance/resolved_config.yaml"
    )
    config.parent.mkdir(parents=True)
    config.write_text(
        yaml.safe_dump(
            {"moving_camera": {"intrinsics_profile": profile.key}}
        )
    )

    delete_profile(tmp_path, profile)

    assert not root.exists()


def test_unknown_hidden_field_is_ignored_and_profile_is_visible(
    tmp_path: Path,
) -> None:
    root = _profile(tmp_path)
    manifest = root / "profile.yaml"
    payload = yaml.safe_load(manifest.read_text())
    payload["hidden"] = True
    manifest.write_text(yaml.safe_dump(payload))

    profiles = discover_intrinsic_profiles(tmp_path)

    assert len(profiles) == 1
    assert not hasattr(profiles[0], "hidden")


def test_repository_intrinsics_use_managed_fingerprint_layout() -> None:
    repository = Path(__file__).resolve().parents[1]

    profiles = {
        profile.profile_id: profile
        for profile in discover_intrinsic_profiles(repository)
    }

    assert {
        "iphone_05x_4k",
        "iphone_1x_4k_3840x2160",
    }.issubset(profiles)
    assert profiles["iphone_05x_4k"].root.name == "d5444b68272f"
    assert (
        profiles["iphone_1x_4k_3840x2160"].root.name
        == "3ed2f8d6f7fe"
    )
    assert not (
        repository
        / "config/intrinsics/IMG_4320_v1"
    ).exists()
