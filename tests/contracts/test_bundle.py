from __future__ import annotations

from pathlib import Path

import pytest

from unity_psf.contracts import (
    BundleArtifact,
    ChannelLayout,
    MeasurementChannelSpec,
    ModelBundleManifest,
    load_bundle_manifest,
    save_bundle_manifest,
    sha256_file,
)


def _bundle(tmp_path: Path) -> tuple[Path, ModelBundleManifest]:
    root = tmp_path / "bundle"
    (root / "prototypes").mkdir(parents=True)
    (root / "instances").mkdir()
    prototype = root / "prototypes" / "astigmatism_base.ckpt"
    instance = root / "instances" / "astigmatism_left.ckpt"
    prototype.write_bytes(b"prototype")
    instance.write_bytes(b"instance")
    manifest = ModelBundleManifest(
        unity_psf_version="0.4.0",
        modalities=("astigmatism",),
        channel_layout=ChannelLayout(
            measurement_channels=(MeasurementChannelSpec("left"),)
        ),
        prototypes=(
            BundleArtifact(
                path="prototypes/astigmatism_base.ckpt",
                sha256=sha256_file(prototype),
                expert_type="astigmatism",
            ),
        ),
        instances=(
            BundleArtifact(
                path="instances/astigmatism_left.ckpt",
                sha256=sha256_file(instance),
                expert_type="astigmatism",
                instance_id="astig-left",
                channel_id="left",
                parent_checkpoint_hash=sha256_file(prototype),
            ),
        ),
    )
    return root, manifest


def test_bundle_manifest_round_trip_verifies_relative_files_and_hashes(tmp_path: Path) -> None:
    root, manifest = _bundle(tmp_path)
    manifest_path = save_bundle_manifest(root / "manifest.yaml", manifest)

    restored = load_bundle_manifest(manifest_path)
    assert restored.schema_version == "unity_psf.bundle.v1"
    assert restored.channel_layout.channel_ids == ("left",)
    assert restored.instances[0].parent_checkpoint_hash == restored.prototypes[0].sha256

    moved_root = tmp_path / "moved-bundle"
    root.rename(moved_root)
    moved = load_bundle_manifest(moved_root / "manifest.yaml")
    assert moved.prototypes[0].path == "prototypes/astigmatism_base.ckpt"


def test_bundle_manifest_rejects_absolute_traversal_missing_and_hash_mismatch(tmp_path: Path) -> None:
    root, manifest = _bundle(tmp_path)
    with pytest.raises(ValueError, match="relative"):
        ModelBundleManifest(
            unity_psf_version="0.4.0",
            modalities=("astigmatism",),
            channel_layout=manifest.channel_layout,
            prototypes=(BundleArtifact(path="/tmp/model.ckpt", sha256="a" * 64, expert_type="astigmatism"),),
        )

    manifest_path = save_bundle_manifest(root / "manifest.yaml", manifest)
    root.joinpath("prototypes", "astigmatism_base.ckpt").write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash"):
        load_bundle_manifest(manifest_path)

    with pytest.raises(ValueError, match="relative"):
        BundleArtifact(path="../outside.ckpt", sha256="a" * 64, expert_type="astigmatism")

    missing_manifest = ModelBundleManifest(
        unity_psf_version="0.4.0",
        modalities=("astigmatism",),
        channel_layout=manifest.channel_layout,
        prototypes=(BundleArtifact(path="prototypes/missing.ckpt", sha256="a" * 64, expert_type="astigmatism"),),
    )
    with pytest.raises(ValueError, match="missing"):
        save_bundle_manifest(root / "bad.yaml", missing_manifest)
