"""Portable UnityPSF model-bundle manifest and artifact validation."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Mapping

import yaml

from .checkpoint import _validate_sha256
from .modality import ChannelLayout, PSFModality


BUNDLE_SCHEMA_VERSION = "unity_psf.bundle.v1"


def _relative_path(value: str | Path) -> str:
    raw = str(value).replace("\\", "/")
    posix = PurePosixPath(raw)
    windows = PureWindowsPath(raw)
    if not raw or posix.is_absolute() or windows.is_absolute() or windows.drive:
        raise ValueError(f"bundle artifact path must be relative: {value!r}")
    if ".." in posix.parts or not posix.parts:
        raise ValueError(f"bundle artifact path must be relative and stay within the bundle root: {value!r}")
    return posix.as_posix()


def _identifier(value: str | None, *, field_name: str) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{field_name} must be non-empty when provided")
    return normalized


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 digest of the complete file contents."""

    digest = hashlib.sha256()
    source = Path(path)
    if not source.is_file():
        raise ValueError(f"cannot hash missing bundle file: {source}")
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class BundleArtifact:
    """One checkpoint or calibration artifact referenced by a manifest."""

    path: str
    sha256: str
    expert_type: PSFModality | str
    instance_id: str | None = None
    channel_id: str | None = None
    parent_checkpoint_hash: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", _relative_path(self.path))
        object.__setattr__(self, "sha256", _validate_sha256(self.sha256, field_name="sha256"))
        object.__setattr__(self, "expert_type", PSFModality.parse(self.expert_type))
        object.__setattr__(self, "instance_id", _identifier(self.instance_id, field_name="instance_id"))
        object.__setattr__(self, "channel_id", _identifier(self.channel_id, field_name="channel_id"))
        if self.parent_checkpoint_hash is not None:
            object.__setattr__(
                self,
                "parent_checkpoint_hash",
                _validate_sha256(self.parent_checkpoint_hash, field_name="parent_checkpoint_hash"),
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "sha256": self.sha256,
            "expert_type": self.expert_type.value,
            "instance_id": self.instance_id,
            "channel_id": self.channel_id,
            "parent_checkpoint_hash": self.parent_checkpoint_hash,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "BundleArtifact":
        required = {"path", "sha256", "expert_type"}
        missing = sorted(required.difference(value))
        if missing:
            raise ValueError(f"bundle artifact is missing fields: {', '.join(missing)}")
        return cls(
            path=str(value["path"]),
            sha256=str(value["sha256"]),
            expert_type=value["expert_type"],
            instance_id=value.get("instance_id"),
            channel_id=value.get("channel_id"),
            parent_checkpoint_hash=value.get("parent_checkpoint_hash"),
        )


@dataclass(frozen=True)
class ModelBundleManifest:
    """Self-contained manifest for a movable UnityPSF model package."""

    channel_layout: ChannelLayout
    prototypes: tuple[BundleArtifact, ...] = ()
    instances: tuple[BundleArtifact, ...] = ()
    schema_version: str = BUNDLE_SCHEMA_VERSION
    unity_psf_version: str = "0.4.0"
    modalities: tuple[PSFModality | str, ...] = tuple(PSFModality)

    def __post_init__(self) -> None:
        if self.schema_version != BUNDLE_SCHEMA_VERSION:
            raise ValueError(f"unsupported bundle schema {self.schema_version!r}")
        layout = self.channel_layout
        if not isinstance(layout, ChannelLayout):
            layout = ChannelLayout.from_value(layout)
            object.__setattr__(self, "channel_layout", layout)
        parsed_modalities = tuple(PSFModality.parse(item) for item in self.modalities)
        if not parsed_modalities:
            raise ValueError("bundle manifest requires at least one modality")
        if len(parsed_modalities) != len(set(parsed_modalities)):
            raise ValueError("bundle manifest modalities must be unique")
        object.__setattr__(self, "modalities", parsed_modalities)
        prototypes = tuple(self._coerce_artifact(item) for item in self.prototypes)
        instances = tuple(self._coerce_artifact(item) for item in self.instances)
        all_artifacts = prototypes + instances
        paths = tuple(item.path for item in all_artifacts)
        if len(paths) != len(set(paths)):
            raise ValueError("bundle artifact paths must be unique")
        prototype_hashes = {item.sha256 for item in prototypes}
        for artifact in prototypes:
            if artifact.instance_id is not None or artifact.channel_id is not None or artifact.parent_checkpoint_hash is not None:
                raise ValueError("prototype artifacts cannot carry instance binding fields")
            if artifact.expert_type not in parsed_modalities:
                raise ValueError("prototype artifact expert_type is not listed in modalities")
        for artifact in instances:
            if artifact.instance_id is None or artifact.channel_id is None:
                raise ValueError("instance artifacts require instance_id and channel_id")
            if artifact.parent_checkpoint_hash is None:
                raise ValueError("instance artifacts require parent_checkpoint_hash")
            if artifact.parent_checkpoint_hash not in prototype_hashes:
                raise ValueError("instance parent_checkpoint_hash does not match a prototype artifact")
            if artifact.expert_type not in parsed_modalities:
                raise ValueError("instance artifact expert_type is not listed in modalities")
            if artifact.channel_id not in layout.channel_ids:
                raise ValueError(f"instance channel_id {artifact.channel_id!r} is absent from channel_layout")
        object.__setattr__(self, "prototypes", prototypes)
        object.__setattr__(self, "instances", instances)

    @staticmethod
    def _coerce_artifact(value: BundleArtifact | Mapping[str, Any]) -> BundleArtifact:
        if isinstance(value, BundleArtifact):
            return value
        if not isinstance(value, Mapping):
            raise TypeError("bundle artifacts must be BundleArtifact instances or mappings")
        return BundleArtifact.from_dict(value)

    def to_dict(self) -> dict[str, Any]:
        layout: dict[str, Any] = {
            "channels": [
                {
                    "id": item.channel_id,
                    "crop": list(item.crop) if item.crop is not None else None,
                    "anchor_profile": item.anchor_profile,
                    "calibration_ref": item.calibration_ref,
                }
                for item in self.channel_layout.channels
            ]
        }
        if self.channel_layout.frame_size is not None:
            layout["frame_size"] = list(self.channel_layout.frame_size)
        return {
            "schema_version": self.schema_version,
            "unity_psf_version": self.unity_psf_version,
            "modalities": [item.value for item in self.modalities],
            "channel_layout": layout,
            "prototypes": [item.to_dict() for item in self.prototypes],
            "instances": [item.to_dict() for item in self.instances],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ModelBundleManifest":
        if not isinstance(value, Mapping):
            raise TypeError("bundle manifest must be a mapping")
        layout_value = value.get("channel_layout")
        if not isinstance(layout_value, Mapping):
            raise ValueError("bundle manifest requires channel_layout")
        return cls(
            schema_version=str(value.get("schema_version", "")),
            unity_psf_version=str(value.get("unity_psf_version", "0.4.0")),
            modalities=tuple(value.get("modalities", tuple(item.value for item in PSFModality))),
            channel_layout=ChannelLayout.from_value(layout_value),
            prototypes=tuple(value.get("prototypes", ())),
            instances=tuple(value.get("instances", ())),
        )


BundleManifest = ModelBundleManifest


def validate_bundle_manifest(
    manifest: ModelBundleManifest | Mapping[str, Any],
    *,
    bundle_root: str | Path | None = None,
) -> ModelBundleManifest:
    """Validate manifest structure and, when supplied, every referenced file."""

    normalized = manifest if isinstance(manifest, ModelBundleManifest) else ModelBundleManifest.from_dict(manifest)
    if bundle_root is None:
        return normalized
    root = Path(bundle_root).resolve()
    for artifact in normalized.prototypes + normalized.instances:
        artifact_path = (root / artifact.path).resolve()
        if not artifact_path.is_relative_to(root):
            raise ValueError(f"bundle artifact escapes bundle root: {artifact.path!r}")
        if not artifact_path.is_file():
            raise ValueError(f"bundle artifact file is missing: {artifact.path!r}")
        actual_hash = sha256_file(artifact_path)
        if actual_hash != artifact.sha256:
            raise ValueError(f"bundle artifact hash mismatch for {artifact.path!r}")
    return normalized


def save_bundle_manifest(
    path: str | Path,
    manifest: ModelBundleManifest,
    *,
    bundle_root: str | Path | None = None,
) -> Path:
    """Validate and write a YAML manifest beside the model package."""

    destination = Path(path)
    root = Path(bundle_root) if bundle_root is not None else destination.parent
    normalized = validate_bundle_manifest(manifest, bundle_root=root)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(yaml.safe_dump(normalized.to_dict(), sort_keys=False), encoding="utf-8")
    return destination


def load_bundle_manifest(path: str | Path) -> ModelBundleManifest:
    """Load a manifest and verify all artifacts relative to its manifest directory."""

    manifest_path = Path(path)
    payload = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("bundle manifest YAML must contain a mapping")
    return validate_bundle_manifest(ModelBundleManifest.from_dict(payload), bundle_root=manifest_path.parent)


__all__ = [
    "BUNDLE_SCHEMA_VERSION",
    "BundleArtifact",
    "BundleManifest",
    "ModelBundleManifest",
    "load_bundle_manifest",
    "save_bundle_manifest",
    "sha256_file",
    "validate_bundle_manifest",
]
