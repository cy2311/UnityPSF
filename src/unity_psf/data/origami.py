"""Read-only acquisition manifest for the external Origami 2D TIFF dataset."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import random
import re
from typing import Any


_TIFF_SUFFIXES = (".tif", ".tiff")
_PART_SUFFIX = re.compile(r"_(?:part|chunk)(?:_|-)?\d+$", re.IGNORECASE)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _acquisition_group(filename: str) -> str:
    normalized = filename
    for suffix in (".ome.tiff", ".ome.tif", ".tiff", ".tif"):
        if normalized.lower().endswith(suffix):
            normalized = normalized[: -len(suffix)]
            break
    return _PART_SUFFIX.sub("", normalized)


@dataclass(frozen=True)
class OrigamiFileRecord:
    relative_path: str
    acquisition_group: str
    size_bytes: int
    mtime_ns: int
    sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "relative_path": self.relative_path,
            "acquisition_group": self.acquisition_group,
            "size_bytes": self.size_bytes,
            "mtime_ns": self.mtime_ns,
            "sha256": self.sha256,
        }


@dataclass(frozen=True)
class OrigamiManifest:
    logical_root: str
    resolved_root: str
    files: tuple[OrigamiFileRecord, ...]
    schema_version: str = "unitypsf.origami_manifest.v1"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "logical_root": self.logical_root,
            "resolved_root": self.resolved_root,
            "files": [item.to_dict() for item in self.files],
        }


def build_origami_manifest(root: str | Path) -> OrigamiManifest:
    """Hash external TIFFs in place without copying them into the project."""

    logical_root = Path(root)
    if not logical_root.exists():
        raise ValueError(f"Origami dataset root does not exist: {logical_root}")
    resolved_root = logical_root.resolve()
    paths = sorted(
        path for path in resolved_root.rglob("*")
        if path.is_file() and path.name.lower().endswith(_TIFF_SUFFIXES)
    )
    if not paths:
        raise ValueError(f"Origami dataset contains no TIFF files: {logical_root}")
    records = []
    for path in paths:
        stat = path.stat()
        records.append(
            OrigamiFileRecord(
                relative_path=path.relative_to(resolved_root).as_posix(),
                acquisition_group=_acquisition_group(path.name),
                size_bytes=int(stat.st_size),
                mtime_ns=int(stat.st_mtime_ns),
                sha256=_sha256_file(path),
            )
        )
    return OrigamiManifest(str(logical_root), str(resolved_root), tuple(records))


def split_origami_acquisitions(
    manifest: OrigamiManifest,
    *,
    seed: int,
) -> dict[str, tuple[str, ...]]:
    """Split whole acquisition groups so frames from one spool cannot leak."""

    groups = sorted({item.acquisition_group for item in manifest.files})
    if len(groups) < 3:
        raise ValueError("Origami split requires at least three acquisition groups")
    random.Random(int(seed)).shuffle(groups)
    validation_count = max(1, round(len(groups) * 0.15))
    test_count = max(1, round(len(groups) * 0.15))
    if validation_count + test_count >= len(groups):
        validation_count = test_count = 1
    train_count = len(groups) - validation_count - test_count
    return {
        "train": tuple(groups[:train_count]),
        "validation": tuple(groups[train_count : train_count + validation_count]),
        "test": tuple(groups[train_count + validation_count :]),
    }


__all__ = [
    "OrigamiFileRecord",
    "OrigamiManifest",
    "build_origami_manifest",
    "split_origami_acquisitions",
]
