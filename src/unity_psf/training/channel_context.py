"""Per-channel physical state for high-fidelity localization training."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from unity_psf.contracts.modality import (
    ChannelLayout,
    ExpertInstanceSpec,
    InputFrameSpec,
    MeasurementChannelSpec,
)
from unity_psf.localization.conditioning import ConditioningProviderStore
from unity_psf.optics.profiles import (
    ASTIGMATISM_660NM_ANCHOR_PROFILE,
    AstigmatismAnchorProfile,
    resolve_astigmatism_anchor_profile,
)


def atomic_write_json(path: Path | str, payload: Mapping[str, Any]) -> Path:
    """Write JSON beside the target and publish it with an atomic replacement."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".tmp",
        dir=str(target.parent),
        text=True,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    except BaseException:
        try:
            temporary.unlink(missing_ok=True)
        finally:
            raise
    return target


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass
class ChannelTrainingContext:
    """All physical artifacts owned by one expert/channel training instance."""

    instance: ExpertInstanceSpec
    channel: MeasurementChannelSpec
    input_frame_spec: InputFrameSpec
    raw_crop: tuple[int, int, int, int] | None
    anchor_profile: str
    physical_state_path: Path
    condition_store: ConditioningProviderStore
    frame_size: tuple[int, int] | None = None
    peak_zmap_path: Path | None = None
    coefficient_map_path: Path | None = None
    initial_physical_state_hash: str | None = None
    latest_physical_state_hash: str | None = None
    _layout: Any = field(default=None, repr=False, compare=False)
    _coefficient_entries: dict[str, str] = field(default_factory=dict, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.instance.channel_id != self.channel.channel_id:
            raise ValueError("expert instance channel_id must match the context channel")
        providers = self.condition_store.snapshot()[1]
        if providers is not None and len(providers) > 1:
            raise ValueError("ChannelTrainingContext accepts exactly one physical condition provider")
        self.raw_crop = _normalize_crop(self.raw_crop)
        if self.frame_size is not None:
            self.frame_size = (int(self.frame_size[0]), int(self.frame_size[1]))
            if self.raw_crop is not None:
                x, y, width, height = self.raw_crop
                frame_height, frame_width = self.frame_size
                if x + width > frame_width or y + height > frame_height:
                    raise ValueError(
                        f"raw_crop {self.raw_crop!r} exceeds frame_size {self.frame_size!r}"
                    )
        self.physical_state_path = Path(self.physical_state_path)
        if self.peak_zmap_path is not None:
            self.peak_zmap_path = Path(self.peak_zmap_path)
        if self.coefficient_map_path is not None:
            self.coefficient_map_path = Path(self.coefficient_map_path)

    @classmethod
    def from_runtime_config(
        cls,
        runtime_config: Mapping[str, Any],
        *,
        layout: Any,
        raw_crop: tuple[int, int, int, int] | None = None,
        anchor_profile: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> "ChannelTrainingContext":
        instance = ExpertInstanceSpec.from_value(runtime_config["expert_instance"])  # type: ignore[arg-type]
        channel_layout = ChannelLayout.from_value(runtime_config["channel_layout"])  # type: ignore[arg-type]
        channel = channel_layout[instance.channel_id]
        frame_spec = InputFrameSpec.from_value(runtime_config["input_frame_spec"])  # type: ignore[arg-type]
        entries = _runtime_coeff_map_entries(runtime_config)
        normalized_entries = ()
        if entries:
            normalized_entries = (
                {"name": channel.channel_id, "coeff_maps_npz": str(_entry_path(entries[0]))},
            )
        store = ConditioningProviderStore.from_coeff_maps(normalized_entries) if normalized_entries else ConditioningProviderStore()
        selected_profile = resolve_astigmatism_anchor_profile(anchor_profile or channel.anchor_profile)
        coefficient_path = None
        coefficient_entries: dict[str, str] = {}
        if entries:
            item = entries[0]
            coefficient_path = _entry_path(item)
            coefficient_entries[channel.channel_id] = str(coefficient_path)
        metadata_value = metadata if metadata is not None else runtime_config.get("metadata")
        peak_zmap_path = _peak_path_from_metadata(metadata_value, channel.channel_id)
        # Each channel already owns a separate run directory, so the state filename
        # can remain stable across main and non-main instances.
        state_name = "current_physical_state.json"
        context = cls(
            instance=instance,
            channel=channel,
            input_frame_spec=frame_spec,
            raw_crop=channel.crop if raw_crop is None else raw_crop,
            anchor_profile=selected_profile.name,
            physical_state_path=Path(layout.metadata_dir) / state_name,
            condition_store=store,
            frame_size=channel_layout.frame_size,
            peak_zmap_path=peak_zmap_path,
            coefficient_map_path=coefficient_path,
            _layout=layout,
            _coefficient_entries=coefficient_entries,
        )
        return context

    def restore_physical_state(
        self,
        state: Mapping[str, Any] | None,
        coefficient_maps: tuple[Mapping[str, Any], ...] = (),
        *,
        initial_state_hash: str | None = None,
    ) -> None:
        """Restore the physical artifact referenced by a training checkpoint."""

        if not state and not coefficient_maps:
            self.write_physical_state(source="initial")
            return
        self.validate_physical_state_identity(state)
        entries = state.get("coeff_maps", ()) if isinstance(state, Mapping) else coefficient_maps
        if not entries and coefficient_maps:
            entries = coefficient_maps
        if not isinstance(entries, (list, tuple)):
            raise ValueError("checkpoint physical state coeff_maps must be a sequence")
        peak_path_value = state.get("peak_zmap_path") if isinstance(state, Mapping) else None
        state_schema = str(state.get("schema_version", "")) if isinstance(state, Mapping) else ""
        if peak_path_value is not None:
            peak_path = Path(str(peak_path_value)).resolve()
            if not peak_path.is_file():
                raise FileNotFoundError(peak_path)
            expected_peak_hash = state.get("peak_zmap_sha256") if isinstance(state, Mapping) else None
            if state_schema == "unitypsf.channel_physical_state.v1" and expected_peak_hash is None:
                raise ValueError("checkpoint physical state peak zmap hash is required")
            if expected_peak_hash is not None and str(expected_peak_hash) != sha256_file(peak_path):
                raise ValueError(f"peak zmap artifact hash mismatch: {peak_path}")
            self.peak_zmap_path = peak_path
        normalized_paths = []
        for item in entries:
            if not isinstance(item, Mapping) or item.get("coeff_maps_npz") is None:
                raise ValueError("checkpoint physical state coefficient entries are invalid")
            item_name = item.get("name")
            if item_name is not None and str(item_name) != self.channel.channel_id:
                raise ValueError("checkpoint physical state coefficient map channel_id does not match the context")
            path = Path(str(item["coeff_maps_npz"])).resolve()
            if not path.is_file():
                raise FileNotFoundError(path)
            expected_hash = item.get("sha256")
            if state_schema == "unitypsf.channel_physical_state.v1" and expected_hash is None:
                raise ValueError("checkpoint physical state coefficient map hash is required")
            if expected_hash is not None and str(expected_hash) != sha256_file(path):
                raise ValueError(f"physical artifact hash mismatch: {path}")
            normalized_paths.append(path)
        if len(normalized_paths) > 1:
            raise ValueError("checkpoint physical state contains multiple channel maps")
        if normalized_paths:
            self.update_coefficient_map(normalized_paths[0])
        persisted_version = state.get("condition_store_version") if isinstance(state, Mapping) else None
        if persisted_version is not None:
            self.condition_store.restore_version(int(persisted_version))
        if state:
            restored = dict(state)
            if normalized_paths:
                restored["coeff_maps"] = [
                    {
                        "name": self.channel.channel_id,
                        "coeff_maps_npz": str(normalized_paths[0]),
                        "sha256": sha256_file(normalized_paths[0]),
                    }
                ]
            atomic_write_json(self.physical_state_path, restored)
            state_hash = _payload_hash(restored)
            self.latest_physical_state_hash = state_hash
            if self.initial_physical_state_hash is None:
                self.initial_physical_state_hash = initial_state_hash or self._manifest_initial_hash() or state_hash
            self._update_manifest(restored, state_hash)
        else:
            self.write_physical_state(source="resume")

    def validate_physical_state_identity(self, state: Mapping[str, Any] | None) -> None:
        """Reject a state payload produced for another channel instance."""

        state_instance = state.get("expert_instance") if isinstance(state, Mapping) else None
        if not isinstance(state_instance, Mapping):
            return
        state_expert_type = str(state_instance.get("expert_type", self.instance.expert_type.value))
        if state_expert_type != self.instance.expert_type.value:
            raise ValueError("checkpoint physical state expert_type does not match the context")
        if str(state_instance.get("channel_id", self.channel.channel_id)) != self.channel.channel_id:
            raise ValueError("checkpoint physical state channel_id does not match the context")
        if str(state_instance.get("instance_id", self.instance.instance_id)) != self.instance.instance_id:
            raise ValueError("checkpoint physical state instance_id does not match the context")

    def _manifest_initial_hash(self) -> str | None:
        if self._layout is None:
            return None
        path = Path(self._layout.metadata_dir) / "run_manifest.json"
        if not path.exists():
            return None
        value = json.loads(path.read_text(encoding="utf-8")).get("initial_physical_state_hash")
        return None if value is None else str(value)

    @property
    def anchor(self) -> AstigmatismAnchorProfile:
        return resolve_astigmatism_anchor_profile(self.anchor_profile)

    @property
    def condition_store_version(self) -> int:
        return self.condition_store.version

    def update_coefficient_map(self, path: Path | str) -> None:
        coefficient_path = Path(path).resolve()
        if not coefficient_path.is_file():
            raise FileNotFoundError(coefficient_path)
        self.coefficient_map_path = coefficient_path
        self._coefficient_entries = {self.channel.channel_id: str(coefficient_path)}
        self.condition_store.update_from_coeff_maps(
            ({"name": self.channel.channel_id, "coeff_maps_npz": str(coefficient_path)},)
        )

    def write_physical_state(
        self,
        *,
        source: str,
        epoch: int | None = None,
        global_step: int | None = None,
        condition_store_version: int | None = None,
    ) -> str:
        entries = tuple(
            {
                "name": str(name),
                "coeff_maps_npz": str(path),
                **({"sha256": sha256_file(path)} if Path(path).is_file() else {}),
            }
            for name, path in sorted(self._coefficient_entries.items())
        )
        payload = {
            "schema_version": "unitypsf.channel_physical_state.v1",
            "source": str(source),
            "epoch": None if epoch is None else int(epoch),
            "global_step": None if global_step is None else int(global_step),
            "condition_store_version": self.condition_store.version
            if condition_store_version is None
            else int(condition_store_version),
            "expert_instance": {
                "expert_type": self.instance.expert_type.value,
                "instance_id": self.instance.instance_id,
                "channel_id": self.instance.channel_id,
            },
            "raw_crop": None if self.raw_crop is None else list(self.raw_crop),
            "anchor_profile": self.anchor_profile,
            "peak_zmap_path": None if self.peak_zmap_path is None else str(self.peak_zmap_path),
            "peak_zmap_sha256": (
                None
                if self.peak_zmap_path is None or not self.peak_zmap_path.is_file()
                else sha256_file(self.peak_zmap_path)
            ),
            "coeff_maps": list(entries),
        }
        state_hash = _payload_hash(payload)
        atomic_write_json(self.physical_state_path, payload)
        if self.initial_physical_state_hash is None:
            self.initial_physical_state_hash = state_hash
        self.latest_physical_state_hash = state_hash
        self._update_manifest(payload, state_hash)
        return str(self.physical_state_path)

    def manifest_fields(self) -> dict[str, Any]:
        fields: dict[str, Any] = {
            "physical_state_path": str(self.physical_state_path),
            "initial_physical_state_hash": self.initial_physical_state_hash,
            "latest_physical_state_hash": self.latest_physical_state_hash,
            "peak_zmap_path": None if self.peak_zmap_path is None else str(self.peak_zmap_path),
            "peak_zmap_sha256": (
                None
                if self.peak_zmap_path is None or not self.peak_zmap_path.is_file()
                else sha256_file(self.peak_zmap_path)
            ),
        }
        if self.coefficient_map_path is not None:
            fields["current_physical_coeff_maps"] = [
                {"name": self.channel.channel_id, "coeff_maps_npz": str(self.coefficient_map_path)}
            ]
        return fields

    def _update_manifest(self, payload: Mapping[str, Any], state_hash: str) -> None:
        if self._layout is None:
            return
        manifest_path = Path(self._layout.metadata_dir) / "run_manifest.json"
        if not manifest_path.exists():
            return
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest.update(
            {
                "current_physical_state": dict(payload),
                "current_physical_coeff_maps": list(payload["coeff_maps"]),
                "physical_state_path": str(self.physical_state_path),
                "initial_physical_state_hash": self.initial_physical_state_hash,
                "latest_physical_state_hash": state_hash,
            }
        )
        atomic_write_json(manifest_path, manifest)


def build_channel_training_context(
    runtime_config: Mapping[str, Any],
    *,
    layout: Any,
    raw_crop: tuple[int, int, int, int] | None = None,
    anchor_profile: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> ChannelTrainingContext:
    return ChannelTrainingContext.from_runtime_config(
        runtime_config,
        layout=layout,
        raw_crop=raw_crop,
        anchor_profile=anchor_profile,
        metadata=metadata,
    )


def _runtime_coeff_map_entries(runtime_config: Mapping[str, Any]) -> tuple[Mapping[str, str], ...]:
    provider = runtime_config.get("batch_provider")
    if not isinstance(provider, Mapping):
        return ()
    params = provider.get("params")
    if not isinstance(params, Mapping):
        return ()
    entries = params.get("dual_domain_coeff_maps")
    if not isinstance(entries, (list, tuple)):
        return ()
    if len(entries) > 1:
        raise ValueError("single-channel physical context accepts at most one coefficient map")
    return tuple(item for item in entries if isinstance(item, Mapping))


def _entry_path(entry: Mapping[str, Any]) -> Path:
    value = entry.get("coeff_maps_npz") or entry.get("alternating_coeff_maps_npz") or entry.get("path")
    if value is None:
        raise ValueError("coefficient map entry requires coeff_maps_npz, alternating_coeff_maps_npz, or path")
    return Path(str(value)).resolve()


def _peak_path_from_metadata(value: object, channel_id: str) -> Path | None:
    if not isinstance(value, Mapping):
        return None
    peak = value.get("peak_zmap_bootstrap")
    if not isinstance(peak, Mapping):
        return None
    domains = peak.get("domains")
    if not isinstance(domains, Mapping):
        return None
    item = domains.get(channel_id)
    if not isinstance(item, Mapping) or item.get("zmap_path") is None:
        return None
    return Path(str(item["zmap_path"])).resolve()


def _payload_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(dict(payload), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _normalize_crop(value: tuple[int, int, int, int] | None) -> tuple[int, int, int, int] | None:
    if value is None:
        return None
    if len(value) != 4 or any(int(item) != item for item in value):
        raise ValueError("raw_crop must contain four integers")
    x, y, width, height = (int(item) for item in value)
    if min(x, y) < 0 or min(width, height) <= 0:
        raise ValueError("raw_crop coordinates must be non-negative and dimensions must be positive")
    return x, y, width, height


__all__ = [
    "ASTIGMATISM_660NM_ANCHOR_PROFILE",
    "ChannelTrainingContext",
    "atomic_write_json",
    "build_channel_training_context",
    "sha256_file",
]
