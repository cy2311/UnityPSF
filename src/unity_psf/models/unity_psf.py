"""Top-level model API for modality- and channel-routed UnityPSF inference."""

from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn

from unity_psf.contracts.joint_checkpoint import (
    JointCheckpointMetadata,
    JointExpertKey,
    JointExpertState,
    validate_joint_checkpoint,
)
from unity_psf.contracts.modality import MeasurementChannelSpec, PSFModality
from unity_psf.contracts.modality_joint_checkpoint import (
    LEGACY_JOINT_CHECKPOINT_SCHEMA_VERSION,
    MODALITY_JOINT_CHECKPOINT_SCHEMA_VERSION,
    ModalityExpertState,
    validate_modality_joint_checkpoint,
)
from unity_psf.localization.smlm_output import (
    SMLMOutput,
    SMLMOutputChannels,
    decode_smlm_output,
)

from .psf_moe.experts.astigmatism import AstigmatismExpert
from .psf_moe.experts.emitter_2d import Emitter2DExpert
from .psf_moe.router import InstanceRouter, ModalityRouter


@dataclass(frozen=True)
class UnityLocalizationResult:
    """Common localization output plus explicit modality semantics."""

    raw: torch.Tensor
    decoded: SMLMOutput
    modality: PSFModality
    channel_id: str
    z_valid: bool


class UnityPSF(nn.Module):
    """One sparse model containing exactly one complete network per PSF modality."""

    def __init__(
        self,
        experts: Mapping[JointExpertKey | PSFModality | str, nn.Module],
        *,
        channel_states: Mapping[str, Mapping[str, Mapping[str, Any]]] | None = None,
        physical_states: Mapping[str, Mapping[str, Any]] | None = None,
        calibrations: Mapping[str, Mapping[str, Any]] | None = None,
        checkpoint_hash: str | None = None,
        load_mode: str = "eager",
        target_device: str | torch.device = "cpu",
    ) -> None:
        super().__init__()
        if not experts:
            raise ValueError("UnityPSF requires at least one expert")
        legacy_keys = tuple(
            isinstance(raw_key, JointExpertKey) or ":" in str(raw_key)
            for raw_key in experts
        )
        if any(legacy_keys) and not all(legacy_keys):
            raise ValueError("cannot mix modality keys with legacy modality:channel keys")
        self._legacy_v1 = all(legacy_keys)
        normalized: dict[str, nn.Module] = {}
        if self._legacy_v1:
            for raw_key, expert in experts.items():
                key = JointExpertKey.parse(raw_key)
                self._register_expert(normalized, key.storage_key, expert)
            self.router: InstanceRouter | ModalityRouter = InstanceRouter(tuple(normalized))
            self.channel_states: dict[str, dict[str, dict[str, Any]]] = {}
        else:
            for raw_key, expert in experts.items():
                modality = PSFModality.parse(raw_key)
                self._register_expert(normalized, modality.value, expert)
            self.router = ModalityRouter(tuple(normalized))
            self.channel_states = self._normalize_channel_states(channel_states, tuple(normalized))
        selected_mode = str(load_mode).strip().lower()
        if selected_mode not in {"eager", "lazy"}:
            raise ValueError("load_mode must be 'eager' or 'lazy'")
        self.experts = nn.ModuleDict(normalized)
        self.physical_states = {key: dict(value) for key, value in (physical_states or {}).items()}
        self.calibrations = {key: dict(value) for key, value in (calibrations or {}).items()}
        if not self._legacy_v1 and (self.physical_states or self.calibrations):
            raise ValueError("modality-routed UnityPSF stores physical state inside channel_states")
        self.checkpoint_hash = checkpoint_hash
        self.load_mode = selected_mode
        self.target_device = torch.device(target_device)
        self._activation_counts: dict[str, int] = {}
        if self.load_mode == "eager":
            self.experts.to(self.target_device)

    @staticmethod
    def _register_expert(registry: dict[str, nn.Module], key: str, expert: nn.Module) -> None:
        if key in registry:
            raise ValueError(f"duplicate expert key {key!r}")
        if not isinstance(expert, nn.Module):
            raise TypeError("expert registry values must be torch.nn.Module instances")
        registry[key] = expert

    @staticmethod
    def _normalize_channel_states(
        states: Mapping[str, Mapping[str, Mapping[str, Any]]] | None,
        modalities: tuple[str, ...],
    ) -> dict[str, dict[str, dict[str, Any]]]:
        if not isinstance(states, Mapping):
            raise ValueError("modality-routed UnityPSF requires channel_states for every expert")
        normalized: dict[str, dict[str, dict[str, Any]]] = {}
        for raw_modality, raw_channels in states.items():
            modality = PSFModality.parse(raw_modality).value
            if modality in normalized or not isinstance(raw_channels, Mapping) or not raw_channels:
                raise ValueError(f"channel_states for {modality!r} must be a non-empty mapping")
            channels: dict[str, dict[str, Any]] = {}
            for raw_channel, raw_state in raw_channels.items():
                channel_id = MeasurementChannelSpec.from_value(raw_channel).channel_id
                if channel_id in channels or not isinstance(raw_state, Mapping):
                    raise ValueError(f"invalid or duplicate channel state {modality}:{channel_id}")
                state = dict(raw_state)
                for field_name in ("physical_state", "calibration", "provenance"):
                    value = state.get(field_name, {})
                    if not isinstance(value, Mapping):
                        raise TypeError(f"{field_name} for {modality}:{channel_id} must be a mapping")
                    state[field_name] = copy.deepcopy(dict(value))
                channels[channel_id] = state
            normalized[modality] = channels
        if set(normalized) != set(modalities):
            raise ValueError("channel_states must exactly match the modality expert registry")
        return normalized

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        *,
        device: str | torch.device = "cpu",
        load_mode: str = "eager",
    ) -> "UnityPSF":
        """Materialize every registered expert from one joint checkpoint file."""

        checkpoint_path = Path(path)
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if not isinstance(payload, Mapping):
            raise ValueError("joint checkpoint file must contain a mapping")
        schema = payload.get("checkpoint_schema")
        if schema == MODALITY_JOINT_CHECKPOINT_SCHEMA_VERSION:
            validate_modality_joint_checkpoint(payload)
            return cls._from_modality_payload(
                checkpoint_path,
                payload,
                device=device,
                load_mode=load_mode,
            )
        if schema != LEGACY_JOINT_CHECKPOINT_SCHEMA_VERSION:
            raise ValueError(f"unsupported UnityPSF joint checkpoint schema {schema!r}")
        validate_joint_checkpoint(payload)
        JointCheckpointMetadata.from_dict(payload["metadata"])
        experts: dict[str, nn.Module] = {}
        physical_states: dict[str, Mapping[str, Any]] = {}
        calibrations: dict[str, Mapping[str, Any]] = {}
        for storage_key, value in payload["experts"].items():
            state = JointExpertState.from_dict(storage_key, value)
            expert = cls._build_expert(state.key.modality, state.model_config)
            expert.load_state_dict(state.model_state_dict, strict=True)
            experts[storage_key] = expert
            physical_states[storage_key] = state.physical_state
            calibrations[storage_key] = state.calibration
        digest = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
        return cls(
            experts,
            physical_states=physical_states,
            calibrations=calibrations,
            checkpoint_hash=digest,
            load_mode=load_mode,
            target_device=device,
        )

    @classmethod
    def _from_modality_payload(
        cls,
        checkpoint_path: Path,
        payload: Mapping[str, Any],
        *,
        device: str | torch.device,
        load_mode: str,
    ) -> "UnityPSF":
        experts: dict[str, nn.Module] = {}
        for modality, value in payload["experts"].items():
            state = ModalityExpertState.from_payload(
                modality,
                value,
                payload["channel_states"][modality],
            )
            expert = cls._build_expert(state.modality, state.model_config)
            expert.load_state_dict(state.model_state_dict, strict=True)
            experts[state.modality.value] = expert
        digest = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
        return cls(
            experts,
            channel_states=payload["channel_states"],
            checkpoint_hash=digest,
            load_mode=load_mode,
            target_device=device,
        )

    @staticmethod
    def _build_expert(
        modality: PSFModality | str,
        model_config: Mapping[str, Any],
    ) -> nn.Module:
        parsed = PSFModality.parse(modality)
        if parsed is PSFModality.EMITTER_2D:
            return Emitter2DExpert(**dict(model_config))
        if parsed is PSFModality.ASTIGMATISM:
            return AstigmatismExpert(**dict(model_config))
        raise ValueError(f"unsupported trained expert modality {parsed.value!r}")

    def _selected_expert(self, storage_key: str) -> nn.Module:
        expert = self.experts[storage_key]
        if self.load_mode == "lazy":
            expert.to(self.target_device)
        return expert

    @staticmethod
    def _channel_id(value: str) -> str:
        return MeasurementChannelSpec.from_value(value).channel_id

    def _resolve(self, modality: PSFModality | str, channel_id: str) -> tuple[PSFModality, str, str]:
        normalized_channel = self._channel_id(channel_id)
        if self._legacy_v1:
            key = self.router.resolve(modality, normalized_channel)
            return key.modality, key.channel_id, key.storage_key
        parsed_modality = self.router.resolve(modality)
        self._channel_context(parsed_modality, normalized_channel)
        return parsed_modality, normalized_channel, parsed_modality.value

    def _channel_context(self, modality: PSFModality | str, channel_id: str) -> dict[str, Any]:
        parsed = PSFModality.parse(modality)
        normalized_channel = self._channel_id(channel_id)
        if self._legacy_v1:
            storage_key = JointExpertKey(parsed, normalized_channel).storage_key
            if storage_key not in self.experts:
                raise ValueError(f"unsupported channel {normalized_channel!r} for {parsed.value!r}")
            return {
                "physical_state": self.physical_states.get(storage_key, {}),
                "calibration": self.calibrations.get(storage_key, {}),
                "provenance": {},
            }
        channels = self.channel_states.get(parsed.value, {})
        if normalized_channel not in channels:
            available = ", ".join(sorted(channels))
            raise ValueError(
                f"unsupported channel {normalized_channel!r} for {parsed.value!r}; "
                f"available channels: {available}"
            )
        return channels[normalized_channel]

    def channel_state(self, modality: PSFModality | str, channel_id: str) -> dict[str, Any]:
        """Return a defensive copy of one channel's physical and calibration state."""

        return copy.deepcopy(self._channel_context(modality, channel_id))

    def forward(
        self,
        images: torch.Tensor,
        *,
        modality: PSFModality | str,
        channel_id: str,
        conditions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not isinstance(images, torch.Tensor) or images.ndim != 4:
            raise ValueError("images must have shape (N,C,H,W)")
        parsed_modality, _, storage_key = self._resolve(modality, channel_id)
        expert = self._selected_expert(storage_key)
        selected_images = images.to(self.target_device)
        if parsed_modality is PSFModality.ASTIGMATISM:
            if conditions is None:
                raise ValueError("astigmatism expert requires FiLM conditions")
            if conditions.ndim != 2 or conditions.shape[0] != images.shape[0]:
                raise ValueError("condition batch size must match image batch size")
            output = expert(selected_images, conditions.to(self.target_device))
        elif parsed_modality is PSFModality.EMITTER_2D:
            if conditions is not None:
                if conditions.ndim != 2 or conditions.shape[0] != images.shape[0]:
                    raise ValueError("condition batch size must match image batch size")
                output = expert(selected_images, conditions.to(self.target_device))
            else:
                output = expert(selected_images)
        else:
            raise ValueError(f"unsupported trained expert modality {parsed_modality.value!r}")
        if not isinstance(output, torch.Tensor) or output.ndim != 4:
            raise RuntimeError("expert must return a dense (N,10,H,W) tensor")
        if output.shape[1] != SMLMOutputChannels.count:
            raise RuntimeError(f"expert returned {output.shape[1]} channels; expected 10")
        if parsed_modality is PSFModality.EMITTER_2D:
            output = output.clone()
            output[:, SMLMOutputChannels.z_mu] = 0.0
        self._activation_counts[storage_key] = self._activation_counts.get(storage_key, 0) + 1
        return output

    def localize(
        self,
        images: torch.Tensor,
        *,
        modality: PSFModality | str,
        channel_id: str,
        conditions: torch.Tensor | None = None,
    ) -> UnityLocalizationResult:
        parsed_modality = PSFModality.parse(modality)
        raw = self.forward(
            images,
            modality=parsed_modality,
            channel_id=channel_id,
            conditions=conditions,
        )
        return UnityLocalizationResult(
            raw=raw,
            decoded=decode_smlm_output(raw),
            modality=parsed_modality,
            channel_id=self._channel_id(channel_id),
            z_valid=parsed_modality is not PSFModality.EMITTER_2D,
        )

    def activation_audit(self, *, reset: bool = False) -> dict[str, int]:
        counts = dict(sorted(self._activation_counts.items()))
        if reset:
            self._activation_counts.clear()
        return counts

    def describe(self) -> dict[str, Any]:
        if self._legacy_v1:
            present_modalities = {JointExpertKey.parse(key).modality for key in self.experts}
        else:
            present_modalities = {PSFModality.parse(key) for key in self.experts}
        modalities = [item.value for item in PSFModality if item in present_modalities]
        parameter_counts = {
            key: sum(parameter.numel() for parameter in expert.parameters())
            for key, expert in self.experts.items()
        }
        description = {
            "model_family": "UnityPSF",
            "supported_modalities": modalities,
            "parameter_counts": parameter_counts,
            "checkpoint_hash": self.checkpoint_hash,
            "load_mode": self.load_mode,
        }
        if self._legacy_v1:
            description["instances"] = sorted(self.experts.keys())
            description["legacy_checkpoint"] = True
        else:
            description["experts"] = sorted(self.experts.keys())
            description["supported_channels"] = {
                modality: sorted(channels) for modality, channels in sorted(self.channel_states.items())
            }
            description["legacy_checkpoint"] = False
        return description


__all__ = ["UnityLocalizationResult", "UnityPSF"]
