"""Input and output contracts for multimodal PSF localization."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from numbers import Integral
from typing import Mapping, Sequence
import warnings

import torch


class PSFModality(str, Enum):
    """Supported PSF families handled by the UnityPSF MoE."""

    EMITTER_2D = "emitter_2d"
    ASTIGMATISM = "astigmatism"
    DOUBLE_HELIX = "double_helix"

    @classmethod
    def parse(cls, value: "PSFModality | str") -> "PSFModality":
        if isinstance(value, cls):
            return value
        normalized = str(value).strip().lower().replace("-", "_").replace(" ", "_")
        aliases = {
            "2d": cls.EMITTER_2D,
            "emitter": cls.EMITTER_2D,
            "emitter2d": cls.EMITTER_2D,
            "astig": cls.ASTIGMATISM,
            "dh": cls.DOUBLE_HELIX,
            "doublehelix": cls.DOUBLE_HELIX,
        }
        try:
            return cls(normalized)
        except ValueError:
            if normalized in aliases:
                return aliases[normalized]
            choices = ", ".join(item.value for item in cls)
            raise ValueError(f"unknown PSF modality {value!r}; expected one of: {choices}") from None

    @property
    def auxiliary_keys(self) -> tuple[str, ...]:
        return {
            self.EMITTER_2D: (),
            self.ASTIGMATISM: ("astigmatism_width",),
            self.DOUBLE_HELIX: ("lobe_angle", "lobe_separation"),
        }[self]


Modality = PSFModality


def _required_identifier(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value.strip()


def _positive_integer(value: object, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{field_name} must be a positive integer")
    normalized = int(value)
    if normalized <= 0:
        raise ValueError(f"{field_name} must be positive")
    return normalized


def _normalize_crop(value: object) -> tuple[int, int, int, int] | None:
    if value is None:
        return None
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or len(value) != 4:
        raise ValueError("crop must be [x, y, width, height]")
    if any(isinstance(item, bool) or not isinstance(item, Integral) for item in value):
        raise ValueError("crop must contain integer coordinates and dimensions")
    x, y, width, height = (int(item) for item in value)
    if x < 0 or y < 0 or width <= 0 or height <= 0:
        raise ValueError("crop coordinates must be non-negative and dimensions must be positive")
    return x, y, width, height


def _normalize_frame_size(value: object) -> tuple[int, int] | None:
    if value is None:
        return None
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or len(value) != 2:
        raise ValueError("frame_size must be [height, width]")
    if any(isinstance(item, bool) or not isinstance(item, Integral) for item in value):
        raise ValueError("frame_size must contain positive integers")
    height, width = (int(item) for item in value)
    if height <= 0 or width <= 0:
        raise ValueError("frame_size must contain positive integers")
    return height, width


_UNSET = object()


@dataclass(frozen=True, init=False)
class InputFrameSpec:
    """Describes temporal frames supplied to one model invocation."""

    input_frame_channels: int = 3
    frame_order: str | tuple[str, ...] = "temporal"

    def __init__(
        self,
        input_frame_channels: int | object = _UNSET,
        frame_order: str | Sequence[str] | object = _UNSET,
        *,
        channels: int | object = _UNSET,
        order: str | Sequence[str] | object = _UNSET,
    ) -> None:
        if input_frame_channels is _UNSET:
            selected_channels = 3 if channels is _UNSET else channels
        elif channels is _UNSET:
            selected_channels = input_frame_channels
        else:
            if input_frame_channels != channels:
                raise ValueError("input_frame_channels and channels must agree")
            selected_channels = input_frame_channels
        if frame_order is _UNSET:
            selected_order = "temporal" if order is _UNSET else order
        elif order is _UNSET:
            selected_order = frame_order
        else:
            if frame_order != order:
                raise ValueError("frame_order and order must agree")
            selected_order = frame_order
        object.__setattr__(self, "input_frame_channels", selected_channels)
        object.__setattr__(self, "frame_order", selected_order)
        self.__post_init__()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "input_frame_channels",
            _positive_integer(self.input_frame_channels, field_name="input_frame_channels"),
        )
        if isinstance(self.frame_order, str):
            order = self.frame_order.strip()
            if not order:
                raise ValueError("frame_order must be non-empty")
            normalized_order: str | tuple[str, ...] = order
        else:
            if isinstance(self.frame_order, (bytes,)) or not isinstance(self.frame_order, Sequence):
                raise ValueError("frame_order must be a non-empty string or sequence")
            normalized_items = tuple(_required_identifier(item, field_name="frame_order item") for item in self.frame_order)
            if not normalized_items:
                raise ValueError("frame_order must be non-empty")
            if len(normalized_items) != self.input_frame_channels:
                raise ValueError("frame_order length must match input_frame_channels")
            normalized_order = normalized_items
        object.__setattr__(self, "frame_order", normalized_order)

    @property
    def channels(self) -> int:
        """Compatibility alias; this is temporal input width, not measurement count."""

        return self.input_frame_channels

    @property
    def order(self) -> str | tuple[str, ...]:
        return self.frame_order

    @classmethod
    def from_value(cls, value: "InputFrameSpec | Mapping[str, object] | int") -> "InputFrameSpec":
        if isinstance(value, cls):
            return value
        if isinstance(value, Integral) and not isinstance(value, bool):
            return cls(input_frame_channels=int(value))
        if not isinstance(value, Mapping):
            raise TypeError("input frame spec must be an InputFrameSpec, mapping, or integer")
        has_new_key = "input_frame_channels" in value
        has_legacy_key = "channels" in value
        if has_new_key and has_legacy_key and value["input_frame_channels"] != value["channels"]:
            raise ValueError("input_frame_channels and legacy channels config values must agree")
        if has_new_key:
            channels = value["input_frame_channels"]
        elif has_legacy_key:
            warnings.warn(
                "config key 'channels' is deprecated; use 'input_frame_channels' for temporal frames",
                DeprecationWarning,
                stacklevel=2,
            )
            channels = value["channels"]
        else:
            raise ValueError("input frame config requires 'input_frame_channels'")
        has_new_order = "frame_order" in value
        has_legacy_order = "order" in value
        if has_new_order and has_legacy_order and value["frame_order"] != value["order"]:
            raise ValueError("frame_order and legacy order config values must agree")
        order = value["frame_order"] if has_new_order else value.get("order", "temporal")
        return cls(input_frame_channels=channels, frame_order=order)  # type: ignore[arg-type]


@dataclass(frozen=True)
class MeasurementChannelSpec:
    """Describes one physical measurement crop and its calibration references."""

    channel_id: str
    crop: tuple[int, int, int, int] | None = None
    anchor_profile: str | None = None
    calibration_ref: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "channel_id", _required_identifier(self.channel_id, field_name="channel_id"))
        object.__setattr__(self, "crop", _normalize_crop(self.crop))
        for name in ("anchor_profile", "calibration_ref"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _required_identifier(value, field_name=name))

    @classmethod
    def from_value(cls, value: "MeasurementChannelSpec | Mapping[str, object] | str") -> "MeasurementChannelSpec":
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            return cls(channel_id=value)
        if not isinstance(value, Mapping):
            raise TypeError("measurement channel must be a MeasurementChannelSpec, mapping, or string")
        channel_id = value.get("channel_id", value.get("id"))
        if channel_id is None:
            raise ValueError("measurement channel config requires 'channel_id' or 'id'")
        return cls(
            channel_id=channel_id,  # type: ignore[arg-type]
            crop=value.get("crop"),  # type: ignore[arg-type]
            anchor_profile=value.get("anchor_profile"),  # type: ignore[arg-type]
            calibration_ref=value.get("calibration_ref", value.get("calibration")),  # type: ignore[arg-type]
        )


@dataclass(frozen=True, init=False)
class ChannelLayout:
    """Ordered, validated measurement channels for one acquisition."""

    channels: tuple[MeasurementChannelSpec, ...]
    frame_size: tuple[int, int] | None = None

    def __init__(
        self,
        channels: Sequence[MeasurementChannelSpec | Mapping[str, object] | str] | MeasurementChannelSpec | object = _UNSET,
        frame_size: Sequence[int] | None = None,
        *,
        measurement_channels: Sequence[MeasurementChannelSpec | Mapping[str, object] | str]
        | MeasurementChannelSpec
        | object = _UNSET,
    ) -> None:
        if channels is _UNSET:
            selected_channels = measurement_channels if measurement_channels is not _UNSET else ()
        elif measurement_channels is _UNSET:
            selected_channels = channels
        else:
            if channels != measurement_channels:
                raise ValueError("channels and measurement_channels must describe the same layout")
            selected_channels = channels
        object.__setattr__(self, "channels", selected_channels)
        object.__setattr__(self, "frame_size", frame_size)
        self.__post_init__()

    def __post_init__(self) -> None:
        if isinstance(self.channels, MeasurementChannelSpec):
            normalized_channels = (self.channels,)
        else:
            if isinstance(self.channels, (str, bytes)) or not isinstance(self.channels, Sequence):
                raise TypeError("channels must be a non-empty sequence")
            try:
                normalized_channels = tuple(MeasurementChannelSpec.from_value(item) for item in self.channels)
            except TypeError as exc:
                raise TypeError("channels must be a non-empty sequence") from exc
        if not normalized_channels:
            raise ValueError("channel layout requires at least one measurement channel")
        identifiers = tuple(item.channel_id for item in normalized_channels)
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("measurement channel IDs must be unique")
        normalized_frame_size = _normalize_frame_size(self.frame_size)
        if normalized_frame_size is None and any(item.crop is not None for item in normalized_channels):
            raise ValueError("frame_size is required to validate cropped measurement channels")
        if normalized_frame_size is not None:
            frame_height, frame_width = normalized_frame_size
            for item in normalized_channels:
                if item.crop is None:
                    continue
                x, y, width, height = item.crop
                if x + width > frame_width or y + height > frame_height:
                    raise ValueError(
                        f"crop for channel {item.channel_id!r} is out of frame bounds {normalized_frame_size}"
                    )
        object.__setattr__(self, "channels", normalized_channels)
        object.__setattr__(self, "frame_size", normalized_frame_size)

    @property
    def measurement_channels(self) -> tuple[MeasurementChannelSpec, ...]:
        return self.channels

    @property
    def channel_ids(self) -> tuple[str, ...]:
        return tuple(item.channel_id for item in self.channels)

    @property
    def input_instances(self) -> int:
        return len(self.channels)

    def __getitem__(self, channel_id: str) -> MeasurementChannelSpec:
        for channel in self.channels:
            if channel.channel_id == channel_id:
                return channel
        raise KeyError(channel_id)

    @classmethod
    def from_value(
        cls,
        value: "ChannelLayout | Mapping[str, object] | Sequence[MeasurementChannelSpec | Mapping[str, object] | str]",
        *,
        frame_size: Sequence[int] | None = None,
    ) -> "ChannelLayout":
        if isinstance(value, cls):
            if frame_size is not None and value.frame_size != _normalize_frame_size(frame_size):
                raise ValueError("frame_size conflicts with existing channel layout")
            return value
        layout_frame_size = frame_size
        if isinstance(value, Mapping):
            has_channels = "channels" in value
            has_measurement_channels = "measurement_channels" in value
            if has_channels and has_measurement_channels and value["channels"] != value["measurement_channels"]:
                raise ValueError("channels and measurement_channels config values must agree")
            raw_channels = value["channels"] if has_channels else value.get("measurement_channels")
            if raw_channels is None:
                raise ValueError("channel layout config requires 'channels'")
            if layout_frame_size is None:
                layout_frame_size = value.get("frame_size", value.get("image_size"))  # type: ignore[assignment]
        else:
            raw_channels = value
        if isinstance(raw_channels, (str, bytes)) or not isinstance(raw_channels, Sequence):
            raise TypeError("channel layout channels must be a sequence")
        return cls(tuple(MeasurementChannelSpec.from_value(item) for item in raw_channels), layout_frame_size)  # type: ignore[arg-type]


@dataclass(frozen=True)
class ExpertInstanceSpec:
    """Binds one canonical PSF expert type to one runtime channel instance."""

    expert_type: PSFModality | str
    instance_id: str
    channel_id: str
    prototype_ref: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "expert_type", PSFModality.parse(self.expert_type))
        object.__setattr__(self, "instance_id", _required_identifier(self.instance_id, field_name="instance_id"))
        object.__setattr__(self, "channel_id", _required_identifier(self.channel_id, field_name="channel_id"))
        if self.prototype_ref is not None:
            object.__setattr__(self, "prototype_ref", _required_identifier(self.prototype_ref, field_name="prototype_ref"))

    @property
    def modality(self) -> PSFModality:
        return self.expert_type  # type: ignore[return-value]

    @classmethod
    def from_value(cls, value: "ExpertInstanceSpec | Mapping[str, object]") -> "ExpertInstanceSpec":
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise TypeError("expert instance must be an ExpertInstanceSpec or mapping")
        expert_type = value.get("expert_type", value.get("modality"))
        if expert_type is None:
            raise ValueError("expert instance config requires 'expert_type'")
        instance_id = value.get("instance_id")
        channel_id = value.get("channel_id")
        if instance_id is None or channel_id is None:
            raise ValueError("expert instance config requires 'instance_id' and 'channel_id'")
        return cls(
            expert_type=expert_type,  # type: ignore[arg-type]
            instance_id=instance_id,  # type: ignore[arg-type]
            channel_id=channel_id,  # type: ignore[arg-type]
            prototype_ref=value.get("prototype_ref", value.get("prototype_checkpoint")),  # type: ignore[arg-type]
        )


@dataclass(frozen=True)
class ModalityBatch:
    """Validated per-sample modality labels used by a router."""

    values: tuple[PSFModality, ...]

    @classmethod
    def from_value(
        cls,
        value: PSFModality | str | Sequence[PSFModality | str],
        *,
        batch_size: int,
    ) -> "ModalityBatch":
        size = int(batch_size)
        if size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if isinstance(value, (PSFModality, str)):
            parsed = (PSFModality.parse(value),) * size
        else:
            parsed = tuple(PSFModality.parse(item) for item in value)
            if len(parsed) != size:
                raise ValueError(f"modality count {len(parsed)} does not match batch size {size}")
        return cls(parsed)

    def indices(self, order: Sequence[PSFModality]) -> torch.Tensor:
        positions = {item: index for index, item in enumerate(order)}
        try:
            return torch.tensor([positions[item] for item in self.values], dtype=torch.long)
        except KeyError as exc:
            raise ValueError(f"modality {exc.args[0]!r} is not present in router order") from None


@dataclass(frozen=True)
class PSFExpertOutput:
    """Common dense output contract emitted by every PSF expert."""

    detection_logits: torch.Tensor
    xy_offset: torch.Tensor
    z: torch.Tensor
    photons: torch.Tensor
    auxiliary: Mapping[str, torch.Tensor] = field(default_factory=dict)

    def validate(self, *, batch_size: int | None = None) -> "PSFExpertOutput":
        tensors = {
            "detection_logits": self.detection_logits,
            "xy_offset": self.xy_offset,
            "z": self.z,
            "photons": self.photons,
        }
        for name, tensor in tensors.items():
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"{name} must be a torch.Tensor, got {type(tensor)!r}")
            if tensor.ndim != 3 and name != "xy_offset":
                raise ValueError(f"{name} must have shape (N,H,W), got {tuple(tensor.shape)}")
        if self.xy_offset.ndim != 4 or self.xy_offset.shape[1] != 2:
            raise ValueError(f"xy_offset must have shape (N,2,H,W), got {tuple(self.xy_offset.shape)}")
        spatial = self.detection_logits.shape
        if self.xy_offset.shape[0] != spatial[0] or self.xy_offset.shape[-2:] != spatial[-2:]:
            raise ValueError("xy_offset batch/spatial shape does not match detection_logits")
        for name in ("z", "photons"):
            tensor = tensors[name]
            if tensor.shape != spatial:
                raise ValueError(f"{name} shape {tuple(tensor.shape)} does not match {tuple(spatial)}")
        if batch_size is not None and int(spatial[0]) != int(batch_size):
            raise ValueError(f"output batch {spatial[0]} does not match expected {batch_size}")
        for name, tensor in self.auxiliary.items():
            if not isinstance(name, str) or not isinstance(tensor, torch.Tensor):
                raise TypeError("auxiliary outputs must map string names to torch.Tensor values")
            if tensor.ndim < 3 or tensor.shape[0] != spatial[0] or tensor.shape[-2:] != spatial[-2:]:
                raise ValueError(f"auxiliary output {name!r} has incompatible shape {tuple(tensor.shape)}")
        return self


__all__ = [
    "ChannelLayout",
    "ExpertInstanceSpec",
    "InputFrameSpec",
    "MeasurementChannelSpec",
    "Modality",
    "ModalityBatch",
    "PSFExpertOutput",
    "PSFModality",
]
