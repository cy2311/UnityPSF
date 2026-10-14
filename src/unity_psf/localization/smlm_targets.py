from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace

import torch


V03_PXYZ_TARGET_ORDER = ("x_px", "y_px", "z", "photons")
V03_PXYZ_TARGET_UNITS = ("pixel", "pixel", "physical_z", "photons")
LEGACY_IWAE_PXYZ_TARGET_ORDER = ("photons", "x_px", "y_px", "z")
CANONICAL_PXYZ_TARGET_ORDER = ("x_px", "y_px", "z_nm", "photons")
MODEL_PXYZ_OUTPUT_ORDER = ("photons", "x_px", "y_px", "z_normalized")


@dataclass(frozen=True)
class SMLMTargetConvention:
    photon_scale: float | None = 1.0
    z_scale: float | None = None
    z_activation: str = "tanh"


@dataclass(frozen=True)
class SMLMTargetContract:
    target_order: str
    photon_scale: float | None
    z_scale: float | None
    z_scale_nm: float | None
    z_unit: str | None
    z_activation: str

    def validate(self) -> "SMLMTargetContract":
        normalized_order = normalize_pxyz_target_order(self.target_order)
        if self.photon_scale is not None and float(self.photon_scale) <= 0.0:
            raise ValueError("photon_scale must be positive or None")
        if self.z_scale is not None and float(self.z_scale) <= 0.0:
            raise ValueError("z_scale must be positive or None")
        if self.z_scale_nm is None and self.z_scale is not None:
            raise ValueError("z_scale_nm is required when z_scale is configured")
        if self.z_scale_nm is not None and float(self.z_scale_nm) <= 0.0:
            raise ValueError("z_scale_nm must be positive or None")
        if self.z_unit is not None and str(self.z_unit).lower() not in {"um", "nm"}:
            raise ValueError("z_unit must be 'um', 'nm', or None")
        if str(self.z_activation).strip().lower() not in {"tanh", "sigmoid"}:
            raise ValueError("z_activation must be 'tanh' or 'sigmoid'")
        return replace(self, target_order=normalized_order, z_activation=str(self.z_activation).strip().lower())

    def with_target_order(self, target_order: str) -> "SMLMTargetContract":
        return replace(self, target_order=normalize_pxyz_target_order(target_order)).validate()

    def to_canonical_physical(self, targets: torch.Tensor) -> torch.Tensor:
        self.validate()
        _validate_target_tensor(targets)
        if self.target_order == "legacy_iwae":
            photons = targets[..., 0] * (1.0 if self.photon_scale is None else float(self.photon_scale))
            z_nm = targets[..., 3] * (1.0 if self.z_scale_nm is None else float(self.z_scale_nm))
            return torch.stack((targets[..., 1], targets[..., 2], z_nm, photons), dim=-1)
        z_nm = targets[..., 2]
        if self.z_unit == "um":
            z_nm = z_nm * 1000.0
        photons = targets[..., 3]
        return torch.stack((targets[..., 0], targets[..., 1], z_nm, photons), dim=-1)

    def from_canonical_physical(self, targets_nm: torch.Tensor) -> torch.Tensor:
        self.validate()
        _validate_target_tensor(targets_nm)
        if self.target_order == "legacy_iwae":
            photons = targets_nm[..., 3] / (1.0 if self.photon_scale is None else float(self.photon_scale))
            z = targets_nm[..., 2] / (1.0 if self.z_scale_nm is None else float(self.z_scale_nm))
            return torch.stack((photons, targets_nm[..., 0], targets_nm[..., 1], z), dim=-1)
        z = targets_nm[..., 2]
        if self.z_unit == "um":
            z = z / 1000.0
        return torch.stack((targets_nm[..., 0], targets_nm[..., 1], z, targets_nm[..., 3]), dim=-1)

    def as_dict(self) -> dict[str, object]:
        normalized = self.validate()
        return {
            "target_order": normalized.target_order,
            "target_order_tuple": pxyz_target_order_tuple(normalized.target_order),
            "canonical_order": CANONICAL_PXYZ_TARGET_ORDER,
            "model_output_order": MODEL_PXYZ_OUTPUT_ORDER,
            "canonical_units": ("pixel", "pixel", "nm", "photons"),
            "photon_scale": normalized.photon_scale,
            "z_scale": normalized.z_scale,
            "z_scale_nm": normalized.z_scale_nm,
            "z_unit": normalized.z_unit,
            "z_activation": normalized.z_activation,
        }


def _validate_target_tensor(targets: torch.Tensor) -> None:
    if targets.shape[-1] != 4:
        raise ValueError(f"target tensor must have last dimension 4, got {tuple(targets.shape)}")


def normalize_pxyz_target_order(value: str) -> str:
    key = str(value or "legacy_iwae").strip().lower()
    if key in {"legacy_iwae", "iwae", "old", "phot_xyz", "phot,x,y,z", "photons_x_y_z"}:
        return "legacy_iwae"
    if key in {"v03", "xyzph", "x,y,z,phot", "x_y_z_photons"}:
        return "v03"
    raise ValueError(f"unsupported pxyz target_order: {value!r}")


def pxyz_target_order_tuple(value: str) -> tuple[str, ...]:
    if normalize_pxyz_target_order(value) == "legacy_iwae":
        return LEGACY_IWAE_PXYZ_TARGET_ORDER
    return V03_PXYZ_TARGET_ORDER


def v03_pxyz_to_legacy_iwae(
    pxyz_targets: torch.Tensor,
    *,
    photon_scale: float | None = None,
    z_scale: float | None = None,
) -> torch.Tensor:
    _validate_target_tensor(pxyz_targets)
    if photon_scale is not None and float(photon_scale) <= 0.0:
        raise ValueError("photon_scale must be positive or None")
    if z_scale is not None and float(z_scale) <= 0.0:
        raise ValueError("z_scale must be positive or None")
    photons = pxyz_targets[..., 3]
    z = pxyz_targets[..., 2]
    if photon_scale is not None:
        photons = photons / float(photon_scale)
    if z_scale is not None:
        z = z / float(z_scale)
    return torch.stack((photons, pxyz_targets[..., 0], pxyz_targets[..., 1], z), dim=-1)


def target_pixel_indices(pxyz_targets: torch.Tensor, *, height: int, width: int) -> tuple[torch.Tensor, torch.Tensor]:
    if pxyz_targets.ndim != 2 or int(pxyz_targets.shape[1]) != 4:
        raise ValueError(f"pxyz_targets must have shape (M,4), got {tuple(pxyz_targets.shape)}")
    x = pxyz_targets[:, 0].round().long().clamp(0, int(width) - 1)
    y = pxyz_targets[:, 1].round().long().clamp(0, int(height) - 1)
    return x, y


def legacy_iwae_pxyz_to_v03(pxyz_targets: torch.Tensor) -> torch.Tensor:
    if pxyz_targets.shape[-1] != 4:
        raise ValueError(f"legacy pxyz targets must have last dimension 4, got {tuple(pxyz_targets.shape)}")
    return torch.stack(
        [
            pxyz_targets[..., 1],
            pxyz_targets[..., 2],
            pxyz_targets[..., 3],
            pxyz_targets[..., 0],
        ],
        dim=-1,
    )


def legacy_iwae_target_process_to_v03(
    pxyz_targets: torch.Tensor,
    *,
    disable_attr: int | tuple[int, ...] | list[int] | None = None,
    phot_max: float | None = None,
    z_max: float | None = None,
) -> torch.Tensor:
    if pxyz_targets.shape[-1] != 4:
        raise ValueError(f"legacy pxyz targets must have last dimension 4, got {tuple(pxyz_targets.shape)}")
    if phot_max is not None and float(phot_max) <= 0.0:
        raise ValueError("phot_max must be positive or None")
    if z_max is not None and float(z_max) <= 0.0:
        raise ValueError("z_max must be positive or None")
    processed = pxyz_targets.clone()
    if disable_attr is not None:
        attrs = (int(disable_attr),) if isinstance(disable_attr, int) else tuple(int(item) for item in disable_attr)
        processed[..., list(attrs)] = 0.0
    if phot_max is not None:
        processed[..., 0] = processed[..., 0] / float(phot_max)
    if z_max is not None:
        processed[..., 3] = processed[..., 3] / float(z_max)
    return legacy_iwae_pxyz_to_v03(processed)


def absolute_pxyz_to_local_targets(
    pxyz_targets: torch.Tensor,
    *,
    x: torch.Tensor,
    y: torch.Tensor,
    convention: SMLMTargetConvention,
) -> torch.Tensor:
    if pxyz_targets.ndim != 2 or int(pxyz_targets.shape[1]) != 4:
        raise ValueError(f"pxyz_targets must have shape (M,4), got {tuple(pxyz_targets.shape)}")
    if convention.z_scale is not None and float(convention.z_scale) <= 0.0:
        raise ValueError("z_scale must be positive or None")
    if convention.photon_scale is not None and float(convention.photon_scale) <= 0.0:
        raise ValueError("photon_scale must be positive or None")
    z_activation = str(convention.z_activation).strip().lower()
    local = pxyz_targets.clone()
    local[:, 0] = pxyz_targets[:, 0] - x.to(dtype=pxyz_targets.dtype, device=pxyz_targets.device)
    local[:, 1] = pxyz_targets[:, 1] - y.to(dtype=pxyz_targets.dtype, device=pxyz_targets.device)
    if convention.z_scale is not None:
        z = pxyz_targets[:, 2] / float(convention.z_scale)
    else:
        z = pxyz_targets[:, 2]
    if z_activation == "tanh":
        local[:, 2] = z.clamp(-1.0, 1.0)
    elif z_activation == "sigmoid":
        local[:, 2] = (0.5 * (z + 1.0)).clamp(0.0, 1.0)
    else:
        raise ValueError("z_activation must be 'tanh' or 'sigmoid'")
    if convention.photon_scale is not None:
        local[:, 3] = pxyz_targets[:, 3] / float(convention.photon_scale)
    return local


__all__ = [
    "LEGACY_IWAE_PXYZ_TARGET_ORDER",
    "CANONICAL_PXYZ_TARGET_ORDER",
    "MODEL_PXYZ_OUTPUT_ORDER",
    "SMLMTargetContract",
    "SMLMTargetConvention",
    "V03_PXYZ_TARGET_ORDER",
    "V03_PXYZ_TARGET_UNITS",
    "absolute_pxyz_to_local_targets",
    "legacy_iwae_target_process_to_v03",
    "legacy_iwae_pxyz_to_v03",
    "normalize_pxyz_target_order",
    "pxyz_target_order_tuple",
    "target_pixel_indices",
    "v03_pxyz_to_legacy_iwae",
]
