"""Canonical resolution of normalized localization output scales."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping


@dataclass(frozen=True)
class ScaleConflict:
    field: str
    selected_source: str
    selected_value: float
    other_source: str
    other_value: float


@dataclass(frozen=True)
class PhysicalScaleSpec:
    photon_scale: float | None
    z_scale: float | None
    z_scale_nm: float | None
    photon_source: str | None
    z_source: str | None
    z_unit: str | None
    conflicts: tuple[ScaleConflict, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "photon_scale": self.photon_scale,
            "z_scale": self.z_scale,
            "z_scale_nm": self.z_scale_nm,
            "photon_source": self.photon_source,
            "z_source": self.z_source,
            "z_unit": self.z_unit,
            "conflicts": [
                {
                    "field": conflict.field,
                    "selected_source": conflict.selected_source,
                    "selected_value": conflict.selected_value,
                    "other_source": conflict.other_source,
                    "other_value": conflict.other_value,
                }
                for conflict in self.conflicts
            ],
        }


def resolve_physical_scales(train_cfg: Mapping[str, Any]) -> PhysicalScaleSpec:
    """Resolve photon and z scales from one training configuration.

    Explicit loss parameters are authoritative. ``train.scaling`` is the
    canonical legacy contract, while normalization and online-generation
    values are compatibility fallbacks only. Conflicting values are retained
    in the result so callers can expose the configuration problem instead of
    silently selecting a different scale.
    """

    loss_cfg = _mapping(train_cfg.get("loss"))
    loss_params = _mapping(loss_cfg.get("params"))
    if not loss_params and loss_cfg:
        loss_params = loss_cfg
    scaling_cfg = _mapping(train_cfg.get("scaling"))
    normalization_cfg = _mapping(train_cfg.get("normalization"))
    online_cfg = _mapping(train_cfg.get("online_generation"))
    train_params_cfg = _mapping(train_cfg.get("train_params"))

    photon_candidates = _candidates(
        (
            ("loss.params.photon_scale", loss_params.get("photon_scale")),
            ("scaling.photon_max", scaling_cfg.get("photon_max")),
            ("scaling.phot_max", scaling_cfg.get("phot_max")),
            ("scaling.ph_scale", scaling_cfg.get("ph_scale")),
            ("normalization.photon_scale", normalization_cfg.get("photon_scale")),
            ("online_generation.photon_scale", online_cfg.get("photon_scale")),
        ),
        field="photon_scale",
    )
    z_candidates = _candidates(
        (
            ("loss.params.z_scale", loss_params.get("z_scale")),
            ("scaling.z_max", scaling_cfg.get("z_max")),
            ("train_params.z_max", train_params_cfg.get("z_max")),
            ("online_generation.z_scale", online_cfg.get("z_scale")),
        ),
        field="z_scale",
    )
    photon_scale, photon_source, photon_conflicts = _select("photon_scale", photon_candidates)
    z_scale, z_source, z_conflicts = _select("z_scale", z_candidates)
    z_unit = _resolve_z_unit(
        loss_params,
        scaling_cfg,
        train_params_cfg,
        online_cfg,
        z_scale=z_scale,
    )
    z_scale_nm = None if z_scale is None else _to_nm(z_scale, z_unit)
    return PhysicalScaleSpec(
        photon_scale=photon_scale,
        z_scale=z_scale,
        z_scale_nm=z_scale_nm,
        photon_source=photon_source,
        z_source=z_source,
        z_unit=z_unit,
        conflicts=tuple(photon_conflicts + z_conflicts),
    )


def resolve_target_contract(
    train_cfg: Mapping[str, Any],
    *,
    target_order: str | None = None,
    z_activation: str | None = None,
    scales: PhysicalScaleSpec | None = None,
):
    from unity_psf.localization.smlm_targets import SMLMTargetContract

    loss_cfg = _mapping(train_cfg.get("loss"))
    loss_params = _mapping(loss_cfg.get("params"))
    online_cfg = _mapping(train_cfg.get("online_generation"))
    if scales is None:
        scales = resolve_physical_scales(train_cfg)
    resolved_order = target_order or loss_params.get("target_order") or online_cfg.get("pxyz_target_order", "legacy_iwae")
    resolved_activation = z_activation or loss_params.get("z_activation") or online_cfg.get("z_activation", "tanh")
    return SMLMTargetContract(
        target_order=str(resolved_order),
        photon_scale=scales.photon_scale,
        z_scale=scales.z_scale,
        z_scale_nm=scales.z_scale_nm,
        z_unit=scales.z_unit,
        z_activation=str(resolved_activation),
    ).validate()


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _candidates(
    values: tuple[tuple[str, Any], ...],
    *,
    field: str,
) -> list[tuple[str, float]]:
    result: list[tuple[str, float]] = []
    for source, value in values:
        if value is None:
            continue
        result.append((source, _validate_scale(value, field=field, source=source)))
    return result


def _select(
    field: str,
    candidates: list[tuple[str, float]],
) -> tuple[float | None, str | None, list[ScaleConflict]]:
    if not candidates:
        return None, None, []
    selected_source, selected_value = candidates[0]
    conflicts = [
        ScaleConflict(
            field=field,
            selected_source=selected_source,
            selected_value=selected_value,
            other_source=source,
            other_value=value,
        )
        for source, value in candidates[1:]
        if not math.isclose(value, selected_value, rel_tol=1e-9, abs_tol=1e-9)
    ]
    return selected_value, selected_source, conflicts


def _validate_scale(value: Any, *, field: str, source: str) -> float:
    try:
        resolved = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} from {source} must be a finite positive number") from exc
    if not math.isfinite(resolved) or resolved <= 0.0:
        raise ValueError(f"{field} from {source} must be a finite positive number")
    return resolved


def _resolve_z_unit(
    loss_params: Mapping[str, Any],
    scaling_cfg: Mapping[str, Any],
    train_params_cfg: Mapping[str, Any],
    online_cfg: Mapping[str, Any],
    *,
    z_scale: float | None,
) -> str | None:
    if z_scale is None:
        return None
    for config, source in (
        (loss_params, "loss.params.z_scale_unit"),
        (scaling_cfg, "scaling.z_scale_unit"),
        (train_params_cfg, "train_params.z_scale_unit"),
        (online_cfg, "online_generation.z_scale_unit"),
    ):
        if "z_scale_unit" in config:
            return _normalize_z_unit(config["z_scale_unit"], source=source)
    return "um" if z_scale <= 10.0 else "nm"


def _normalize_z_unit(value: Any, *, source: str) -> str:
    unit = str(value).strip().lower().replace("μ", "u")
    if unit in {"um", "micrometer", "micrometers", "micron", "microns"}:
        return "um"
    if unit in {"nm", "nanometer", "nanometers"}:
        return "nm"
    raise ValueError(f"{source} must be 'um' or 'nm', got {value!r}")


def _to_nm(value: float, unit: str) -> float:
    return value * 1000.0 if unit == "um" else value


def z_scale_to_nm(value: float, *, unit: str | None = None) -> float:
    """Convert a legacy z scale to nanometers using the canonical unit rule."""
    resolved = _normalize_z_unit(unit, source="z_scale_unit") if unit is not None else ("um" if value <= 10.0 else "nm")
    return _to_nm(_validate_scale(value, field="z_scale", source="z_scale"), resolved)


__all__ = [
    "PhysicalScaleSpec",
    "ScaleConflict",
    "resolve_physical_scales",
    "resolve_target_contract",
    "z_scale_to_nm",
]
