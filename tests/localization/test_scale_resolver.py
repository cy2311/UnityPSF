from __future__ import annotations

import math

import pytest

from unity_psf.localization.scale_resolver import resolve_physical_scales


def test_explicit_loss_scales_win_and_report_conflicts() -> None:
    resolved = resolve_physical_scales(
        {
            "loss": {"params": {"photon_scale": 12_000.0, "z_scale": 0.8}},
            "scaling": {"photon_max": 31_000.0, "z_max": 0.6},
            "normalization": {"photon_scale": 20_000.0},
        }
    )

    assert resolved.photon_scale == 12_000.0
    assert resolved.z_scale == 0.8
    assert resolved.z_scale_nm == 800.0
    assert resolved.photon_source == "loss.params.photon_scale"
    assert resolved.z_source == "loss.params.z_scale"
    assert {item.field for item in resolved.conflicts} == {"photon_scale", "z_scale"}


def test_scaling_is_canonical_before_legacy_normalization_and_online_values() -> None:
    resolved = resolve_physical_scales(
        {
            "scaling": {"photon_max": 31_000.0, "z_max": 0.6},
            "normalization": {"photon_scale": 20_000.0},
            "online_generation": {"photon_scale": 11_000.0, "z_scale": 1.2},
        }
    )

    assert resolved.photon_scale == 31_000.0
    assert resolved.z_scale == 0.6
    assert resolved.photon_source == "scaling.photon_max"
    assert resolved.z_source == "scaling.z_max"


@pytest.mark.parametrize(
    ("value", "unit", "expected_nm"),
    [(0.6, "um", 600.0), (600.0, "nm", 600.0), (0.6, None, 600.0), (600.0, None, 600.0)],
)
def test_z_scale_is_resolved_to_nm_with_explicit_or_legacy_unit(
    value: float, unit: str | None, expected_nm: float
) -> None:
    train_cfg = {"loss": {"params": {"z_scale": value}}}
    if unit is not None:
        train_cfg["loss"]["params"]["z_scale_unit"] = unit

    resolved = resolve_physical_scales(train_cfg)

    assert resolved.z_scale_nm == expected_nm
    assert resolved.z_unit == (unit or ("um" if value <= 10.0 else "nm"))


def test_legacy_train_params_z_max_is_supported() -> None:
    resolved = resolve_physical_scales({"train_params": {"z_max": 0.75}})

    assert resolved.z_scale == 0.75
    assert resolved.z_scale_nm == 750.0
    assert resolved.z_source == "train_params.z_max"


@pytest.mark.parametrize("field", ["photon_scale", "z_scale"])
@pytest.mark.parametrize("value", [0.0, -1.0, math.nan, math.inf])
def test_invalid_scales_are_rejected(field: str, value: float) -> None:
    with pytest.raises(ValueError, match=field):
        resolve_physical_scales({"loss": {"params": {field: value}}})


def test_invalid_z_unit_is_rejected() -> None:
    with pytest.raises(ValueError, match="z_scale_unit"):
        resolve_physical_scales({"loss": {"params": {"z_scale": 1.0, "z_scale_unit": "pixels"}}})
