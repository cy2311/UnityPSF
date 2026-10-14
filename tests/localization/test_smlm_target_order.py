from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from unity_psf.localization.data.online_targets import finalize_pxyz_targets
from unity_psf.localization.losses import ActiveSMLMGMMTargetAdapter
from unity_psf.localization.smlm_targets import (
    LEGACY_IWAE_PXYZ_TARGET_ORDER,
    V03_PXYZ_TARGET_ORDER,
    normalize_pxyz_target_order,
    pxyz_target_order_tuple,
    SMLMTargetContract,
    v03_pxyz_to_legacy_iwae,
)
from unity_psf.localization.scale_resolver import resolve_target_contract


@pytest.mark.parametrize("value", ["legacy_iwae", "IWAE", "old", "phot,x,y,z", "photons_x_y_z"])
def test_normalize_pxyz_target_order_accepts_legacy_aliases(value: str) -> None:
    assert normalize_pxyz_target_order(value) == "legacy_iwae"
    assert pxyz_target_order_tuple(value) == LEGACY_IWAE_PXYZ_TARGET_ORDER


@pytest.mark.parametrize("value", ["v03", "XYZPH", "x,y,z,phot", "x_y_z_photons"])
def test_normalize_pxyz_target_order_accepts_v03_aliases(value: str) -> None:
    assert normalize_pxyz_target_order(value) == "v03"
    assert pxyz_target_order_tuple(value) == V03_PXYZ_TARGET_ORDER


def test_normalize_pxyz_target_order_rejects_unknown_order() -> None:
    with pytest.raises(ValueError, match="unsupported pxyz target_order"):
        normalize_pxyz_target_order("xy_phot_z")


def test_v03_pxyz_to_legacy_iwae_reorders_and_scales_targets() -> None:
    target = torch.tensor([[2.0, 3.0, 400.0, 1200.0]])

    converted = v03_pxyz_to_legacy_iwae(target, photon_scale=6000.0, z_scale=800.0)

    torch.testing.assert_close(converted, torch.tensor([[0.2, 2.0, 3.0, 0.5]]))


def test_loss_and_online_targets_share_canonical_v03_conversion() -> None:
    target = torch.tensor([[2.0, 3.0, 400.0, 1200.0]])
    expected = v03_pxyz_to_legacy_iwae(target, photon_scale=6000.0, z_scale=800.0)
    adapter = ActiveSMLMGMMTargetAdapter(
        target_order="v03",
        photon_scale=6000.0,
        z_scale=800.0,
    )
    config = SimpleNamespace(
        pxyz_target_order="legacy_iwae",
        photon_scale=6000.0,
        z_scale=800.0,
    )

    online_target, online_order = finalize_pxyz_targets(config, target)

    torch.testing.assert_close(adapter.to_gmm_order(target), expected)
    torch.testing.assert_close(online_target, expected)
    assert online_order == LEGACY_IWAE_PXYZ_TARGET_ORDER


def test_target_contract_round_trips_legacy_and_v03_without_unit_drift() -> None:
    contract = SMLMTargetContract(
        target_order="legacy_iwae",
        photon_scale=6_000.0,
        z_scale=0.8,
        z_scale_nm=800.0,
        z_unit="um",
        z_activation="tanh",
    )
    canonical_nm = torch.tensor([[2.0, 3.0, 400.0, 1_200.0]])

    legacy = contract.from_canonical_physical(canonical_nm)
    restored = contract.to_canonical_physical(legacy)

    torch.testing.assert_close(legacy, torch.tensor([[0.2, 2.0, 3.0, 0.5]]))
    torch.testing.assert_close(restored, canonical_nm)
    assert contract.as_dict()["canonical_order"] == ("x_px", "y_px", "z_nm", "photons")


def test_target_contract_rejects_incomplete_physical_scale() -> None:
    with pytest.raises(ValueError, match="z_scale_nm"):
        SMLMTargetContract(
            target_order="v03",
            photon_scale=6_000.0,
            z_scale=0.8,
            z_scale_nm=None,
            z_unit="um",
            z_activation="tanh",
        ).validate()


def test_resolved_training_contract_carries_target_semantics() -> None:
    contract = resolve_target_contract(
        {
            "loss": {"params": {"target_order": "v03", "z_activation": "sigmoid", "z_scale": 0.6}},
            "scaling": {"photon_max": 31_000.0},
        }
    )

    assert contract.target_order == "v03"
    assert contract.z_activation == "sigmoid"
    assert contract.z_scale_nm == 600.0
    assert contract.as_dict()["model_output_order"] == ("photons", "x_px", "y_px", "z_normalized")
