from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch

from unity_psf.localization.conditioning import (
    ConditioningProviderStore,
    FullResZernikeConditioning,
    build_astigmatism_anchor_conditioning_maps,
)
from unity_psf.localization.data.online import (
    OnlineBatchProviderConfig,
    _default_condition_provider,
    _simulator_config,
    build_online_batch_provider,
)
from unity_psf.localization.simulator import _VectorEmitterRenderer


def _config(profile: str, **changes: object) -> OnlineBatchProviderConfig:
    return replace(
        OnlineBatchProviderConfig(
            batch_size=1,
            height=8,
            width=8,
            conditioning_profile=profile,
            simulation_backend="native",
            npupil=8,
            vector_psf_size=5,
            vector_batch_size=1,
        ),
        **changes,
    )


def _psf(config: OnlineBatchProviderConfig, provider: FullResZernikeConditioning) -> torch.Tensor:
    simulator = _simulator_config(
        config, seed=1, field_origin=(0, 0), frames_per_sample=1,
        condition_provider=provider,
    )
    renderer = _VectorEmitterRenderer(simulator)
    patches, _, _ = renderer._render_patches(
        xs=torch.tensor([4.5]), ys=torch.tensor([4.5]),
        z_um=torch.tensor([0.0]), photons=torch.tensor([1.0]),
    )
    return patches


def test_zero_and_astigmatism_anchor_render_distinct_deterministic_psfs() -> None:
    zero = _default_condition_provider(_config("zero"))
    astig = _default_condition_provider(_config("astigmatism_660nm"))
    index = astig.mode_order.index((2, 2))

    assert torch.count_nonzero(zero.full_maps_nm) == 0
    assert torch.all(astig.full_maps_nm[index] == 100.0)
    zero_psf = _psf(_config("zero"), zero)
    astig_psf = _psf(_config("astigmatism_660nm"), astig)
    assert not torch.allclose(zero_psf, astig_psf)
    assert torch.equal(astig_psf, _psf(_config("astigmatism_660nm"), astig))


@pytest.mark.parametrize(
    ("profile", "mode", "left_nm", "right_nm"),
    [
        ("zero", (2, 0), 0.0, 120.0),
        ("astigmatism_660nm", (2, 2), 100.0, 120.0),
    ],
)
def test_left_right_absolute_maps_render_their_own_coefficients_once(
    tmp_path, profile: str, mode: tuple[int, int], left_nm: float, right_nm: float,
) -> None:
    anchor = _default_condition_provider(_config(profile))
    index = anchor.mode_order.index(mode)
    entries = []
    for channel, amplitude in (("left", left_nm), ("right", right_nm)):
        maps = anchor.full_maps_nm.clone()
        maps[index].fill_(amplitude)
        path = tmp_path / f"{channel}.npz"
        np.savez(path, zernike_maps_nm=maps.numpy(), mode_order=np.asarray(anchor.mode_order))
        entries.append({"name": channel, "coeff_maps_npz": str(path)})

    config = _config(profile, dual_domain_coeff_maps=tuple(entries))
    build_online_batch_provider(config)
    providers = ConditioningProviderStore.from_coeff_maps(entries).snapshot()[1]
    assert providers is not None
    assert [name for name, _ in providers] == ["left", "right"]
    assert torch.all(providers[0][1].full_maps_nm[index] == left_nm)
    assert torch.all(providers[1][1].full_maps_nm[index] == right_nm)

    anchor_psf = _psf(config, anchor)
    left_psf = _psf(config, providers[0][1])
    right_psf = _psf(config, providers[1][1])
    assert torch.equal(left_psf, anchor_psf)
    assert not torch.allclose(right_psf, anchor_psf)

    anchor_image = build_online_batch_provider(_config(profile))(1)[0].inputs.model_input
    left_image = build_online_batch_provider(
        _config(profile, dual_domain_coeff_maps=(entries[0],), domain_count=1)
    )(1)[0].inputs.model_input
    right_image = build_online_batch_provider(
        _config(profile, dual_domain_coeff_maps=(entries[1],), domain_count=1)
    )(1)[0].inputs.model_input
    assert torch.equal(left_image, anchor_image)
    assert not torch.allclose(right_image, anchor_image)


@pytest.mark.parametrize("profile", ["astigmatism_660nm", "zero"])
def test_absolute_map_and_fixed_anchor_cannot_be_combined(tmp_path, profile: str) -> None:
    maps = build_astigmatism_anchor_conditioning_maps(width=8, height=8)
    path = tmp_path / "map.npz"
    np.savez(path, zernike_maps_nm=maps.full_maps_nm.numpy(), mode_order=np.asarray(maps.mode_order))
    config = _config(
        profile,
        fixed_astig_anchor_nm=100.0,
        dual_domain_coeff_maps=({"name": "left", "coeff_maps_npz": str(path)},),
        domain_count=1,
    )
    with pytest.raises(ValueError, match="fixed_astig_anchor_nm.*coefficient maps"):
        build_online_batch_provider(config)


def test_astigmatism_profile_and_fixed_anchor_cannot_be_combined() -> None:
    with pytest.raises(ValueError, match="fixed_astig_anchor_nm.*conditioning_profile"):
        build_online_batch_provider(_config("astigmatism_660nm", fixed_astig_anchor_nm=100.0))


def test_fixed_anchor_rejects_an_updated_external_absolute_map(tmp_path) -> None:
    maps = build_astigmatism_anchor_conditioning_maps(width=8, height=8)
    path = tmp_path / "map.npz"
    np.savez(path, zernike_maps_nm=maps.full_maps_nm.numpy(), mode_order=np.asarray(maps.mode_order))
    store = ConditioningProviderStore()
    provider = build_online_batch_provider(_config("zero", fixed_astig_anchor_nm=100.0), condition_store=store)
    store.update_from_coeff_maps(({"name": "left", "coeff_maps_npz": str(path)},))

    with pytest.raises(ValueError, match="fixed_astig_anchor_nm.*coefficient maps"):
        provider(1)
