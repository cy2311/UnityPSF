from __future__ import annotations

from dataclasses import replace

import torch

from unity_psf.localization.conditioning import build_zero_conditioning_maps
from unity_psf.localization.online import OnlineBatchProviderConfig, build_online_batch_provider


def test_zero_conditioning_maps_have_all_film_modes_but_no_aberration() -> None:
    conditioning = build_zero_conditioning_maps(width=8, height=8)

    assert conditioning.image_shape_hw == (8, 8)
    assert tuple(conditioning.input_mode_order) == (
        (2, 0),
        (3, 1),
        (3, -1),
        (4, 0),
        (3, -3),
        (3, 3),
    )
    assert torch.count_nonzero(conditioning.full_maps_nm) == 0


def test_zero_conditioning_profile_keeps_xy_and_zeros_zernike_features() -> None:
    config = OnlineBatchProviderConfig(
        batch_size=1,
        channels=3,
        height=8,
        width=8,
        emitters_per_sample=1,
        steps_per_epoch=1,
        conditioning_mode="film",
        condition_feature_dim=8,
        condition_dim=8,
        conditioning_profile="zero",
        npupil=8,
        vector_psf_size=5,
        vector_batch_size=1,
    )
    provider = build_online_batch_provider(config)
    batch = next(iter(provider(1))).inputs
    _images, conditions = batch.model_input

    assert conditions.shape == (1, 8)
    assert torch.count_nonzero(conditions[:, 2:]) == 0


def _seed_test_config(
    *,
    seed: int,
    batch_strategy: str = "triplet",
    conditioning_profile: str = "zero",
) -> OnlineBatchProviderConfig:
    return OnlineBatchProviderConfig(
        batch_size=1,
        channels=3,
        height=8,
        width=8,
        emitters_per_sample=1,
        seed=seed,
        steps_per_epoch=2,
        conditioning_profile=conditioning_profile,
        batch_strategy=batch_strategy,
        sequence_count=1,
        npupil=8,
        vector_psf_size=5,
        vector_batch_size=1,
    )


def test_online_batch_seeds_do_not_repeat_across_epochs_or_data_streams() -> None:
    left = build_online_batch_provider(_seed_test_config(seed=41))
    right = build_online_batch_provider(_seed_test_config(seed=42))

    left_epoch_1 = {int(batch.inputs.metadata["seed"]) for batch in left(1)}
    left_epoch_2 = {int(batch.inputs.metadata["seed"]) for batch in left(2)}
    right_epoch_1 = {int(batch.inputs.metadata["seed"]) for batch in right(1)}

    assert left_epoch_1.isdisjoint(left_epoch_2)
    assert left_epoch_1.isdisjoint(right_epoch_1)


def test_cached_window_seed_changes_between_epochs() -> None:
    left = build_online_batch_provider(_seed_test_config(seed=51, batch_strategy="cached_window"))
    right = build_online_batch_provider(_seed_test_config(seed=52, batch_strategy="cached_window"))

    left_epoch_1 = next(iter(left(1))).inputs
    left_epoch_2 = next(iter(left(2))).inputs
    right_epoch_1 = next(iter(right(1))).inputs

    assert len({left_epoch_1.metadata["seed"], left_epoch_2.metadata["seed"], right_epoch_1.metadata["seed"]}) == 3
    assert not torch.equal(left_epoch_1.model_input, left_epoch_2.model_input)
    assert not torch.equal(left_epoch_1.model_input, right_epoch_1.model_input)


def test_astigmatism_anchor_profile_is_a_real_99nm_renderer_coefficient() -> None:
    from unity_psf.localization.conditioning import build_astigmatism_anchor_conditioning_maps

    conditioning = build_astigmatism_anchor_conditioning_maps(
        width=8,
        height=8,
        profile_name="astigmatism_660nm",
    )
    mode_to_index = {mode: index for index, mode in enumerate(conditioning.mode_order)}

    assert torch.all(conditioning.full_maps_nm[mode_to_index[(2, 2)]] == 99.0)
    assert torch.count_nonzero(conditioning.full_maps_nm[mode_to_index[(2, -2)]]) == 0
    assert torch.count_nonzero(conditioning.full_maps_nm[mode_to_index[(2, 0)]]) == 0

    zero_config = replace(
        _seed_test_config(seed=61),
        conditioning_mode="film",
        condition_feature_dim=8,
        condition_dim=8,
    )
    astigmatism_config = replace(
        zero_config,
        conditioning_profile="astigmatism_660nm",
    )
    zero = build_online_batch_provider(zero_config)
    astigmatism = build_online_batch_provider(
        astigmatism_config
    )
    zero_batch = next(iter(zero(1))).inputs
    astigmatism_batch = next(iter(astigmatism(1))).inputs

    assert astigmatism_batch.metadata["condition_source"] == "astigmatism_anchor_profile"
    assert not torch.allclose(zero_batch.model_input[0], astigmatism_batch.model_input[0])


def test_cached_lut_temporal_camera_frames_are_distinct_and_data_seed_reproducible() -> None:
    config = replace(
        _seed_test_config(seed=71, batch_strategy="cached_window"),
        steps_per_epoch=1,
        simulation_backend="lut",
        simulation_output_device="cpu",
        lut_field_stride=8,
        lut_z_steps=1,
        z_range=(0.0, 0.0),
        camera_qe=0.9,
        camera_spurious_charge=0.002,
        camera_baseline=398.6,
        camera_e_per_adu=1.020784562122306,
    )

    first = next(iter(build_online_batch_provider(config)(1))).inputs
    second = next(iter(build_online_batch_provider(config)(1))).inputs
    images = first.model_input

    assert first.metadata["input_domain"] == "raw_adu"
    assert first.metadata["simulation_backend_effective"] == "lut_patch_bank"
    assert not torch.equal(images[:, 0], images[:, 1])
    assert not torch.equal(images[:, 1], images[:, 2])
    assert torch.equal(images, second.model_input)
