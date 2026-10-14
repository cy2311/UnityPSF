from __future__ import annotations

from dataclasses import replace

import torch
import pytest

from unity_psf.localization.conditioning import build_zero_conditioning_maps
from unity_psf.localization.data.online import OnlineBatchProviderConfig, build_online_batch_provider
from unity_psf.localization.training_adapter import prefetch_training_batches
from unity_psf.training.loop import TrainingBatch


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
        simulation_backend="native",
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
        simulation_backend="native",
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

    left_epoch_1_batches = [batch.inputs for batch in left(1)]
    left_epoch_2_batches = [batch.inputs for batch in left(2)]
    right_epoch_1_batches = [batch.inputs for batch in right(1)]
    left_epoch_1 = left_epoch_1_batches[0]
    left_epoch_2 = left_epoch_2_batches[0]
    right_epoch_1 = right_epoch_1_batches[0]

    assert len({left_epoch_1.metadata["seed"], left_epoch_2.metadata["seed"], right_epoch_1.metadata["seed"]}) == 3
    assert left_epoch_1.metadata["cached_window_cache_epoch"] == 1
    assert left_epoch_2.metadata["cached_window_cache_epoch"] == 2
    assert right_epoch_1.metadata["cached_window_cache_epoch"] == 1


def test_cached_window_reuses_cpu_cache_after_configured_epoch_slots() -> None:
    config = replace(
        _seed_test_config(seed=53, batch_strategy="cached_window"),
        steps_per_epoch=1,
        cached_window_cache_epochs=2,
        cached_window_cache_storage="cpu",
        cached_window_prefetch_batches=1,
    )
    provider = build_online_batch_provider(config)

    epoch_1 = next(iter(provider(1))).inputs
    epoch_3 = next(iter(provider(3))).inputs

    assert torch.equal(epoch_1.model_input, epoch_3.model_input)
    assert torch.equal(epoch_1.pxyz_tar, epoch_3.pxyz_tar)
    assert epoch_1.metadata["cached_window_cache_slot"] == epoch_3.metadata["cached_window_cache_slot"]
    assert epoch_1.metadata["cached_window_cache_epoch"] == 1
    assert epoch_3.metadata["cached_window_cache_epoch"] == 1


def test_fixed_cache_uses_liteloc_cursor_without_epoch_shuffle() -> None:
    config = replace(
        _seed_test_config(seed=56, batch_strategy="cached_window"),
        steps_per_epoch=2,
        batch_size=1,
        cached_window_cache_frames=4,
        cached_window_cache_storage="cpu",
        cached_window_order="shuffle",
        sequence_count=1,
    )
    provider = build_online_batch_provider(config)

    epoch_1 = list(provider(1))
    epoch_2 = list(provider(2))
    epoch_3 = list(provider(3))

    assert [b.inputs.metadata["center_target_frame_indices"][0] for b in epoch_1] == [1, 2]
    assert [b.inputs.metadata["center_target_frame_indices"][0] for b in epoch_2] == [3, 4]
    assert [b.inputs.metadata["center_target_frame_indices"][0] for b in epoch_3] == [1, 2]
    assert all(b.inputs.metadata["cached_window_order"] == "temporal" for b in epoch_1)
    assert all(b.inputs.metadata["cached_window_cache_frames"] == 4 for b in epoch_1)


def test_cached_window_default_order_preserves_temporal_center_order() -> None:
    config = replace(
        _seed_test_config(seed=55, batch_strategy="cached_window"),
        batch_size=1,
        steps_per_epoch=3,
        sequence_count=1,
        cached_window_order="temporal",
    )
    batch = next(iter(build_online_batch_provider(config)(1))).inputs

    assert batch.metadata["cached_window_order"] == "temporal"
    assert batch.metadata["center_target_frame_indices"] == [1]
    all_batches = list(build_online_batch_provider(config)(1))
    assert [item.inputs.metadata["center_target_frame_indices"][0] for item in all_batches] == [1, 2, 3]


def test_cached_window_physical_refresh_discards_persistent_slots() -> None:
    config = replace(_seed_test_config(seed=54, batch_strategy="cached_window"), steps_per_epoch=1)
    provider = build_online_batch_provider(config)

    first = provider(1)
    next(iter(first))
    provider.refresh_physical_model()
    second = provider(1)
    next(iter(second))

    assert first._loc_batches is not second._loc_batches


def test_prefetch_helper_is_a_cpu_noop() -> None:
    batch = TrainingBatch(
        inputs=torch.ones((1, 1)),
        targets=torch.zeros((1, 1)),
    )
    result = list(prefetch_training_batches([batch], torch.device("cpu"), prefetch_batches=1))
    assert result[0] is batch


def test_astigmatism_anchor_profile_is_a_real_100nm_renderer_coefficient() -> None:
    from unity_psf.localization.conditioning import build_astigmatism_anchor_conditioning_maps

    conditioning = build_astigmatism_anchor_conditioning_maps(
        width=8,
        height=8,
        profile_name="astigmatism_660nm",
    )
    mode_to_index = {mode: index for index, mode in enumerate(conditioning.mode_order)}

    assert torch.all(conditioning.full_maps_nm[mode_to_index[(2, 2)]] == 100.0)
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


def test_cached_cuda_gemm_frames_are_distinct_and_data_seed_reproducible() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA-GEMM simulation requires CUDA")
    config = replace(
        _seed_test_config(seed=71, batch_strategy="cached_window"),
        steps_per_epoch=1,
        simulation_backend="cuda_gemm",
        simulation_output_device="cpu",
        z_range=(0.0, 0.0),
        camera_qe=0.9,
        camera_spurious_charge=0.002,
        camera_baseline=398.6,
        camera_e_per_adu=1.020784562122306,
    )

    first = next(iter(build_online_batch_provider(config)(1))).inputs
    second = next(iter(build_online_batch_provider(config)(1))).inputs
    images = first.model_input

    assert first.metadata["simulation_backend"] == "cuda_gemm"
    assert not torch.equal(images[:, 0], images[:, 1])
    assert not torch.equal(images[:, 1], images[:, 2])
    assert torch.equal(images, second.model_input)


def test_removed_lut_backend_is_rejected_before_simulation() -> None:
    config = replace(_seed_test_config(seed=72), simulation_backend="lut")
    with pytest.raises(ValueError, match="simulation_backend must be 'native' or 'cuda_gemm'"):
        build_online_batch_provider(config)


def test_camera_readout_is_applied_once_to_cached_sequence() -> None:
    config = replace(
        _seed_test_config(seed=73, batch_strategy="cached_window"),
        simulation_backend="native",
        camera_readout=True,
        camera_baseline=398.6,
        camera_e_per_adu=1.020784562122306,
        camera_qe=0.9,
        camera_spurious_charge=0.002,
        signal=0.0,
        background=0.0,
        photon_range=(0.0, 0.0),
        background_range=(0.0, 0.0),
    )
    batch = next(iter(build_online_batch_provider(config)(1))).inputs

    assert batch.metadata["input_domain"] == "raw_adu"
    assert 390.0 < float(batch.model_input.mean()) < 405.0
