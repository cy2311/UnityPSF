from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from unity_psf.localization.online import OnlineBatchProviderConfig, build_online_batch_provider
from unity_psf.localization.smlm_targets import LEGACY_IWAE_PXYZ_TARGET_ORDER
from unity_psf.localization.training_adapter import LocalizationTrainBatch


def _provider_config(**overrides: object) -> OnlineBatchProviderConfig:
    values: dict[str, object] = {
        "batch_size": 1,
        "channels": 3,
        "height": 8,
        "width": 8,
        "emitters_per_sample": 1,
        "seed": 73,
        "steps_per_epoch": 1,
        "npupil": 8,
        "vector_psf_size": 5,
        "vector_batch_size": 1,
        "simulation_output_device": "cpu",
        "z_range": (0.0, 0.0),
    }
    values.update(overrides)
    return OnlineBatchProviderConfig(**values)


def _single_batch(config: OnlineBatchProviderConfig) -> LocalizationTrainBatch:
    training_batch = next(iter(build_online_batch_provider(config)(1)))
    assert isinstance(training_batch.inputs, LocalizationTrainBatch)
    return training_batch.inputs


@pytest.mark.parametrize(
    "config",
    [
        _provider_config(),
        _provider_config(simulation_backend="lut", lut_field_stride=8, lut_z_steps=1),
        _provider_config(batch_strategy="cached_window", sequence_count=1),
    ],
    ids=["native", "lut", "cached_window"],
)
def test_provider_routes_preserve_training_batch_contract(config: OnlineBatchProviderConfig) -> None:
    batch = _single_batch(config)

    assert isinstance(batch.model_input, torch.Tensor)
    assert batch.model_input.shape == (1, 3, 8, 8)
    assert batch.model_input.dtype == torch.float32
    assert batch.model_input.device.type == "cpu"
    assert batch.detect_tar.shape == (1, 8, 8)
    assert batch.bkg_tar.shape == (1, 8, 8)
    assert batch.pxyz_tar.shape[-1] == 4
    assert batch.mask_tar.shape == batch.pxyz_tar.shape[:2]
    assert batch.pxyz_tar.dtype == torch.float32
    assert batch.mask_tar.dtype == torch.bool
    assert tuple(batch.metadata["pxyz_target_order"]) == LEGACY_IWAE_PXYZ_TARGET_ORDER
    assert int(batch.metadata["seed"]) >= 0


@pytest.mark.parametrize(
    "config",
    [
        _provider_config(),
        _provider_config(simulation_backend="lut", lut_field_stride=8, lut_z_steps=1),
        _provider_config(batch_strategy="cached_window", sequence_count=1),
    ],
    ids=["native", "lut", "cached_window"],
)
def test_provider_routes_preserve_seed_order_and_cache_snapshot(config: OnlineBatchProviderConfig) -> None:
    first = _single_batch(config)
    second = _single_batch(config)

    torch.testing.assert_close(first.model_input, second.model_input)
    assert first.metadata["seed"] == second.metadata["seed"]
    assert first.metadata["batch_strategy"] == second.metadata["batch_strategy"]
    assert first.metadata["lut_cache_generation"] == second.metadata["lut_cache_generation"]
    if config.batch_strategy == "cached_window":
        assert first.metadata["cached_window_order"] == second.metadata["cached_window_order"]
        assert first.metadata["window_global_center_indices"] == second.metadata["window_global_center_indices"]




def test_lut_camera_readout_is_seed_deterministic_and_uses_raw_adu_domain() -> None:
    config = _provider_config(
        simulation_backend="lut",
        lut_field_stride=8,
        lut_z_steps=1,
        background=0.0,
        camera_qe=1.0,
        camera_spurious_charge=0.0,
        camera_baseline=17.0,
        camera_e_per_adu=1.0,
    )

    first = _single_batch(config)
    second = _single_batch(config)

    torch.testing.assert_close(first.model_input, second.model_input)
    assert first.metadata["input_domain"] == "raw_adu"
    assert float(first.model_input.min()) >= 17.0


def test_unknown_simulation_backend_fails_at_provider_boundary() -> None:
    with pytest.raises(ValueError, match="simulation_backend must be 'native' or 'lut'"):
        build_online_batch_provider(_provider_config(simulation_backend="unknown"))


def test_film_conditioning_does_not_mutate_the_image_batch_contract() -> None:
    image_batch = _single_batch(_provider_config())
    film_batch = _single_batch(
        _provider_config(
            conditioning_mode="film",
            condition_feature_dim=8,
            condition_dim=8,
            conditioning_profile="zero",
        )
    )

    assert isinstance(film_batch.model_input, tuple)
    image, condition = film_batch.model_input
    assert image.shape == image_batch.model_input.shape
    assert condition.shape == (1, 8)
    assert film_batch.detect_tar.shape == image_batch.detect_tar.shape
    assert film_batch.metadata["conditioning_mode"] == "film"


def test_cached_window_film_conditioning_materializes_precomputed_features() -> None:
    batch = _single_batch(
        _provider_config(
            batch_strategy="cached_window",
            sequence_count=1,
            conditioning_mode="film",
            condition_feature_dim=8,
            condition_dim=8,
            conditioning_profile="zero",
        )
    )

    image, condition = batch.model_input
    assert image.shape == (1, 3, 8, 8)
    assert condition.shape == (1, 8)
