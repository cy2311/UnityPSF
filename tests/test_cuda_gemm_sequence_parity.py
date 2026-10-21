import numpy as np
import torch

import neptune_v03.localization.online as online
from neptune_v03.localization.online import OnlineBatchProviderConfig
from neptune_v03.localization.simulator import LocalizationSimulatorConfig


def test_lut_and_cuda_gemm_share_seeded_temporal_events_and_targets() -> None:
    lut_config = OnlineBatchProviderConfig(
        batch_size=4,
        height=24,
        width=28,
        channels=3,
        emitters_per_sample=6,
        seed=721,
        steps_per_epoch=4,
        photon_range=(500.0, 1200.0),
        background_range=(2.0, 5.0),
        z_range=(-0.5, 0.5),
        lifetime_avg=1.5,
        warmup_frames=3.0,
    )
    cuda_config = OnlineBatchProviderConfig(
        **{**lut_config.__dict__, "simulation_backend": "cuda_gemm"}
    )
    sim_config = LocalizationSimulatorConfig(
        batch_size=1,
        height=lut_config.height,
        width=lut_config.width,
        seed=18421,
        output_device="cpu",
    )

    lut_events = online._sample_lut_sequence_events(lut_config, sim_config, frame_count=6)
    cuda_events = online._sample_lut_sequence_events(cuda_config, sim_config, frame_count=6)

    assert lut_events["counts"] == cuda_events["counts"]
    for key in ("frame_indices", "emitter_indices", "overlaps", "pxyz", "mask", "background"):
        assert torch.equal(lut_events[key], cuda_events[key])
    assert int(lut_events["mask"].sum()) == sum(lut_events["counts"])


def test_cached_window_cuda_gemm_uses_shared_temporal_sequence_path(monkeypatch) -> None:
    calls = []

    def shared_sequence(*_args, **_kwargs):
        calls.append("shared")
        return {"sequence_source": "shared_lut_sampling_cuda_renderer"}

    def native_sequence(*_args, **_kwargs):
        calls.append("native")
        raise AssertionError("CUDA GEMM must not switch to native sequence sampling")

    monkeypatch.setattr(online, "_simulate_lut_like_sequence", shared_sequence)
    monkeypatch.setattr(online, "_simulate_native_sequence", native_sequence)

    result = online._simulate_cached_window_sequence(
        OnlineBatchProviderConfig(batch_size=1, simulation_backend="cuda_gemm"),
        LocalizationSimulatorConfig(batch_size=1),
        epoch=1,
        sequence_idx=0,
        frame_count=5,
    )

    assert calls == ["shared"]
    assert result["sequence_source"] == "shared_lut_sampling_cuda_renderer"


def test_lut_and_cuda_gemm_build_same_seeded_cached_eval_batch(monkeypatch) -> None:
    class FakeLutBank:
        patches = torch.zeros((1, 1, 1, 1, 1))
        subpixel_bins = 1

    class FakeLutBankCache:
        def get(self, *_args, **_kwargs):
            return FakeLutBank()

    class FakeCudaRenderer:
        device = torch.device("cpu")

        def render_frames(self, *, frame_count, height, width, **_kwargs):
            return torch.zeros((frame_count, height, width))

    class FakeRendererCache:
        def get(self, *_args, **_kwargs):
            return FakeCudaRenderer()

    monkeypatch.setattr(
        online,
        "_lookup_lut_patches",
        lambda _bank, *, xs, **_kwargs: torch.zeros((xs.numel(), 3, 3)),
    )
    monkeypatch.setattr(online, "_apply_lut_subpixel_shift", lambda patches, **_kwargs: patches)

    common = dict(
        batch_size=4,
        height=24,
        width=28,
        channels=3,
        emitters_per_sample=6,
        photon_range=(500.0, 1200.0),
        background_range=(2.0, 5.0),
        z_range=(-0.5, 0.5),
        lifetime_avg=1.5,
        warmup_frames=3.0,
        simulation_output_device="cpu",
    )
    sim_config = LocalizationSimulatorConfig(
        batch_size=1,
        height=24,
        width=28,
        seed=18421,
        output_device="cpu",
    )
    torch.manual_seed(18421)
    lut_batch = online._build_lut_like_sequence_batch(
        OnlineBatchProviderConfig(**common, simulation_backend="lut"),
        sim_config,
        epoch=1,
        step=1,
        renderer_cache=FakeRendererCache(),
        lut_patch_bank_cache=FakeLutBankCache(),
    )
    torch.manual_seed(18421)
    cuda_batch = online._build_lut_like_sequence_batch(
        OnlineBatchProviderConfig(**common, simulation_backend="cuda_gemm"),
        sim_config,
        epoch=1,
        step=1,
        renderer_cache=FakeRendererCache(),
        lut_patch_bank_cache=FakeLutBankCache(),
    )

    assert torch.equal(lut_batch.pxyz_tar, cuda_batch.pxyz_tar)
    assert torch.equal(lut_batch.mask_tar, cuda_batch.mask_tar)
    assert torch.equal(lut_batch.detect_tar, cuda_batch.detect_tar)
    assert torch.equal(lut_batch.bkg_tar, cuda_batch.bkg_tar)
    assert torch.equal(lut_batch.model_input, cuda_batch.model_input)
    assert lut_batch.model_input.shape == (4, 3, 24, 28)
    assert lut_batch.metadata["emitter_counts"] == cuda_batch.metadata["emitter_counts"]
