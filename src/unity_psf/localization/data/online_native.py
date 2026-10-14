"""Native triplet and sequence simulation."""

from __future__ import annotations

from dataclasses import replace
import torch
from unity_psf.localization.conditioning import FullResZernikeConditioning
from unity_psf.localization.smlm_targets import V03_PXYZ_TARGET_ORDER
from unity_psf.localization.simulator import LocalizationSimulatorConfig
from unity_psf.localization.simulator import build_emitter_renderer
from unity_psf.localization.simulator import max_emitters_per_sample
from unity_psf.localization.simulator import sample_counts
from unity_psf.localization.simulator import sample_photons
from unity_psf.localization.simulator import sample_range
from unity_psf.localization.simulator import simulate_localization_batch
from unity_psf.localization.training_adapter import LocalizationTrainBatch
from .online_types import OnlineBatchProviderConfig
from .online_physical import VectorRendererCache
from .online_conditioning import attach_film_conditioning
from .online_geometry import condition_origin
from .online_geometry import condition_roi_extent
from .online_targets import finalize_pxyz_targets
from .online_camera import poisson_camera_readout
from .online_targets import pxyz_target_order_tuple
from .online_conditioning import domain_index as resolve_domain_index
from .online_conditioning import select_condition_provider
from .online_physical import simulator_config
from .online_targets import target_contract_for_config


def build_native_online_batch(
    config: OnlineBatchProviderConfig,
    *,
    epoch: int,
    seed: int,
    step: int,
    global_step: int,
    condition_providers: tuple[tuple[str | None, FullResZernikeConditioning], ...] | None,
    condition_store_version: int | None = None,
    cache_generation: int = 0,
    renderer_cache: VectorRendererCache | None = None,
) -> LocalizationTrainBatch:
    strategy = str(config.batch_strategy)
    if strategy not in {"triplet", "sequence_window"}:
        raise ValueError(f"unsupported online batch_strategy: {strategy}")
    domain_index = resolve_domain_index(config, global_step=global_step)
    condition_provider, _, _ = select_condition_provider(
        config,
        domain_index=domain_index,
        condition_providers=condition_providers,
    )
    roi_width = condition_roi_extent(int(config.width), int(config.nat_grid_size[0] if isinstance(config.nat_grid_size, tuple) else config.nat_grid_size))
    roi_height = condition_roi_extent(int(config.height), int(config.nat_grid_size[1] if isinstance(config.nat_grid_size, tuple) else config.nat_grid_size))
    field_origin = condition_origin(
        config,
        sample_index=0,
        global_step=global_step,
        roi_width=roi_width,
        roi_height=roi_height,
    )
    sim_config = simulator_config(
        config,
        seed=int(seed),
        field_origin=field_origin,
        frames_per_sample=int(config.channels),
        condition_provider=condition_provider,
    )
    if strategy == "sequence_window":
        loc_batch = build_sequence_window_batch(config, sim_config, epoch=int(epoch), step=int(step))
    else:
        renderer = None if renderer_cache is None else renderer_cache.get(
            sim_config,
            physical_model_version=condition_store_version,
        )
        loc_batch = simulate_localization_batch(
            sim_config,
            epoch=int(epoch),
            step=int(step),
            source="native_online",
            renderer=renderer,
        )
        if config.camera_readout:
            loc_batch = replace(
                loc_batch,
                model_input=poisson_camera_readout(config, loc_batch.model_input, seed=int(sim_config.seed)),
                metadata={**loc_batch.metadata, "input_domain": "raw_adu"},
            )
    loc_batch.metadata.update(
        {
            "batch_strategy": strategy,
            "frames_per_sample": int(config.channels),
            "sequence_window_chunks": int(config.sequence_window_chunks),
            **({} if condition_store_version is None else {"physical_model_version": int(condition_store_version)}),
            "cache_generation": int(cache_generation),
        }
    )
    target_order = loc_batch.metadata.get("pxyz_target_order")
    expected_target_order = pxyz_target_order_tuple(config)
    if tuple(target_order or ()) != expected_target_order:
        pxyz_tar, target_order = finalize_pxyz_targets(config, loc_batch.pxyz_tar)
        loc_batch = LocalizationTrainBatch(
            model_input=loc_batch.model_input,
            detect_tar=loc_batch.detect_tar,
            bkg_tar=loc_batch.bkg_tar,
            pxyz_tar=pxyz_tar,
            mask_tar=loc_batch.mask_tar,
            metadata={
                **loc_batch.metadata,
                "pxyz_target_order": target_order,
                "target_contract": target_contract_for_config(config).as_dict(),
            },
        )
    if str(config.conditioning_mode) == "film":
        loc_batch = attach_film_conditioning(
            config,
            loc_batch,
            global_step=global_step,
            domain_index=domain_index,
            condition_providers=condition_providers,
            condition_store_version=condition_store_version,
        )
    return loc_batch


def simulate_native_sequence(
    config: OnlineBatchProviderConfig,
    sim_config: LocalizationSimulatorConfig,
    *,
    epoch: int,
    sequence_idx: int,
    frame_count: int,
    physical_model_version: int | None = None,
    cache_generation: int = 0,
    renderer_cache: VectorRendererCache | None = None,
) -> dict[str, object]:
    sequence_config = replace(sim_config, batch_size=int(frame_count), frames_per_sample=1)
    cache_hits_before = 0 if renderer_cache is None else int(renderer_cache.hits)
    cache_misses_before = 0 if renderer_cache is None else int(renderer_cache.misses)
    sequence = simulate_native_sequence_batched(
        config,
        sequence_config,
        physical_model_version=physical_model_version,
        renderer_cache=renderer_cache,
    )
    cache_hits_after = 0 if renderer_cache is None else int(renderer_cache.hits)
    cache_misses_after = 0 if renderer_cache is None else int(renderer_cache.misses)
    sequence.metadata.update(
        {
            "epoch": int(epoch),
            "step": int(sequence_idx) + 1,
            "seed": int(sequence_config.seed),
            "source": "native_cached_window_batched_sequence_frames",
            "sequence_source": "native_cached_window_batched_sequence_frames",
            "renderer_cache_hits": cache_hits_after - cache_hits_before,
            "renderer_cache_misses": cache_misses_after - cache_misses_before,
            "render_many_calls": 0,
            "render_frames_calls": int(sequence.metadata.get("render_frames_calls", 0)),
            **({} if physical_model_version is None else {"physical_model_version": int(physical_model_version)}),
            "cache_generation": int(cache_generation),
        }
    )
    return {
        "frames": sequence.model_input[:, 0],
        "pxyz": sequence.pxyz_tar,
        "mask": sequence.mask_tar,
        "bkg": sequence.bkg_tar,
        "counts": [int(item) for item in sequence.metadata.get("emitter_counts", [])],
        "metadata": dict(sequence.metadata),
        "sequence_source": "native_cached_window_batched_sequence_frames",
    }


def simulate_native_sequence_batched(
    config: OnlineBatchProviderConfig,
    sim_config: LocalizationSimulatorConfig,
    *,
    physical_model_version: int | None = None,
    renderer_cache: VectorRendererCache | None = None,
) -> LocalizationTrainBatch:
    frame_count = int(sim_config.batch_size)
    height = int(sim_config.height)
    width = int(sim_config.width)
    max_emitters = max_emitters_per_sample(sim_config)
    if min(frame_count, height, width, max_emitters) <= 0:
        raise ValueError("native sequence dimensions and emitter count must be positive")

    generator = torch.Generator().manual_seed(int(sim_config.seed))
    counts = sample_counts(sim_config, batch_size=frame_count, generator=generator)
    xs = torch.zeros((frame_count, max_emitters), dtype=torch.float32)
    ys = torch.zeros((frame_count, max_emitters), dtype=torch.float32)
    photons = torch.zeros((frame_count, max_emitters), dtype=torch.float32)
    z = torch.zeros((frame_count, max_emitters), dtype=torch.float32)
    mask = torch.zeros((frame_count, max_emitters), dtype=torch.bool)
    for frame_idx, count in enumerate(counts.tolist()):
        count_i = int(count)
        if count_i <= 0:
            continue
        mask[frame_idx, :count_i] = True
        xs[frame_idx, :count_i] = torch.rand((count_i,), generator=generator) * float(width - 1)
        ys[frame_idx, :count_i] = torch.rand((count_i,), generator=generator) * float(height - 1)
        photons[frame_idx, :count_i] = sample_photons(sim_config, emitters=count_i, generator=generator)
        z[frame_idx, :count_i] = sample_range(sim_config.z_range, shape=(count_i,), fallback=0.0, generator=generator)
    background = sample_range(
        sim_config.background_range,
        shape=(frame_count,),
        fallback=float(sim_config.background),
        generator=generator,
    )

    if renderer_cache is None:
        renderer = build_emitter_renderer(sim_config)
    else:
        renderer = renderer_cache.get(sim_config, physical_model_version=physical_model_version)
    device = renderer.device
    xs = xs.to(device=device)
    ys = ys.to(device=device)
    photons = photons.to(device=device)
    z = z.to(device=device)
    mask = mask.to(device=device)
    background = background.to(device=device)

    frames = background.view(frame_count, 1, 1).expand(frame_count, height, width).clone()
    render_frames_calls = 0
    active = mask.reshape(-1)
    if bool(active.any()):
        all_frame_indices = torch.arange(frame_count, device=device, dtype=torch.long).view(frame_count, 1).expand(frame_count, max_emitters)
        flat_frame_indices = all_frame_indices.reshape(-1)[active]
        flat_xs = xs.reshape(-1)[active]
        flat_ys = ys.reshape(-1)[active]
        flat_z = z.reshape(-1)[active]
        flat_photons = photons.reshape(-1)[active]
        frames += renderer.render_frames(
            frame_count=frame_count,
            height=height,
            width=width,
            frame_indices=flat_frame_indices,
            xs=flat_xs,
            ys=flat_ys,
            z_um=flat_z,
            photons=flat_photons,
            field_origin_xy=tuple(int(v) for v in sim_config.field_origin_xy),
        )
        render_frames_calls = 1

    if config.camera_readout or str(sim_config.psf_type).strip().lower() == "empirical_focal":
        frames = poisson_camera_readout(config, frames, seed=int(sim_config.seed))
    bkg_tar = background.view(frame_count, 1, 1).expand(frame_count, height, width).clone() / max(float(config.background_scale), 1e-12)
    batch = LocalizationTrainBatch(
        model_input=frames[:, None].to(dtype=torch.float32).contiguous(),
        detect_tar=torch.zeros((frame_count, height, width), dtype=torch.float32, device=device),
        bkg_tar=bkg_tar.to(dtype=torch.float32).contiguous(),
        pxyz_tar=torch.stack([xs, ys, z, photons], dim=-1).to(dtype=torch.float32).contiguous(),
        mask_tar=mask.contiguous(),
        metadata={
            "seed": int(sim_config.seed),
            "simulation_backend": str(sim_config.simulation_backend),
            "input_domain": "raw_adu" if config.camera_readout or str(sim_config.psf_type).strip().lower() == "empirical_focal" else "photons",
            "pxyz_target_order": V03_PXYZ_TARGET_ORDER,
            "target_contract": target_contract_for_config(config).with_target_order("v03").as_dict(),
            "background_scale": float(config.background_scale),
            "psf_type": str(sim_config.psf_type).lower(),
            "emitter_counts": [int(v) for v in counts.tolist()],
            "render_frames_calls": render_frames_calls,
            **renderer.metadata,
        },
    )
    if str(config.simulation_output_device).strip().lower() == "cpu":
        return LocalizationTrainBatch(
            model_input=batch.model_input.cpu(),
            detect_tar=batch.detect_tar.cpu(),
            bkg_tar=batch.bkg_tar.cpu(),
            pxyz_tar=batch.pxyz_tar.cpu(),
            mask_tar=batch.mask_tar.cpu(),
            metadata=batch.metadata,
        )
    if str(config.simulation_output_device).strip().lower() == "renderer":
        return batch
    raise ValueError("simulation_output_device must be 'cpu' or 'renderer'")


def build_sequence_window_batch(
    config: OnlineBatchProviderConfig,
    sim_config: LocalizationSimulatorConfig,
    *,
    epoch: int,
    step: int,
) -> LocalizationTrainBatch:
    window_size = int(config.channels)
    if window_size != 3:
        raise ValueError("sequence_window currently expects channels/window_size=3")
    center_offset = window_size // 2
    center_count = int(config.batch_size)
    sequence_frames = center_count + window_size - 1
    sequence = simulate_localization_batch(
        replace(sim_config, batch_size=sequence_frames, frames_per_sample=1),
        epoch=int(epoch),
        step=int(step),
        source="native_online_sequence_frames",
    )
    frame_images = sequence.model_input[:, 0]
    if config.camera_readout:
        frame_images = poisson_camera_readout(config, frame_images, seed=int(sim_config.seed))
    frames = []
    detect = []
    bkg = []
    pxyz = []
    mask = []
    target_indices = []
    for start in range(center_count):
        target_idx = int(start) + center_offset
        frames.append(frame_images[start : start + window_size])
        detect.append(sequence.detect_tar[target_idx])
        bkg.append(sequence.bkg_tar[target_idx])
        target, target_order = finalize_pxyz_targets(config, sequence.pxyz_tar[target_idx])
        pxyz.append(target)
        mask.append(sequence.mask_tar[target_idx])
        target_indices.append(target_idx)
    metadata = {
        **sequence.metadata,
        "source": "native_online",
        "sequence_source": "native_online_sequence_frames",
        "frames_per_sample": window_size,
        "sequence_simulated_frames": sequence_frames,
        "center_target_frame_indices": target_indices,
        "emitter_counts": [int(sequence.metadata["emitter_counts"][idx]) for idx in target_indices],
        "pxyz_target_order": target_order,
        "target_contract": target_contract_for_config(config).as_dict(),
        "input_domain": "raw_adu" if config.camera_readout else "photons",
    }
    return LocalizationTrainBatch(
        model_input=torch.stack(frames, dim=0).contiguous(),
        detect_tar=torch.stack(detect, dim=0).contiguous(),
        bkg_tar=torch.stack(bkg, dim=0).contiguous(),
        pxyz_tar=torch.stack(pxyz, dim=0).contiguous(),
        mask_tar=torch.stack(mask, dim=0).contiguous(),
        metadata=metadata,
    )
