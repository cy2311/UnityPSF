from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Mapping

import numpy as np
import torch

from neptune_v03.localization.conditioning import (
    ConditioningProviderStore,
    FullResZernikeConditioning,
    _load_conditioning_providers,
    build_default_conditioning_maps,
    condition_feature_order,
)
from neptune_v03.localization.smlm_targets import LEGACY_IWAE_PXYZ_TARGET_ORDER, V03_PXYZ_TARGET_ORDER
from neptune_v03.localization.simulator import LocalizationSimulatorConfig, _build_vector_renderer, simulate_localization_batch
from neptune_v03.localization.training_adapter import LocalizationTrainBatch, to_training_batch
from neptune_v03.training.loop import TrainingBatch


@dataclass(frozen=True)
class OnlineBatchProviderConfig:
    batch_size: int
    channels: int = 3
    height: int = 128
    width: int = 128
    emitters_per_sample: int = 8
    seed: int = 0
    steps_per_epoch: int = 1
    background: float = 0.0
    signal: float = 1.0
    simulation_backend: str = "native"
    psf_type: str = "vector"
    pixel_size_nm_x: float = 101.11
    pixel_size_nm_y: float = 98.83
    wavelength_nm: float = 660.0
    na: float = 1.4
    npupil: int = 128
    vector_psf_size: int = 51
    vector_batch_size: int = 96
    emitter_density_um2: float | None = None
    lifetime_avg: float = 1.0
    warmup_frames: float = 6.0
    photon_range: tuple[float, float] | None = None
    photon_mean: float | None = None
    photon_sigma: float | None = None
    background_range: tuple[float, float] | None = None
    background_scale: float = 1.0
    z_range: tuple[float, float] | None = None
    conditioning_mode: str = "channels"
    nat_simulation_mode: str = "tile_center"
    nat_grid_size: int | tuple[int, int] = 32
    nat_grid_z_steps: int = 41
    append_domain_onehot: bool = False
    condition_feature_dim: int = 0
    condition_dim: int = 0
    domain_count: int = 2
    domain_balance_mode: str = "fixed"
    dual_domain_coeff_maps: tuple[Mapping[str, str], ...] = ()
    batch_strategy: str = "triplet"
    sequence_window_chunks: int = 1
    sequence_count: int = 64
    camera_qe: float = 0.9
    camera_spurious_charge: float = 0.002
    camera_baseline: float = 398.6
    camera_e_per_adu: float = 1.020784562122306
    pxyz_target_order: str = "legacy_iwae"
    photon_scale: float | None = None
    z_scale: float | None = None


def build_online_batch_provider(config: OnlineBatchProviderConfig, *, condition_store: ConditioningProviderStore | None = None):
    if config.dual_domain_coeff_maps and int(config.domain_count) != len(config.dual_domain_coeff_maps):
        raise ValueError("domain_count must match dual_domain_coeff_maps length")
    if condition_store is None and config.dual_domain_coeff_maps:
        condition_store = ConditioningProviderStore.from_coeff_maps(config.dual_domain_coeff_maps)
    condition_providers = None if condition_store is not None else _load_condition_providers(config)

    def provider(epoch: int) -> list[TrainingBatch]:
        if int(config.steps_per_epoch) <= 0:
            raise ValueError("steps_per_epoch must be positive")
        condition_store_version = None
        active_condition_providers = condition_providers
        if condition_store is not None:
            condition_store_version, active_condition_providers = condition_store.snapshot()

        if str(config.batch_strategy) == "cached_window":
            return [
                to_training_batch(batch)
                for batch in _build_cached_window_epoch_batches(
                    config,
                    epoch=int(epoch),
                    condition_providers=active_condition_providers,
                    condition_store_version=condition_store_version,
                )
            ]

        batches: list[TrainingBatch] = []
        base_seed = int(config.seed) + int(epoch)
        for step_idx in range(int(config.steps_per_epoch)):
            seed = base_seed + step_idx
            loc_batch = _build_native_online_batch(
                config,
                epoch=int(epoch),
                seed=seed,
                step=step_idx + 1,
                global_step=(int(epoch) - 1) * int(config.steps_per_epoch) + step_idx,
                condition_providers=active_condition_providers,
                condition_store_version=condition_store_version,
            )
            batches.append(to_training_batch(loc_batch))
        return batches

    return provider


def _build_native_online_batch(
    config: OnlineBatchProviderConfig,
    *,
    epoch: int,
    seed: int,
    step: int,
    global_step: int,
    condition_providers: tuple[tuple[str | None, FullResZernikeConditioning], ...] | None,
    condition_store_version: int | None = None,
) -> LocalizationTrainBatch:
    strategy = str(config.batch_strategy)
    if strategy not in {"triplet", "sequence_window"}:
        raise ValueError(f"unsupported online batch_strategy: {strategy}")
    domain_index = _domain_index(config, global_step=global_step)
    condition_provider, _, _ = _condition_provider(
        config,
        domain_index=domain_index,
        condition_providers=condition_providers,
    )
    roi_width = _condition_roi_extent(int(config.width), int(config.nat_grid_size[0] if isinstance(config.nat_grid_size, tuple) else config.nat_grid_size))
    roi_height = _condition_roi_extent(int(config.height), int(config.nat_grid_size[1] if isinstance(config.nat_grid_size, tuple) else config.nat_grid_size))
    field_origin = _condition_origin(
        config,
        sample_index=0,
        global_step=global_step,
        roi_width=roi_width,
        roi_height=roi_height,
    )
    sim_config = _simulator_config(
        config,
        seed=int(seed),
        field_origin=field_origin,
        frames_per_sample=int(config.channels),
        condition_provider=condition_provider,
    )
    if str(config.simulation_backend).strip().lower() == "lut":
        loc_batch = _build_lut_like_sequence_batch(config, sim_config, epoch=int(epoch), step=int(step))
    elif strategy == "sequence_window":
        loc_batch = _build_sequence_window_batch(config, sim_config, epoch=int(epoch), step=int(step))
    else:
        loc_batch = simulate_localization_batch(
            sim_config,
            epoch=int(epoch),
            step=int(step),
            source="native_online",
        )
    loc_batch.metadata.update(
        {
            "batch_strategy": strategy,
            "frames_per_sample": int(config.channels),
            "sequence_window_chunks": int(config.sequence_window_chunks),
        }
    )
    target_order = loc_batch.metadata.get("pxyz_target_order")
    expected_target_order = _pxyz_target_order_tuple(config)
    if tuple(target_order or ()) != expected_target_order:
        pxyz_tar, target_order = _finalize_pxyz_targets(config, loc_batch.pxyz_tar)
        loc_batch = LocalizationTrainBatch(
            model_input=loc_batch.model_input,
            detect_tar=loc_batch.detect_tar,
            bkg_tar=loc_batch.bkg_tar,
            pxyz_tar=pxyz_tar,
            mask_tar=loc_batch.mask_tar,
            metadata={**loc_batch.metadata, "pxyz_target_order": target_order},
        )
    if str(config.conditioning_mode) == "film":
        loc_batch = _attach_film_conditioning(
            config,
            loc_batch,
            global_step=global_step,
            domain_index=domain_index,
            condition_providers=condition_providers,
            condition_store_version=condition_store_version,
        )
    return loc_batch


def _attach_film_conditioning(
    config: OnlineBatchProviderConfig,
    loc_batch: LocalizationTrainBatch,
    *,
    global_step: int,
    condition_providers: tuple[tuple[str | None, FullResZernikeConditioning], ...] | None,
    domain_index: int | None = None,
    condition_store_version: int | None = None,
) -> LocalizationTrainBatch:
    image = loc_batch.model_input
    if not isinstance(image, torch.Tensor):
        raise TypeError("film conditioning expects tensor image input before conditioning is attached")
    feature_dim = _condition_feature_dim(config)
    domain_count = int(config.domain_count)
    selected_domain = _domain_index(config, global_step=global_step) if domain_index is None else int(domain_index)
    condition, condition_source, domain_name = _build_condition_features(
        config,
        global_step=global_step,
        domain_index=selected_domain,
        image=image,
        condition_providers=condition_providers,
    )
    if bool(config.append_domain_onehot):
        onehot = torch.zeros((int(image.shape[0]), domain_count), dtype=image.dtype, device=image.device)
        onehot[:, selected_domain] = 1.0
        condition = torch.cat([condition, onehot], dim=1)
        domain_onehot_slice = (feature_dim, feature_dim + domain_count)
    else:
        domain_onehot_slice = None
    return LocalizationTrainBatch(
        model_input=(image, condition),
        detect_tar=loc_batch.detect_tar,
        bkg_tar=loc_batch.bkg_tar,
        pxyz_tar=loc_batch.pxyz_tar,
        mask_tar=loc_batch.mask_tar,
        metadata={
            **loc_batch.metadata,
            "conditioning_mode": "film",
            "condition_source": condition_source,
            "condition_dim": int(condition.shape[1]),
            "condition_feature_dim": feature_dim,
            "condition_feature_order": condition_feature_order(
                include_domain_onehot=bool(config.append_domain_onehot),
                domain_count=domain_count,
            ),
            "condition_domain_onehot_slice": domain_onehot_slice,
            "domain_count": domain_count,
            "domain_index": selected_domain,
            **({} if condition_store_version is None else {"condition_store_version": int(condition_store_version)}),
            **({} if domain_name is None else {"domain_name": domain_name}),
        },
    )


def _build_cached_window_epoch_batches(
    config: OnlineBatchProviderConfig,
    *,
    epoch: int,
    condition_providers: tuple[tuple[str | None, FullResZernikeConditioning], ...] | None,
    condition_store_version: int | None = None,
) -> list[LocalizationTrainBatch]:
    window_size = int(config.channels)
    if window_size != 3:
        raise ValueError("cached_window currently expects channels/window_size=3")
    batch_size = int(config.batch_size)
    center_samples = int(config.steps_per_epoch) * batch_size
    sequence_count = max(1, int(config.sequence_count))
    centers_per_sequence = int(math.ceil(float(center_samples) / float(sequence_count)))
    frames_per_sequence = centers_per_sequence + window_size - 1
    base_seed = int(config.seed) + int(epoch)
    global_step_base = (int(epoch) - 1) * int(config.steps_per_epoch)

    sequences = []
    for sequence_idx in range(sequence_count):
        domain_index = _domain_index(config, global_step=global_step_base + sequence_idx)
        condition_provider, _, _ = _condition_provider(
            config,
            domain_index=domain_index,
            condition_providers=condition_providers,
        )
        roi_width = _condition_roi_extent(
            int(config.width),
            int(config.nat_grid_size[0] if isinstance(config.nat_grid_size, tuple) else config.nat_grid_size),
        )
        roi_height = _condition_roi_extent(
            int(config.height),
            int(config.nat_grid_size[1] if isinstance(config.nat_grid_size, tuple) else config.nat_grid_size),
        )
        field_origin = _condition_origin(
            config,
            sample_index=0,
            global_step=global_step_base + sequence_idx,
            roi_width=roi_width,
            roi_height=roi_height,
        )
        sim_config = _simulator_config(
            config,
            seed=base_seed + sequence_idx * 10_000,
            field_origin=field_origin,
            frames_per_sample=1,
            condition_provider=condition_provider,
        )
        if str(config.simulation_backend).strip().lower() == "lut":
            sequence = _simulate_lut_like_sequence(config, sim_config, frame_count=frames_per_sequence)
        else:
            sequence = _simulate_native_sequence(config, sim_config, epoch=epoch, sequence_idx=sequence_idx, frame_count=frames_per_sequence)
        sequences.append(
            {
                **sequence,
                "domain_index": domain_index,
            "domain_name": (
                None
                if not condition_providers
                else condition_providers[int(domain_index) % len(condition_providers)][0]
            ),
                "field_origin": field_origin,
                "global_step": global_step_base + sequence_idx,
            }
        )

    center_records = []
    for sequence_idx, sequence in enumerate(sequences):
        for center_offset in range(centers_per_sequence):
            if len(center_records) >= center_samples:
                break
            center_idx = center_offset + window_size // 2
            center_records.append((sequence_idx, center_idx, len(center_records)))

    rng = np.random.default_rng(base_seed)
    order = np.arange(len(center_records), dtype=np.int64)
    rng.shuffle(order)
    batches: list[LocalizationTrainBatch] = []
    for batch_idx in range(int(config.steps_per_epoch)):
        selected = [center_records[int(idx)] for idx in order[batch_idx * batch_size : (batch_idx + 1) * batch_size]]
        batches.append(
            _slice_cached_window_batch(
                config,
                sequences=sequences,
                selected=selected,
                epoch=epoch,
                batch_idx=batch_idx,
                center_samples=center_samples,
                sequence_count=sequence_count,
                frames_per_sequence=frames_per_sequence,
                condition_providers=condition_providers,
                condition_store_version=condition_store_version,
            )
        )
    return batches


def _simulate_native_sequence(
    config: OnlineBatchProviderConfig,
    sim_config: LocalizationSimulatorConfig,
    *,
    epoch: int,
    sequence_idx: int,
    frame_count: int,
) -> dict[str, object]:
    sequence = simulate_localization_batch(
        replace(sim_config, batch_size=int(frame_count), frames_per_sample=1),
        epoch=int(epoch),
        step=int(sequence_idx) + 1,
        source="native_cached_window_sequence_frames",
    )
    return {
        "frames": sequence.model_input[:, 0],
        "pxyz": sequence.pxyz_tar,
        "mask": sequence.mask_tar,
        "bkg": sequence.bkg_tar,
        "counts": [int(item) for item in sequence.metadata.get("emitter_counts", [])],
        "metadata": dict(sequence.metadata),
        "sequence_source": "native_cached_window_sequence_frames",
    }


def _simulate_lut_like_sequence(
    config: OnlineBatchProviderConfig,
    sim_config: LocalizationSimulatorConfig,
    *,
    frame_count: int,
) -> dict[str, object]:
    frame_photons, pxyz_by_frame, mask_by_frame, bkg_by_frame, counts = _simulate_lut_like_sequence_frames(
        config,
        sim_config,
        frame_count=int(frame_count),
    )
    raw_adu = torch.poisson(
        frame_photons * float(config.camera_qe) + float(config.camera_spurious_charge)
    ) / max(float(config.camera_e_per_adu), 1e-12) + float(config.camera_baseline)
    return {
        "frames": raw_adu.to(dtype=torch.float32),
        "pxyz": pxyz_by_frame,
        "mask": mask_by_frame,
        "bkg": bkg_by_frame / max(float(config.background_scale), 1e-12),
        "counts": [int(item) for item in counts],
        "metadata": {
            "simulation_backend": "lut",
            "input_domain": "raw_adu",
            "frame_emitter_policy": "center_frame_only",
            "background_scale": float(config.background_scale),
            "psf_type": str(config.psf_type).lower(),
        },
        "sequence_source": "native_lut_like_cached_window_sequence_frames",
    }


def _slice_cached_window_batch(
    config: OnlineBatchProviderConfig,
    *,
    sequences: list[dict[str, object]],
    selected: list[tuple[int, int, int]],
    epoch: int,
    batch_idx: int,
    center_samples: int,
    sequence_count: int,
    frames_per_sequence: int,
    condition_providers: tuple[tuple[str | None, FullResZernikeConditioning], ...] | None,
    condition_store_version: int | None = None,
) -> LocalizationTrainBatch:
    window_size = int(config.channels)
    frames = []
    detect = []
    bkg = []
    pxyz = []
    mask = []
    counts = []
    target_indices = []
    sequence_indices = []
    frame_indices = []
    global_center_indices = []
    domain_indices = []
    domain_names = []
    condition_origins = []
    sequence_global_steps = []
    for sequence_idx, center_idx, global_center_idx in selected:
        sequence = sequences[int(sequence_idx)]
        start = int(center_idx) - window_size // 2
        end = start + window_size
        sequence_frames = sequence["frames"]
        sequence_pxyz = sequence["pxyz"]
        sequence_mask = sequence["mask"]
        sequence_bkg = sequence["bkg"]
        if not isinstance(sequence_frames, torch.Tensor):
            raise TypeError("cached sequence frames must be a tensor")
        if not isinstance(sequence_pxyz, torch.Tensor) or not isinstance(sequence_mask, torch.Tensor) or not isinstance(sequence_bkg, torch.Tensor):
            raise TypeError("cached sequence targets must be tensors")
        target = sequence_pxyz[int(center_idx)]
        active = sequence_mask[int(center_idx)]
        frames.append(sequence_frames[start:end])
        detect.append(_detect_from_v03_targets(target, active, height=int(config.height), width=int(config.width)))
        bkg.append(sequence_bkg[int(center_idx)])
        final_target, target_order = _finalize_pxyz_targets(config, target[active])
        pxyz.append(final_target)
        mask.append(torch.ones((int(final_target.shape[0]),), dtype=torch.bool, device=final_target.device))
        sequence_counts = sequence["counts"]
        counts.append(int(sequence_counts[int(center_idx)] if int(center_idx) < len(sequence_counts) else int(active.sum().item())))
        target_indices.append(int(center_idx))
        sequence_indices.append(int(sequence_idx))
        frame_indices.append(tuple(range(start, end)))
        global_center_indices.append(int(global_center_idx))
        domain_indices.append(int(sequence["domain_index"]))
        domain_names.append(None if sequence.get("domain_name") is None else str(sequence["domain_name"]))
        condition_origins.append(tuple(int(item) for item in sequence["field_origin"]))
        sequence_global_steps.append(int(sequence["global_step"]))

    base_metadata = dict(sequences[0].get("metadata", {})) if sequences else {}
    padded_pxyz, padded_mask = _pad_pxyz_targets(pxyz, mask)
    loc_batch = LocalizationTrainBatch(
        model_input=torch.stack(frames, dim=0).to(dtype=torch.float32).contiguous(),
        detect_tar=torch.stack(detect, dim=0).contiguous(),
        bkg_tar=torch.stack(bkg, dim=0).contiguous(),
        pxyz_tar=padded_pxyz.contiguous(),
        mask_tar=padded_mask.contiguous(),
        metadata={
            **base_metadata,
            "epoch": int(epoch),
            "step": int(batch_idx) + 1,
            "seed": int(config.seed) + int(epoch),
            "source": "native_online",
            "batch_strategy": "cached_window",
            "sequence_source": str(sequences[0]["sequence_source"]) if sequences else "cached_window_sequence_frames",
            "frames_per_sample": window_size,
            "sequence_window_chunks": int(config.sequence_window_chunks),
            "sequence_count": int(sequence_count),
            "center_samples": int(center_samples),
            "sequence_simulated_frames": int(frames_per_sequence) * int(sequence_count),
            "frames_per_sequence": int(frames_per_sequence),
            "loader_batch_index": int(batch_idx),
            "shuffle": True,
            "center_target_frame_indices": target_indices,
            "window_sequence_indices": sequence_indices,
            "window_sequence_domain_indices": domain_indices,
            "window_sequence_domain_names": domain_names,
            "window_condition_origins": condition_origins,
            "window_sequence_global_steps": sequence_global_steps,
            "window_frame_indices": frame_indices,
            "window_global_center_indices": global_center_indices,
            "emitter_counts": counts,
            "pxyz_target_order": target_order,
        },
    )
    if str(config.conditioning_mode) == "film":
        loc_batch = _attach_cached_window_film_conditioning(
            config,
            loc_batch,
            condition_providers=condition_providers,
            condition_store_version=condition_store_version,
        )
    return loc_batch


def _pad_pxyz_targets(targets: list[torch.Tensor], masks: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    if not targets:
        return torch.zeros((0, 0, 4), dtype=torch.float32), torch.zeros((0, 0), dtype=torch.bool)
    max_count = max(int(target.shape[0]) for target in targets)
    dtype = targets[0].dtype
    device = targets[0].device
    padded_targets = torch.zeros((len(targets), max_count, 4), dtype=dtype, device=device)
    padded_masks = torch.zeros((len(targets), max_count), dtype=torch.bool, device=device)
    for sample_idx, (target, mask) in enumerate(zip(targets, masks, strict=True)):
        count = int(target.shape[0])
        if count <= 0:
            continue
        padded_targets[sample_idx, :count] = target
        padded_masks[sample_idx, :count] = mask.to(dtype=torch.bool, device=device)
    return padded_targets, padded_masks


def _attach_cached_window_film_conditioning(
    config: OnlineBatchProviderConfig,
    loc_batch: LocalizationTrainBatch,
    *,
    condition_providers: tuple[tuple[str | None, FullResZernikeConditioning], ...] | None,
    condition_store_version: int | None = None,
) -> LocalizationTrainBatch:
    image = loc_batch.model_input
    if not isinstance(image, torch.Tensor):
        raise TypeError("cached-window film conditioning expects tensor image input before conditioning is attached")
    feature_dim = _condition_feature_dim(config)
    domain_count = int(config.domain_count)
    domain_indices = [int(item) for item in loc_batch.metadata["window_sequence_domain_indices"]]
    condition_origins = [tuple(int(v) for v in item) for item in loc_batch.metadata["window_condition_origins"]]
    roi_width = _condition_roi_extent(
        int(config.width),
        int(config.nat_grid_size[0] if isinstance(config.nat_grid_size, tuple) else config.nat_grid_size),
    )
    roi_height = _condition_roi_extent(
        int(config.height),
        int(config.nat_grid_size[1] if isinstance(config.nat_grid_size, tuple) else config.nat_grid_size),
    )
    vectors = []
    condition_sources = []
    for sample_idx, domain_index in enumerate(domain_indices):
        provider, condition_source, _ = _condition_provider(
            config,
            domain_index=domain_index,
            condition_providers=condition_providers,
        )
        x0, y0 = condition_origins[sample_idx]
        vector = provider.condition_vector_from_xy(
            x0=x0,
            y0=y0,
            height=roi_height,
            width=roi_width,
            device=image.device,
            dtype=image.dtype,
        )
        vectors.append(_match_feature_dim(vector, feature_dim=feature_dim))
        condition_sources.append(condition_source)
    condition = torch.stack(vectors, dim=0).contiguous()
    if bool(config.append_domain_onehot):
        onehot = torch.zeros((int(image.shape[0]), domain_count), dtype=image.dtype, device=image.device)
        for sample_idx, domain_index in enumerate(domain_indices):
            onehot[sample_idx, int(domain_index) % domain_count] = 1.0
        condition = torch.cat([condition, onehot], dim=1)
        domain_onehot_slice = (feature_dim, feature_dim + domain_count)
    else:
        domain_onehot_slice = None
    source_set = tuple(sorted(set(condition_sources)))
    condition_source = source_set[0] if len(source_set) == 1 else source_set
    return LocalizationTrainBatch(
        model_input=(image, condition),
        detect_tar=loc_batch.detect_tar,
        bkg_tar=loc_batch.bkg_tar,
        pxyz_tar=loc_batch.pxyz_tar,
        mask_tar=loc_batch.mask_tar,
        metadata={
            **loc_batch.metadata,
            "conditioning_mode": "film",
            "condition_source": condition_source,
            "condition_dim": int(condition.shape[1]),
            "condition_feature_dim": feature_dim,
            "condition_feature_order": condition_feature_order(
                include_domain_onehot=bool(config.append_domain_onehot),
                domain_count=domain_count,
            ),
            "condition_domain_onehot_slice": domain_onehot_slice,
            "domain_count": domain_count,
            "domain_index": domain_indices,
            "domain_name": loc_batch.metadata["window_sequence_domain_names"],
            **({} if condition_store_version is None else {"condition_store_version": int(condition_store_version)}),
        },
    )


def _finalize_pxyz_targets(config: OnlineBatchProviderConfig, target: torch.Tensor) -> tuple[torch.Tensor, tuple[str, ...]]:
    order = _normalize_pxyz_target_order(config.pxyz_target_order)
    if order == "legacy_iwae":
        converted = torch.stack((target[..., 3], target[..., 0], target[..., 1], target[..., 2]), dim=-1)
        if config.photon_scale is not None:
            converted[..., 0] = converted[..., 0] / max(float(config.photon_scale), 1e-12)
        if config.z_scale is not None:
            converted[..., 3] = converted[..., 3] / max(float(config.z_scale), 1e-12)
        return converted, LEGACY_IWAE_PXYZ_TARGET_ORDER
    return target, V03_PXYZ_TARGET_ORDER


def _pxyz_target_order_tuple(config: OnlineBatchProviderConfig) -> tuple[str, ...]:
    order = _normalize_pxyz_target_order(config.pxyz_target_order)
    if order == "legacy_iwae":
        return LEGACY_IWAE_PXYZ_TARGET_ORDER
    return V03_PXYZ_TARGET_ORDER


def _normalize_pxyz_target_order(value: str) -> str:
    key = str(value or "legacy_iwae").strip().lower()
    if key in {"legacy_iwae", "iwae", "old", "phot_xyz", "phot,x,y,z", "photons_x_y_z"}:
        return "legacy_iwae"
    if key in {"v03", "xyzph", "x,y,z,phot", "x_y_z_photons"}:
        return "v03"
    raise ValueError(f"unsupported online pxyz_target_order: {value!r}")


def _simulator_config(
    config: OnlineBatchProviderConfig,
    *,
    seed: int,
    field_origin: tuple[int, int],
    frames_per_sample: int,
    condition_provider: FullResZernikeConditioning,
) -> LocalizationSimulatorConfig:
    return LocalizationSimulatorConfig(
        batch_size=int(config.batch_size),
        frames_per_sample=int(frames_per_sample),
        height=int(config.height),
        width=int(config.width),
        emitters_per_sample=int(config.emitters_per_sample),
        seed=int(seed),
        photons=float(config.signal),
        background=float(config.background),
        psf_type=str(config.psf_type),
        pixel_size_nm_x=float(config.pixel_size_nm_x),
        pixel_size_nm_y=float(config.pixel_size_nm_y),
        wavelength_nm=float(config.wavelength_nm),
        na=float(config.na),
        npupil=int(config.npupil),
        vector_psf_size=int(config.vector_psf_size),
        vector_batch_size=int(config.vector_batch_size),
        emitter_density_um2=config.emitter_density_um2,
        lifetime_avg=float(config.lifetime_avg),
        warmup_frames=float(config.warmup_frames),
        photon_range=config.photon_range,
        photon_mean=config.photon_mean,
        photon_sigma=config.photon_sigma,
        background_range=config.background_range,
        background_scale=float(config.background_scale),
        z_range=config.z_range,
        field_origin_xy=field_origin,
        coeff_maps_nm=condition_provider.full_maps_nm if str(config.psf_type).lower() == "vector" else None,
        coeff_mode_order=tuple(condition_provider.mode_order) if str(config.psf_type).lower() == "vector" else (),
    )


def _build_sequence_window_batch(
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
        target, target_order = _finalize_pxyz_targets(config, sequence.pxyz_tar[target_idx])
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
    }
    return LocalizationTrainBatch(
        model_input=torch.stack(frames, dim=0).contiguous(),
        detect_tar=torch.stack(detect, dim=0).contiguous(),
        bkg_tar=torch.stack(bkg, dim=0).contiguous(),
        pxyz_tar=torch.stack(pxyz, dim=0).contiguous(),
        mask_tar=torch.stack(mask, dim=0).contiguous(),
        metadata=metadata,
    )


def _build_lut_like_sequence_batch(
    config: OnlineBatchProviderConfig,
    sim_config: LocalizationSimulatorConfig,
    *,
    epoch: int,
    step: int,
) -> LocalizationTrainBatch:
    window_size = int(config.channels)
    if window_size != 3:
        raise ValueError("lut sequence generation currently expects channels/window_size=3")
    center_count = int(config.batch_size)
    sequence_frames = center_count + window_size - 1
    frame_photons, pxyz_by_frame, mask_by_frame, bkg_by_frame, counts = _simulate_lut_like_sequence_frames(
        config,
        sim_config,
        frame_count=sequence_frames,
    )
    raw_adu = torch.poisson(
        frame_photons * float(config.camera_qe) + float(config.camera_spurious_charge)
    ) / max(float(config.camera_e_per_adu), 1e-12) + float(config.camera_baseline)
    frames = []
    detect = []
    bkg = []
    pxyz = []
    mask = []
    target_indices = []
    for start in range(center_count):
        target_idx = int(start) + window_size // 2
        frames.append(raw_adu[start : start + window_size])
        target = pxyz_by_frame[target_idx]
        active = mask_by_frame[target_idx]
        detect.append(_detect_from_v03_targets(target, active, height=int(config.height), width=int(config.width)))
        bkg.append(bkg_by_frame[target_idx] / max(float(config.background_scale), 1e-12))
        final_target, target_order = _finalize_pxyz_targets(config, target)
        pxyz.append(final_target)
        mask.append(active)
        target_indices.append(target_idx)
    return LocalizationTrainBatch(
        model_input=torch.stack(frames, dim=0).to(dtype=torch.float32).contiguous(),
        detect_tar=torch.stack(detect, dim=0).contiguous(),
        bkg_tar=torch.stack(bkg, dim=0).contiguous(),
        pxyz_tar=torch.stack(pxyz, dim=0).contiguous(),
        mask_tar=torch.stack(mask, dim=0).contiguous(),
        metadata={
            "epoch": int(epoch),
            "step": int(step),
            "seed": int(config.seed),
            "source": "native_online",
            "sequence_source": "native_lut_like_sequence_frames",
            "simulation_backend": "lut",
            "input_domain": "raw_adu",
            "frame_emitter_policy": "center_frame_only",
            "frames_per_sample": window_size,
            "sequence_simulated_frames": sequence_frames,
            "center_target_frame_indices": target_indices,
            "emitter_counts": [int(counts[idx]) for idx in target_indices],
            "background_scale": float(config.background_scale),
            "psf_type": str(config.psf_type).lower(),
            "pxyz_target_order": target_order,
        },
    )


def _simulate_lut_like_sequence_frames(
    config: OnlineBatchProviderConfig,
    sim_config: LocalizationSimulatorConfig,
    *,
    frame_count: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[int]]:
    height = int(config.height)
    width = int(config.width)
    rng = np.random.default_rng(int(sim_config.seed))
    target_active = _target_active_emitters(config)
    pool = max(
        1,
        int(math.ceil(target_active * (int(frame_count) + float(config.warmup_frames)) / (1.0 + max(float(config.lifetime_avg), 1e-6)))),
    )
    xy = rng.uniform(0.0, [float(width - 1), float(height - 1)], size=(pool, 2)).astype(np.float32)
    z_lo, z_hi = config.z_range if config.z_range is not None else (-0.6, 0.6)
    z = rng.uniform(float(z_lo), float(z_hi), size=pool).astype(np.float32)
    photons = _sample_lut_photons(config, rng=rng, count=pool)
    t0 = rng.uniform(-0.5 * float(config.warmup_frames), float(frame_count) + 0.5 * float(config.warmup_frames), size=pool)
    t1 = t0 + rng.exponential(max(float(config.lifetime_avg), 1e-6), size=pool)
    bg_values = _sample_lut_background(config, rng=rng, count=int(frame_count))
    frames = torch.as_tensor(bg_values[:, None, None], dtype=torch.float32).expand(int(frame_count), height, width).clone()
    events_by_frame: list[list[tuple[int, float]]] = [[] for _ in range(int(frame_count))]
    for frame_idx in range(int(frame_count)):
        overlap = np.minimum(t1, float(frame_idx + 1)) - np.maximum(t0, float(frame_idx))
        active = np.nonzero(overlap > 0)[0]
        for emitter_idx in active:
            events_by_frame[frame_idx].append((int(emitter_idx), float(overlap[emitter_idx])))

    max_count = max(1, max(len(items) for items in events_by_frame))
    pxyz = torch.zeros((int(frame_count), max_count, 4), dtype=torch.float32)
    mask = torch.zeros((int(frame_count), max_count), dtype=torch.bool)
    counts: list[int] = []
    if str(config.psf_type).strip().lower() != "vector":
        raise ValueError("online LUT simulation requires psf_type='vector'")
    vector_renderer = _build_vector_renderer(sim_config)
    for frame_idx, items in enumerate(events_by_frame):
        counts.append(len(items))
        if not items:
            continue
        emitter_idx = np.asarray([item[0] for item in items], dtype=np.int64)
        overlaps = np.asarray([item[1] for item in items], dtype=np.float32)
        xs = torch.as_tensor(xy[emitter_idx, 0], dtype=torch.float32)
        ys = torch.as_tensor(xy[emitter_idx, 1], dtype=torch.float32)
        z_t = torch.as_tensor(z[emitter_idx], dtype=torch.float32)
        photons_t = torch.as_tensor(photons[emitter_idx] * overlaps, dtype=torch.float32)
        frames[frame_idx] += vector_renderer.render_many(
            height=height,
            width=width,
            xs=xs,
            ys=ys,
            z_um=z_t,
            photons=photons_t,
        )
        count = len(items)
        pxyz[frame_idx, :count, 0] = xs
        pxyz[frame_idx, :count, 1] = ys
        pxyz[frame_idx, :count, 2] = z_t
        pxyz[frame_idx, :count, 3] = photons_t
        mask[frame_idx, :count] = True
    bkg = torch.as_tensor(bg_values[:, None, None], dtype=torch.float32).expand(int(frame_count), height, width).clone()
    return frames, pxyz, mask, bkg, counts


def _target_active_emitters(config: OnlineBatchProviderConfig) -> float:
    if config.emitter_density_um2 is None:
        return float(config.emitters_per_sample)
    area_um2 = (
        float(config.width)
        * float(config.pixel_size_nm_x)
        / 1000.0
        * float(config.height)
        * float(config.pixel_size_nm_y)
        / 1000.0
    )
    return float(config.emitter_density_um2) * area_um2


def _sample_lut_photons(config: OnlineBatchProviderConfig, *, rng: np.random.Generator, count: int) -> np.ndarray:
    if config.photon_mean is not None and config.photon_sigma is not None:
        values = rng.normal(float(config.photon_mean), float(config.photon_sigma), size=int(count)).astype(np.float32)
    else:
        lo, hi = config.photon_range if config.photon_range is not None else (float(config.signal), float(config.signal))
        values = rng.uniform(float(lo), float(hi), size=int(count)).astype(np.float32)
    if config.photon_range is not None:
        lo, hi = config.photon_range
        values = np.clip(values, float(lo), float(hi))
    return values.astype(np.float32, copy=False)


def _sample_lut_background(config: OnlineBatchProviderConfig, *, rng: np.random.Generator, count: int) -> np.ndarray:
    if config.background_range is None:
        return np.full((int(count),), float(config.background), dtype=np.float32)
    lo, hi = config.background_range
    return rng.uniform(float(lo), float(hi), size=int(count)).astype(np.float32)


def _detect_from_v03_targets(target: torch.Tensor, mask: torch.Tensor, *, height: int, width: int) -> torch.Tensor:
    detect = torch.zeros((int(height), int(width)), dtype=torch.float32)
    if not bool(mask.any()):
        return detect
    active = target[mask]
    cols = torch.round(active[:, 0]).to(dtype=torch.long).clamp_(0, int(width) - 1)
    rows = torch.round(active[:, 1]).to(dtype=torch.long).clamp_(0, int(height) - 1)
    detect[rows, cols] = 1.0
    return detect


def _condition_feature_dim(config: OnlineBatchProviderConfig) -> int:
    if int(config.condition_feature_dim) > 0:
        return int(config.condition_feature_dim)
    if int(config.condition_dim) > 0:
        domain_terms = int(config.domain_count) if bool(config.append_domain_onehot) else 0
        return max(0, int(config.condition_dim) - domain_terms)
    return 8


def _domain_index(config: OnlineBatchProviderConfig, *, global_step: int) -> int:
    domain_count = int(config.domain_count)
    if domain_count <= 0:
        raise ValueError("domain_count must be positive")
    if str(config.domain_balance_mode) == "alternate_step":
        return int(global_step) % domain_count
    return 0


def _build_condition_features(
    config: OnlineBatchProviderConfig,
    *,
    global_step: int,
    domain_index: int,
    image: torch.Tensor,
    condition_providers: tuple[tuple[str | None, FullResZernikeConditioning], ...] | None,
) -> tuple[torch.Tensor, str, str | None]:
    feature_dim = _condition_feature_dim(config)
    provider, condition_source, domain_name = _condition_provider(
        config,
        domain_index=domain_index,
        condition_providers=condition_providers,
    )
    roi_width = _condition_roi_extent(int(config.width), int(config.nat_grid_size[0] if isinstance(config.nat_grid_size, tuple) else config.nat_grid_size))
    roi_height = _condition_roi_extent(int(config.height), int(config.nat_grid_size[1] if isinstance(config.nat_grid_size, tuple) else config.nat_grid_size))
    vectors = []
    for sample_idx in range(int(config.batch_size)):
        x0, y0 = _condition_origin(config, sample_index=sample_idx, global_step=global_step, roi_width=roi_width, roi_height=roi_height)
        vector = provider.condition_vector_from_xy(
            x0=x0,
            y0=y0,
            height=roi_height,
            width=roi_width,
            device=image.device,
            dtype=image.dtype,
        )
        vectors.append(_match_feature_dim(vector, feature_dim=feature_dim))
    return torch.stack(vectors, dim=0).contiguous(), condition_source, domain_name


def _condition_provider(
    config: OnlineBatchProviderConfig,
    *,
    domain_index: int,
    condition_providers: tuple[tuple[str | None, FullResZernikeConditioning], ...] | None,
) -> tuple[FullResZernikeConditioning, str, str | None]:
    if condition_providers:
        domain_name, provider = condition_providers[int(domain_index) % len(condition_providers)]
        return provider, "dual_domain_coeff_maps", domain_name
    return build_default_conditioning_maps(width=int(config.width), height=int(config.height)), "synthetic_nat", None


def _load_condition_providers(
    config: OnlineBatchProviderConfig,
) -> tuple[tuple[str | None, FullResZernikeConditioning], ...] | None:
    return _load_conditioning_providers(config.dual_domain_coeff_maps)


def _condition_roi_extent(axis_size: int, grid_size: int) -> int:
    return max(1, min(int(axis_size), int(axis_size) // max(1, int(grid_size))))


def _condition_origin(
    config: OnlineBatchProviderConfig,
    *,
    sample_index: int,
    global_step: int,
    roi_width: int,
    roi_height: int,
) -> tuple[int, int]:
    grid = config.nat_grid_size
    if isinstance(grid, tuple):
        grid_x = max(1, int(grid[0]))
        grid_y = max(1, int(grid[1] if len(grid) > 1 else grid[0]))
    else:
        grid_x = grid_y = max(1, int(grid))
    cell = (int(global_step) * int(config.batch_size) + int(sample_index)) % (grid_x * grid_y)
    col = cell % grid_x
    row = cell // grid_x
    max_x0 = max(0, int(config.width) - int(roi_width))
    max_y0 = max(0, int(config.height) - int(roi_height))
    x0 = 0 if grid_x <= 1 else round(float(col) * float(max_x0) / float(grid_x - 1))
    y0 = 0 if grid_y <= 1 else round(float(row) * float(max_y0) / float(grid_y - 1))
    return int(x0), int(y0)


def _match_feature_dim(vector: torch.Tensor, *, feature_dim: int) -> torch.Tensor:
    if int(feature_dim) == int(vector.shape[0]):
        return vector
    if int(feature_dim) < int(vector.shape[0]):
        return vector[: int(feature_dim)].contiguous()
    out = torch.zeros((int(feature_dim),), dtype=vector.dtype, device=vector.device)
    out[: int(vector.shape[0])] = vector
    return out
