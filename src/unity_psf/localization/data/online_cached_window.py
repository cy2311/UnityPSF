"""Cached temporal-window batch assembly."""

from __future__ import annotations

import torch
from unity_psf.localization.conditioning import FullResZernikeConditioning
from unity_psf.localization.training_adapter import LocalizationTrainBatch
from unity_psf.runtime.profiling import time_block
from .online_types import OnlineBatchProviderConfig
from .online_conditioning import attach_cached_window_film_conditioning
from .online_conditioning import attach_cached_window_precomputed_film_conditioning
from .online_targets import detect_from_v03_targets
from .online_targets import finalize_pxyz_targets
from .online_validation import online_item_seed
from .online_targets import pad_pxyz_targets
from .online_targets import pxyz_target_order_tuple
from .online_targets import target_contract_for_config


def slice_cached_window_batch(
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
    condition_origin_indices = []
    condition_origin_bank_sizes = []
    precomputed_condition_vectors = []
    precomputed_condition_sources = []
    sequence_global_steps = []
    with time_block("cached_window_slice_loop"):
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
            precomputed_offset = int(center_idx) - int(sequence.get("precomputed_center_start", -10**9))
            precomputed_windows = sequence.get("precomputed_window_frames")
            precomputed_detect = sequence.get("precomputed_detect")
            precomputed_bkg = sequence.get("precomputed_bkg")
            precomputed_pxyz = sequence.get("precomputed_pxyz")
            precomputed_mask = sequence.get("precomputed_mask")
            if (
                isinstance(precomputed_windows, torch.Tensor)
                and isinstance(precomputed_detect, torch.Tensor)
                and isinstance(precomputed_bkg, torch.Tensor)
                and isinstance(precomputed_pxyz, tuple)
                and isinstance(precomputed_mask, tuple)
                and 0 <= precomputed_offset < int(precomputed_windows.shape[0])
            ):
                frames.append(precomputed_windows[precomputed_offset])
                detect.append(precomputed_detect[precomputed_offset])
                bkg.append(precomputed_bkg[precomputed_offset])
                pxyz.append(precomputed_pxyz[precomputed_offset])
                mask.append(precomputed_mask[precomputed_offset])
                target_order = tuple(sequence.get("precomputed_target_order", pxyz_target_order_tuple(config)))
                active_count = int(precomputed_mask[precomputed_offset].sum().item())
            else:
                target = sequence_pxyz[int(center_idx)]
                active = sequence_mask[int(center_idx)]
                frames.append(sequence_frames[start:end])
                detect.append(detect_from_v03_targets(target, active, height=int(config.height), width=int(config.width)))
                bkg.append(sequence_bkg[int(center_idx)])
                final_target, target_order = finalize_pxyz_targets(config, target[active])
                pxyz.append(final_target)
                mask.append(torch.ones((int(final_target.shape[0]),), dtype=torch.bool, device=final_target.device))
                active_count = int(active.sum().item())
            sequence_counts = sequence["counts"]
            counts.append(int(sequence_counts[int(center_idx)] if int(center_idx) < len(sequence_counts) else active_count))
            target_indices.append(int(center_idx))
            sequence_indices.append(int(sequence.get("sequence_idx", sequence_idx)))
            frame_indices.append(tuple(range(start, end)))
            global_center_indices.append(int(global_center_idx))
            domain_indices.append(int(sequence["domain_index"]))
            domain_names.append(None if sequence.get("domain_name") is None else str(sequence["domain_name"]))
            condition_origins.append(tuple(int(item) for item in sequence["field_origin"]))
            condition_origin_indices.append(
                None if sequence.get("field_origin_index") is None else int(sequence["field_origin_index"])
            )
            condition_origin_bank_sizes.append(
                None if sequence.get("field_origin_bank_size") is None else int(sequence["field_origin_bank_size"])
            )
            if isinstance(sequence.get("precomputed_condition_vector"), torch.Tensor):
                precomputed_condition_vectors.append(sequence["precomputed_condition_vector"])
                precomputed_condition_sources.append(str(sequence.get("precomputed_condition_source", "unknown")))
            sequence_global_steps.append(int(sequence["global_step"]))

    unique_sequences: list[dict[str, object]] = []
    seen_sequence_keys: set[object] = set()
    with time_block("cached_window_unique_sequence_aggregate"):
        for sequence in sequences:
            key: object = sequence.get("sequence_idx", id(sequence))
            if key in seen_sequence_keys:
                continue
            seen_sequence_keys.add(key)
            unique_sequences.append(sequence)

        base_metadata = dict(sequences[0].get("metadata", {})) if sequences else {}
        aggregate_metadata = {
            "renderer_cache_hits": sum(int(dict(sequence.get("metadata", {})).get("renderer_cache_hits", 0)) for sequence in unique_sequences),
            "renderer_cache_misses": sum(int(dict(sequence.get("metadata", {})).get("renderer_cache_misses", 0)) for sequence in unique_sequences),
            "render_many_calls": sum(int(dict(sequence.get("metadata", {})).get("render_many_calls", 0)) for sequence in unique_sequences),
            "render_frames_calls": sum(int(dict(sequence.get("metadata", {})).get("render_frames_calls", 0)) for sequence in unique_sequences),
        }
    with time_block("cached_window_pad_pxyz_targets"):
        padded_pxyz, padded_mask = pad_pxyz_targets(pxyz, mask)
    with time_block("cached_window_stack_tensors"):
        model_input = torch.stack(frames, dim=0).to(dtype=torch.float32).contiguous()
        detect_tar = torch.stack(detect, dim=0).contiguous()
        bkg_tar = torch.stack(bkg, dim=0).contiguous()
        pxyz_tar = padded_pxyz.contiguous()
        mask_tar = padded_mask.contiguous()
    loc_batch = LocalizationTrainBatch(
        model_input=model_input,
        detect_tar=detect_tar,
        bkg_tar=bkg_tar,
        pxyz_tar=pxyz_tar,
        mask_tar=mask_tar,
        metadata={
            **base_metadata,
            **aggregate_metadata,
            "epoch": int(epoch),
            "step": int(batch_idx) + 1,
            "seed": online_item_seed(
                data_seed=int(config.seed),
                epoch=int(epoch),
                item_index=0,
                items_per_epoch=int(sequence_count),
            ),
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
            "window_condition_origin_indices": condition_origin_indices,
            "window_condition_origin_bank_sizes": condition_origin_bank_sizes,
            "window_sequence_global_steps": sequence_global_steps,
            "window_frame_indices": frame_indices,
            "window_global_center_indices": global_center_indices,
            "emitter_counts": counts,
            "pxyz_target_order": target_order,
            "target_contract": target_contract_for_config(config).as_dict(),
        },
    )
    if str(config.conditioning_mode) == "film":
        with time_block("cached_window_attach_film_conditioning"):
            if len(precomputed_condition_vectors) == len(frames):
                loc_batch = attach_cached_window_precomputed_film_conditioning(
                    config,
                    loc_batch,
                    condition_vectors=precomputed_condition_vectors,
                    condition_sources=precomputed_condition_sources,
                    condition_store_version=condition_store_version,
                )
            else:
                loc_batch = attach_cached_window_film_conditioning(
                    config,
                    loc_batch,
                    condition_providers=condition_providers,
                    condition_store_version=condition_store_version,
                )
    return loc_batch
