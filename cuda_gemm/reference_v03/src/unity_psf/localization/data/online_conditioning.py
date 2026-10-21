"""Condition vectors, providers and film batch attachment."""

from __future__ import annotations

import torch
from unity_psf.localization.conditioning import FullResZernikeConditioning
from unity_psf.localization.conditioning import condition_feature_order
from unity_psf.localization.conditioning import load_conditioning_providers
from unity_psf.localization.conditioning import build_astigmatism_anchor_conditioning_maps
from unity_psf.localization.conditioning import build_circular_psf_anchor_conditioning_maps
from unity_psf.localization.conditioning import build_default_conditioning_maps
from unity_psf.localization.conditioning import build_zero_conditioning_maps
from unity_psf.localization.ppg_conditioning import PPGFieldConditioning
from unity_psf.localization.training_adapter import LocalizationTrainBatch
from unity_psf.runtime.profiling import time_block
from .online_types import OnlineBatchProviderConfig
from .online_geometry import condition_origin
from .online_geometry import condition_roi_extent


_CONDITION_FIELD_ALIASES = {
    "zernike_0": "zernike_nm_mean:n2_m0",
    "zernike_1": "zernike_nm_mean:n3_m1",
    "field_x": "x_norm",
    "field_y": "y_norm",
}


def condition_feature_order_for_config(config) -> tuple[str, ...]:
    fields = tuple(str(item) for item in config.condition_fields)
    names = list(fields) if fields else list(condition_feature_order(include_domain_onehot=False, domain_count=0))
    if bool(config.append_domain_onehot):
        names.extend(f"domain_onehot:{idx}" for idx in range(int(config.domain_count)))
    return tuple(names)


def condition_feature_dim(config) -> int:
    if int(config.condition_feature_dim) > 0:
        return int(config.condition_feature_dim)
    if int(config.condition_dim) > 0:
        domain_terms = int(config.domain_count) if bool(config.append_domain_onehot) else 0
        return max(0, int(config.condition_dim) - domain_terms)
    return 8


def condition_vector_for_config(config, vector: torch.Tensor) -> torch.Tensor:
    fields = tuple(str(item) for item in config.condition_fields)
    if not fields:
        return match_feature_dim(vector, feature_dim=condition_feature_dim(config))
    if len(fields) != condition_feature_dim(config):
        raise ValueError("condition_fields length must equal condition_feature_dim")
    if any(field.startswith("ppg_pca_") for field in fields):
        selected = []
        for field in fields:
            if field == "field_x":
                selected.append(0)
            elif field == "field_y":
                selected.append(1)
            elif field.startswith("ppg_pca_"):
                try:
                    index = int(field.removeprefix("ppg_pca_"))
                except ValueError as exc:
                    raise ValueError(f"unsupported condition field {field!r}") from exc
                if not 0 <= index < 4:
                    raise ValueError(f"unsupported condition field {field!r}")
                selected.append(2 + index)
            else:
                raise ValueError(f"unsupported PPG condition field {field!r}")
        if max(selected, default=-1) >= int(vector.shape[0]):
            raise ValueError("PPG condition provider returned fewer features than configured")
        return vector[selected].contiguous()
    source_names = condition_feature_order(include_domain_onehot=False, domain_count=0)
    source_indices = {name: index for index, name in enumerate(source_names)}
    selected = []
    for field in fields:
        source_name = _CONDITION_FIELD_ALIASES.get(field, field)
        if source_name not in source_indices:
            raise ValueError(f"unsupported condition field {field!r}")
        selected.append(source_indices[source_name])
    if max(selected, default=-1) >= int(vector.shape[0]):
        raise ValueError("condition provider returned fewer features than the configured condition fields")
    return vector[selected].contiguous()


def domain_index(config, *, global_step: int) -> int:
    domain_count = int(config.domain_count)
    if domain_count <= 0:
        raise ValueError("domain_count must be positive")
    if str(config.domain_balance_mode) == "alternate_step":
        return int(global_step) % domain_count
    return 0


resolve_domain_index = domain_index


def match_feature_dim(vector: torch.Tensor, *, feature_dim: int) -> torch.Tensor:
    if int(feature_dim) == int(vector.shape[0]):
        return vector
    if int(feature_dim) < int(vector.shape[0]):
        return vector[: int(feature_dim)].contiguous()
    out = torch.zeros((int(feature_dim),), dtype=vector.dtype, device=vector.device)
    out[: int(vector.shape[0])] = vector
    return out


def attach_film_conditioning(
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
    feature_dim = condition_feature_dim(config)
    domain_count = int(config.domain_count)
    selected_domain = resolve_domain_index(config, global_step=global_step) if domain_index is None else int(domain_index)
    condition, condition_source, domain_name = build_condition_features(
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
            "condition_feature_order": condition_feature_order_for_config(config),
            "condition_domain_onehot_slice": domain_onehot_slice,
            "domain_count": domain_count,
            "domain_index": selected_domain,
            **({} if condition_store_version is None else {"condition_store_version": int(condition_store_version)}),
            **({} if domain_name is None else {"domain_name": domain_name}),
        },
    )


def attach_cached_window_film_conditioning(
    config: OnlineBatchProviderConfig,
    loc_batch: LocalizationTrainBatch,
    *,
    condition_providers: tuple[tuple[str | None, FullResZernikeConditioning], ...] | None,
    condition_store_version: int | None = None,
) -> LocalizationTrainBatch:
    image = loc_batch.model_input
    if not isinstance(image, torch.Tensor):
        raise TypeError("cached-window film conditioning expects tensor image input before conditioning is attached")
    feature_dim = condition_feature_dim(config)
    domain_count = int(config.domain_count)
    domain_indices = [int(item) for item in loc_batch.metadata["window_sequence_domain_indices"]]
    condition_origins = [tuple(int(v) for v in item) for item in loc_batch.metadata["window_condition_origins"]]
    roi_width = int(config.width)
    roi_height = int(config.height)
    vectors = []
    condition_sources = []
    with time_block("cached_window_condition_vector_loop"):
        for sample_idx, domain_index in enumerate(domain_indices):
            provider, condition_source, _ = select_condition_provider(
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
            vectors.append(condition_vector_for_config(config, vector))
            condition_sources.append(condition_source)
    with time_block("cached_window_condition_stack_onehot"):
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
            "condition_feature_order": condition_feature_order_for_config(config),
            "condition_domain_onehot_slice": domain_onehot_slice,
            "domain_count": domain_count,
            "domain_index": domain_indices,
            "domain_name": loc_batch.metadata["window_sequence_domain_names"],
            **({} if condition_store_version is None else {"condition_store_version": int(condition_store_version)}),
        },
    )


def attach_cached_window_precomputed_film_conditioning(
    config: OnlineBatchProviderConfig,
    loc_batch: LocalizationTrainBatch,
    *,
    condition_vectors: list[torch.Tensor],
    condition_sources: list[str],
    condition_store_version: int | None = None,
) -> LocalizationTrainBatch:
    image = loc_batch.model_input
    if not isinstance(image, torch.Tensor):
        raise TypeError("cached-window film conditioning expects tensor image input before conditioning is attached")
    feature_dim = condition_feature_dim(config)
    domain_count = int(config.domain_count)
    domain_indices = [int(item) for item in loc_batch.metadata["window_sequence_domain_indices"]]
    with time_block("cached_window_condition_stack_onehot"):
        vectors = [
            match_feature_dim(vector.to(device=image.device, dtype=image.dtype), feature_dim=feature_dim)
            for vector in condition_vectors
        ]
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
            "condition_feature_order": condition_feature_order_for_config(config),
            "condition_domain_onehot_slice": domain_onehot_slice,
            "domain_count": domain_count,
            "domain_index": domain_indices,
            "domain_name": loc_batch.metadata["window_sequence_domain_names"],
            **({} if condition_store_version is None else {"condition_store_version": int(condition_store_version)}),
        },
    )


def build_condition_features(
    config: OnlineBatchProviderConfig,
    *,
    global_step: int,
    domain_index: int,
    image: torch.Tensor,
    condition_providers: tuple[tuple[str | None, FullResZernikeConditioning], ...] | None,
) -> tuple[torch.Tensor, str, str | None]:
    provider, condition_source, domain_name = select_condition_provider(
        config,
        domain_index=domain_index,
        condition_providers=condition_providers,
    )
    roi_width = condition_roi_extent(int(config.width), int(config.nat_grid_size[0] if isinstance(config.nat_grid_size, tuple) else config.nat_grid_size))
    roi_height = condition_roi_extent(int(config.height), int(config.nat_grid_size[1] if isinstance(config.nat_grid_size, tuple) else config.nat_grid_size))
    vectors = []
    for sample_idx in range(int(config.batch_size)):
        x0, y0 = condition_origin(config, sample_index=sample_idx, global_step=global_step, roi_width=roi_width, roi_height=roi_height)
        vector = provider.condition_vector_from_xy(
            x0=x0,
            y0=y0,
            height=roi_height,
            width=roi_width,
            device=image.device,
            dtype=image.dtype,
        )
        vectors.append(condition_vector_for_config(config, vector))
    return torch.stack(vectors, dim=0).contiguous(), condition_source, domain_name


def select_condition_provider(
    config: OnlineBatchProviderConfig,
    *,
    domain_index: int,
    condition_providers: tuple[tuple[str | None, FullResZernikeConditioning], ...] | None,
) -> tuple[FullResZernikeConditioning, str, str | None]:
    if condition_providers:
        domain_name, provider = condition_providers[int(domain_index) % len(condition_providers)]
        source = conditioning_profile_source(config) if domain_name is None else "dual_domain_coeff_maps"
        return provider, source, domain_name
    provider = default_condition_provider(config)
    return provider, conditioning_profile_source(config), None


def conditioning_profile_source(config: OnlineBatchProviderConfig) -> str:
    profile = str(config.conditioning_profile).strip().lower().replace("-", "_")
    if profile.startswith("circular_2d") or profile.startswith("circular_psf"):
        return "circular_psf_anchor_profile"
    if profile in {"zero", "zero_aberration", "focal_2d"}:
        return "zero_profile"
    if profile in {"coefficient_map", "coefficient_map_absolute"}:
        return "coefficient_map"
    if profile.startswith("astigmatism_660nm"):
        return "astigmatism_anchor_profile"
    return "synthetic_nat"


def default_condition_provider(config: OnlineBatchProviderConfig) -> FullResZernikeConditioning:
    if str(config.psf_type).strip().lower() == "ppg_direct":
        if not config.ppg_map_path:
            raise ValueError("ppg_direct conditioning requires ppg_map_path")
        return PPGFieldConditioning.from_npz(config.ppg_map_path)
    profile = str(config.conditioning_profile).strip().lower().replace("-", "_")
    if profile.startswith("circular_2d") or profile.startswith("circular_psf"):
        return build_circular_psf_anchor_conditioning_maps(
            width=int(config.width),
            height=int(config.height),
            profile_name=profile,
        )
    if profile in {"default", "default_nat", "synthetic_nat"}:
        return build_default_conditioning_maps(width=int(config.width), height=int(config.height))
    if profile in {"zero", "zero_aberration", "focal_2d"}:
        return build_zero_conditioning_maps(width=int(config.width), height=int(config.height))
    if profile in {"coefficient_map", "coefficient_map_absolute"}:
        return build_zero_conditioning_maps(width=int(config.width), height=int(config.height))
    return build_astigmatism_anchor_conditioning_maps(
        width=int(config.width),
        height=int(config.height),
        profile_name=profile,
    )


def load_condition_providers(
    config: OnlineBatchProviderConfig,
) -> tuple[tuple[str | None, FullResZernikeConditioning], ...] | None:
    return load_conditioning_providers(config.dual_domain_coeff_maps)
