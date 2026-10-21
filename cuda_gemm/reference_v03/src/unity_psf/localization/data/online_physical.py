"""Physical renderer cache, lifecycle and simulator configuration."""

from __future__ import annotations

from unity_psf.localization.conditioning import FullResZernikeConditioning
from unity_psf.localization.simulator import LocalizationSimulatorConfig
from unity_psf.localization.simulator import build_emitter_renderer
from .online_types import OnlineBatchProviderConfig


class VectorRendererCache:
    def __init__(self) -> None:
        self._items: dict[tuple[object, ...], object] = {}
        self.hits = 0
        self.misses = 0
        self.generation = 0

    def get(self, config: LocalizationSimulatorConfig, *, physical_model_version: int | None = None):
        key = renderer_cache_key(config, physical_model_version=physical_model_version)
        if key in self._items:
            self.hits += 1
            return self._items[key]
        renderer = build_emitter_renderer(config)
        self._items[key] = renderer
        self.misses += 1
        return renderer

    def clear(self) -> None:
        self._items.clear()
        self.generation += 1


def renderer_cache_key(
    config: LocalizationSimulatorConfig,
    *,
    physical_model_version: int | None = None,
) -> tuple[object, ...]:
    maps = config.coeff_maps_nm
    maps_key = None if maps is None else (int(maps.data_ptr()), tuple(int(v) for v in maps.shape), str(maps.dtype))
    carrier = config.pupil_carrier_complex
    carrier_key = (
        None
        if carrier is None
        else (int(carrier.data_ptr()), tuple(int(v) for v in carrier.shape), str(carrier.dtype))
    )
    return (
        None if physical_model_version is None else int(physical_model_version),
        str(config.psf_type).strip().lower(),
        str(config.simulation_backend).strip().lower(),
        config.empirical_psf_path,
        config.empirical_psf_channel,
        config.empirical_psf_focus_index,
        config.ppg_map_path,
        maps_key,
        carrier_key,
        tuple((int(n), int(m)) for n, m in config.coeff_mode_order),
        float(config.na),
        float(config.wavelength_nm),
        float(config.pixel_size_nm_x),
        float(config.pixel_size_nm_y),
        int(config.npupil),
        int(config.vector_psf_size),
        int(config.vector_batch_size),
        float(config.refmed),
        float(config.refcov),
        float(config.refimm),
        float(config.objstage0),
        None if config.zemit0 is None else float(config.zemit0),
        tuple(float(v) for v in config.otf_rescale_xy),
        None if config.fixed_astig_anchor_nm is None else float(config.fixed_astig_anchor_nm),
        None if config.circular_psf_radius_nm is None else float(config.circular_psf_radius_nm),
    )


class PhysicalVersionedRendererLifecycle:
    """Invalidate renderer and cached-window state when physical maps change."""

    def __init__(self, renderer_cache: VectorRendererCache) -> None:
        self.renderer_cache = renderer_cache
        self.active_version: int | None = None
        self.cache_generation = 0

    def activate(
        self,
        *,
        version: int | None,
        condition_providers: tuple[tuple[str | None, FullResZernikeConditioning], ...] | None = None,
    ) -> None:
        del condition_providers
        version_key = None if version is None else int(version)
        if version_key == self.active_version:
            return
        if self.active_version is not None or version_key is not None:
            self.renderer_cache.clear()
            self.cache_generation += 1
        self.active_version = version_key

    def invalidate(self) -> None:
        self.renderer_cache.clear()
        self.cache_generation += 1


def simulator_config(
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
        refmed=float(config.refmed),
        refcov=float(config.refcov),
        refimm=float(config.refimm),
        objstage0=float(config.objstage0),
        otf_rescale_xy=tuple(float(value) for value in config.otf_rescale_xy),
        npupil=int(config.npupil),
        vector_psf_size=int(config.vector_psf_size),
        vector_batch_size=int(config.vector_batch_size),
        zemit0=None if config.zemit0 is None else float(config.zemit0),
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
        fixed_astig_anchor_nm=(
            None if config.fixed_astig_anchor_nm is None else float(config.fixed_astig_anchor_nm)
        ),
        circular_psf_radius_nm=(
            None if config.circular_psf_radius_nm is None else float(config.circular_psf_radius_nm)
        ),
        anchor_nm=(
            None
            if getattr(condition_provider, "anchor_nm", None) is None
            else float(condition_provider.anchor_nm)
        ),
        anchor_profile=str(config.conditioning_profile),
        pupil_carrier_complex=config.pupil_carrier_complex if str(config.psf_type).lower() == "vector" else None,
        empirical_psf_path=config.empirical_psf_path,
        empirical_psf_channel=config.empirical_psf_channel,
        empirical_psf_focus_index=config.empirical_psf_focus_index,
        ppg_map_path=config.ppg_map_path,
        output_device=str(config.simulation_output_device),
        simulation_backend=str(config.simulation_backend),
    )
