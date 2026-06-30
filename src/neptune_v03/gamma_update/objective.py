from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import torch
import torch.nn.functional as F

from neptune_v03.localization.posterior import DetectionPosteriorSamples
from neptune_v03.optics.nat_field import NATFieldConfig, default_order1_config, evaluate_zernike_from_roi_positions_torch
from neptune_v03.optics.vector_psf import VectorPSFParams, build_vector_psf_context, noll_to_nm, render_vector_psf_bank


@dataclass(frozen=True)
class GammaProjectionObjectiveConfig:
    image_size_x: int = 1024
    image_size_y: int = 1024
    pixel_size_x_nm: float = 95.0
    pixel_size_y_nm: float = 95.0
    patch_size_px: int = 15
    npupil: int = 64
    NA: float = 1.4
    wavelength_nm: float = 660.0
    refmed: float = 1.518
    refcov: float = 1.518
    refimm: float = 1.518
    objstage0: float = 0.0
    otf_rescale_xy: tuple[float, float] = (0.0, 0.0)
    renderer_batch_size: int = 64
    over_cut_px: int = 0
    eps: float = 1e-6
    base_coeff_maps: tuple[tuple[str, str], ...] = ()
    objective_mode: str = "poisson_nll"
    projection_sample_batch_size: int = 16
    projection_emitter_chunk_size: int = 1024


class GammaProjectionObjective:
    """ROI-bank vector-PSF Poisson projection objective for NAT gamma updates."""

    def __init__(
        self,
        config: GammaProjectionObjectiveConfig | None = None,
        *,
        nat_config: NATFieldConfig | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        self.config = GammaProjectionObjectiveConfig() if config is None else config
        self.device = torch.device("cpu" if device is None else device)
        self.nat_config = nat_config or default_order1_config(
            img_size_x=int(self.config.image_size_x),
            img_size_y=int(self.config.image_size_y),
            pixel_size_x_nm=float(self.config.pixel_size_x_nm),
            pixel_size_y_nm=float(self.config.pixel_size_y_nm),
        )
        self.ctx = build_vector_psf_context(
            NA=float(self.config.NA),
            wavelength_nm=float(self.config.wavelength_nm),
            pixel_size_nm_x=float(self.config.pixel_size_x_nm),
            pixel_size_nm_y=float(self.config.pixel_size_y_nm),
            noll_indices=_nat_aberration_noll_indices(self.nat_config),
            params=VectorPSFParams(
                npupil=int(self.config.npupil),
                psf_size=int(self.config.patch_size_px),
                refmed=float(self.config.refmed),
                refcov=float(self.config.refcov),
                refimm=float(self.config.refimm),
                objstage0=float(self.config.objstage0),
                otf_rescale_xy=tuple(float(v) for v in self.config.otf_rescale_xy),
                batch_size=int(self.config.renderer_batch_size),
            ),
            device=self.device,
        )
        self.last_metrics: dict[str, float] = {}
        self.base_maps_by_domain = _load_base_coeff_maps(
            self.config.base_coeff_maps,
            expected_modes=tuple(self.nat_config.aberrations),
            device=self.device,
        )

    @property
    def gamma_size(self) -> int:
        return len(self.nat_config.gammas)

    def initial_gamma(self, *, dtype: torch.dtype = torch.float32) -> torch.nn.Parameter:
        return torch.nn.Parameter(torch.zeros((self.gamma_size,), dtype=dtype, device=self.device))

    def __call__(
        self,
        *,
        gamma: torch.Tensor,
        samples: DetectionPosteriorSamples,
        raw_frames: torch.Tensor,
        background: torch.Tensor,
        roi_origin_xy_px: torch.Tensor | None = None,
        domain_names: list[str] | tuple[str, ...] | None = None,
    ) -> torch.Tensor:
        gamma_t = gamma.to(device=self.device, dtype=torch.float32).reshape(-1)
        if int(gamma_t.numel()) != self.gamma_size:
            raise ValueError(f"gamma length must be {self.gamma_size}, got {int(gamma_t.numel())}")

        raw_all = torch.as_tensor(raw_frames, dtype=torch.float32)
        if raw_all.ndim == 4:
            raw_all = raw_all[:, raw_all.shape[1] // 2]
        if raw_all.ndim != 3:
            raise ValueError(f"raw_frames must have shape (B,H,W) or (B,T,H,W), got {tuple(raw_all.shape)}")
        bkg_all = torch.as_tensor(background, dtype=torch.float32)
        if bkg_all.ndim == 4:
            bkg_all = bkg_all[:, bkg_all.shape[1] // 2]
        if bkg_all.ndim == 2:
            bkg_all = bkg_all.unsqueeze(0).expand(raw_all.shape[0], -1, -1)
        if bkg_all.shape != raw_all.shape:
            raise ValueError(f"background shape {tuple(bkg_all.shape)} must match raw shape {tuple(raw_all.shape)}")

        xyzph_all = samples.xyzph.detach()
        mask_all = samples.mask.detach()
        if xyzph_all.ndim != 3 or int(xyzph_all.shape[-1]) != 4:
            raise ValueError(f"samples.xyzph must have shape (B,N,4), got {tuple(xyzph_all.shape)}")
        if int(xyzph_all.shape[0]) != int(raw_all.shape[0]):
            raise ValueError("samples batch size must match raw_frames batch size")

        objective_mode = str(self.config.objective_mode).strip().lower()
        if objective_mode not in {"poisson_nll", "importance_wake"}:
            raise ValueError(f"Unsupported gamma projection objective_mode: {self.config.objective_mode!r}")
        sample_chunk = max(1, int(self.config.projection_sample_batch_size))
        total_nll = gamma_t.new_tensor(0.0)
        total_pixels = 0
        log_p_chunks: list[torch.Tensor] = []
        valid_emitters = 0
        projected_photons = gamma_t.new_tensor(0.0)
        background_sum = gamma_t.new_tensor(0.0)
        background_pixels = 0
        for start in range(0, int(raw_all.shape[0]), sample_chunk):
            stop = min(start + sample_chunk, int(raw_all.shape[0]))
            chunk_domain_names = None if domain_names is None else list(domain_names[start:stop])
            chunk_origin = None
            if roi_origin_xy_px is not None:
                chunk_origin = torch.as_tensor(roi_origin_xy_px, dtype=torch.float32)[start:stop]
            chunk = self._chunk_log_p(
                gamma_t=gamma_t,
                raw_frames=raw_all[start:stop],
                background=bkg_all[start:stop],
                xyzph=xyzph_all[start:stop],
                mask=mask_all[start:stop],
                roi_origin_xy_px=chunk_origin,
                domain_names=chunk_domain_names,
            )
            total_nll = total_nll + chunk["nll_sum"]
            total_pixels += int(chunk["pixel_count"])
            log_p_chunks.append(chunk["log_p_per_sample"])
            valid_emitters += int(chunk["valid_emitters"])
            projected_photons = projected_photons + chunk["projected_photons"]
            background_sum = background_sum + chunk["background_sum"]
            background_pixels += int(chunk["background_pixels"])

        log_p_per_sample = torch.cat(log_p_chunks, dim=0) if log_p_chunks else gamma_t.new_zeros((0,))
        poisson_loss = total_nll / max(1, int(total_pixels))
        loss = poisson_loss
        importance_metrics = _importance_wake_loss_and_metrics(
            log_p_per_sample,
            metadata=samples.metadata,
            sample_count=int(raw_all.shape[0]),
        )
        if objective_mode == "importance_wake":
            loss = importance_metrics["loss"]
        with torch.no_grad():
            self.last_metrics = {
                "roi_projection_objective": objective_mode,
                "roi_projection_valid_emitters": float(valid_emitters),
                "roi_projection_projected_photons": float(projected_photons.detach().cpu().item()),
                "roi_projection_background_mean": float((background_sum / max(1, background_pixels)).detach().cpu().item()),
                "roi_projection_gamma_size": float(self.gamma_size),
                "roi_projection_renderer": 1.0,
                "roi_projection_base_coeff_maps_enabled": 1.0 if self.base_maps_by_domain else 0.0,
                "roi_projection_poisson_nll": float(poisson_loss.detach().cpu().item()),
                "roi_projection_sample_batch_size": float(sample_chunk),
                "roi_projection_sample_chunk_count": float(math.ceil(int(raw_all.shape[0]) / sample_chunk)),
                "roi_projection_emitter_chunk_size": float(max(1, int(self.config.projection_emitter_chunk_size))),
            }
            self.last_metrics.update(importance_metrics["metrics"])
        return loss

    def _chunk_log_p(
        self,
        *,
        gamma_t: torch.Tensor,
        raw_frames: torch.Tensor,
        background: torch.Tensor,
        xyzph: torch.Tensor,
        mask: torch.Tensor,
        roi_origin_xy_px: torch.Tensor | None,
        domain_names: list[str] | tuple[str, ...] | None,
    ) -> dict[str, torch.Tensor | int]:
        raw = raw_frames.to(device=self.device, dtype=torch.float32)
        bkg = background.to(device=self.device, dtype=torch.float32)
        xyzph_t = xyzph.to(device=self.device, dtype=torch.float32)
        mask_t = mask.to(device=self.device)
        projected_signal = torch.zeros_like(raw)
        active = mask_t & torch.isfinite(xyzph_t).all(dim=-1) & (xyzph_t[..., 3] > 0)
        if bool(active.any()):
            batch_index, emitter_index = active.nonzero(as_tuple=True)
            emitter_chunk = max(1, int(self.config.projection_emitter_chunk_size))
            for start in range(0, int(batch_index.shape[0]), emitter_chunk):
                stop = min(start + emitter_chunk, int(batch_index.shape[0]))
                batch_part = batch_index[start:stop]
                emitter_part = emitter_index[start:stop]
                emitter_xyzph = xyzph_t[batch_part, emitter_part]
                center_x = torch.floor(emitter_xyzph[:, 0]).to(dtype=torch.long)
                center_y = torch.floor(emitter_xyzph[:, 1]).to(dtype=torch.long)
                cell_center_x = center_x.to(dtype=torch.float32) + 0.5
                cell_center_y = center_y.to(dtype=torch.float32) + 0.5
                local_x_nm = (emitter_xyzph[:, 0] - cell_center_x) * float(self.config.pixel_size_x_nm)
                local_y_nm = (emitter_xyzph[:, 1] - cell_center_y) * float(self.config.pixel_size_y_nm)
                origin = _origin_for_batch(
                    roi_origin_xy_px,
                    batch_index=batch_part,
                    device=self.device,
                )
                full_xy = torch.stack([cell_center_x, cell_center_y], dim=1) + origin
                coeffs_nm, _, _ = evaluate_zernike_from_roi_positions_torch(
                    full_xy,
                    gamma_t,
                    self.nat_config,
                    local_x_nm=local_x_nm,
                    local_y_nm=local_y_nm,
                    dtype=torch.float32,
                    device=self.device,
                )
                base_coeffs = self._sample_base_coeffs(full_xy, batch_index=batch_part, domain_names=domain_names)
                if base_coeffs is not None:
                    coeffs_nm = coeffs_nm + base_coeffs
                patches = self._render_vector_patches(
                    local_x_nm=local_x_nm,
                    local_y_nm=local_y_nm,
                    z_nm=emitter_xyzph[:, 2],
                    photons=emitter_xyzph[:, 3].clamp_min(1e-6),
                    coeffs_nm=coeffs_nm,
                )
                projected_signal = projected_signal + _project_patches_to_frames(
                    patches,
                    batch_index=batch_part,
                    center_x_px=center_x,
                    center_y_px=center_y,
                    output_shape=(int(raw.shape[0]), int(raw.shape[1]), int(raw.shape[2])),
                )

        expected = projected_signal + bkg
        raw_c, expected_c = _crop(raw, expected, int(self.config.over_cut_px))
        nll_per_pixel = poisson_nll(raw_c, expected_c, eps=float(self.config.eps))
        return {
            "nll_sum": nll_per_pixel.sum(),
            "pixel_count": int(nll_per_pixel.numel()),
            "log_p_per_sample": -nll_per_pixel.flatten(start_dim=1).sum(dim=1),
            "valid_emitters": int(active.sum().item()),
            "projected_photons": projected_signal.sum(),
            "background_sum": bkg.sum(),
            "background_pixels": int(bkg.numel()),
        }

    def render_reconstruction(
        self,
        *,
        gamma: torch.Tensor,
        samples: DetectionPosteriorSamples,
        background: torch.Tensor,
        batch_index: int,
        roi_origin_xy_px: torch.Tensor | None = None,
        domain_names: list[str] | tuple[str, ...] | None = None,
    ) -> torch.Tensor:
        bkg = torch.as_tensor(background, dtype=torch.float32, device=self.device)
        if bkg.ndim == 3:
            bkg = bkg[int(batch_index)]
        shape = (1, int(bkg.shape[-2]), int(bkg.shape[-1]))
        sample = DetectionPosteriorSamples(
            xyzph=samples.xyzph[int(batch_index) : int(batch_index) + 1],
            mask=samples.mask[int(batch_index) : int(batch_index) + 1],
            logits=samples.logits[int(batch_index) : int(batch_index) + 1],
            metadata=dict(samples.metadata),
        )
        zero_raw = torch.zeros(shape, dtype=torch.float32, device=self.device)
        origin = None
        names = None
        if roi_origin_xy_px is not None:
            origin = torch.as_tensor(roi_origin_xy_px, dtype=torch.float32)[int(batch_index) : int(batch_index) + 1]
        if domain_names is not None:
            names = [str(domain_names[int(batch_index)])]
        self(gamma=gamma, samples=sample, raw_frames=zero_raw, background=bkg.unsqueeze(0), roi_origin_xy_px=origin, domain_names=names)
        projected_plus_bkg = self._last_reconstruction(
            gamma=gamma,
            samples=sample,
            background=bkg.unsqueeze(0),
            roi_origin_xy_px=origin,
            domain_names=names,
        )
        return projected_plus_bkg[0].detach().cpu()

    def render_record_reconstruction(
        self,
        *,
        gamma: torch.Tensor,
        raw_shape: tuple[int, int],
        emitters: object,
        background: torch.Tensor | np.ndarray,
        roi_origin_xy_px: tuple[float, float] | torch.Tensor,
        domain_name: str | None = None,
    ) -> torch.Tensor:
        height, width = int(raw_shape[0]), int(raw_shape[1])
        bkg = torch.as_tensor(background, dtype=torch.float32, device=self.device)
        if tuple(bkg.shape) != (height, width):
            raise ValueError(f"background shape {tuple(bkg.shape)} must match raw shape {(height, width)}")
        emitters_seq = tuple(emitters or ())
        projected_signal = torch.zeros((1, height, width), dtype=torch.float32, device=self.device)
        if emitters_seq:
            xy = torch.tensor(
                [[float(item.local_xy_px[0]), float(item.local_xy_px[1])] for item in emitters_seq],
                dtype=torch.float32,
                device=self.device,
            )
            z_nm = torch.tensor([float(item.mu_z_nm) for item in emitters_seq], dtype=torch.float32, device=self.device)
            photons = torch.tensor(
                [max(float(item.mu_photons), 1e-6) for item in emitters_seq],
                dtype=torch.float32,
                device=self.device,
            )
            center_x = torch.floor(xy[:, 0]).to(dtype=torch.long)
            center_y = torch.floor(xy[:, 1]).to(dtype=torch.long)
            cell_center_x = center_x.to(dtype=torch.float32) + 0.5
            cell_center_y = center_y.to(dtype=torch.float32) + 0.5
            local_x_nm = (xy[:, 0] - cell_center_x) * float(self.config.pixel_size_x_nm)
            local_y_nm = (xy[:, 1] - cell_center_y) * float(self.config.pixel_size_y_nm)
            origin = torch.as_tensor(roi_origin_xy_px, dtype=torch.float32, device=self.device).reshape(1, 2)
            full_xy = torch.stack([cell_center_x, cell_center_y], dim=1) + origin
            gamma_t = gamma.to(device=self.device, dtype=torch.float32).reshape(-1)
            coeffs_nm, _, _ = evaluate_zernike_from_roi_positions_torch(
                full_xy,
                gamma_t,
                self.nat_config,
                local_x_nm=local_x_nm,
                local_y_nm=local_y_nm,
                dtype=torch.float32,
                device=self.device,
            )
            if self.base_maps_by_domain:
                batch_index = torch.zeros((len(emitters_seq),), dtype=torch.long, device=self.device)
                names = [str(domain_name or next(iter(self.base_maps_by_domain)))]
                base_coeffs = self._sample_base_coeffs(full_xy, batch_index=batch_index, domain_names=names)
                if base_coeffs is not None:
                    coeffs_nm = coeffs_nm + base_coeffs
            patches = self._render_vector_patches(
                local_x_nm=local_x_nm,
                local_y_nm=local_y_nm,
                z_nm=z_nm,
                photons=photons,
                coeffs_nm=coeffs_nm,
            )
            projected_signal = _project_patches_to_frames(
                patches,
                batch_index=torch.zeros((len(emitters_seq),), dtype=torch.long, device=self.device),
                center_x_px=center_x,
                center_y_px=center_y,
                output_shape=(1, height, width),
            )
        return (projected_signal[0] + bkg).detach().cpu()

    def _last_reconstruction(
        self,
        *,
        gamma: torch.Tensor,
        samples: DetectionPosteriorSamples,
        background: torch.Tensor,
        roi_origin_xy_px: torch.Tensor | None = None,
        domain_names: list[str] | tuple[str, ...] | None = None,
    ) -> torch.Tensor:
        with torch.enable_grad():
            raw = torch.zeros_like(torch.as_tensor(background, dtype=torch.float32, device=self.device))
            self(gamma=gamma, samples=samples, raw_frames=raw, background=background, roi_origin_xy_px=roi_origin_xy_px, domain_names=domain_names)
            # Recompute without log-likelihood side effects for diagnostic output.
            bkg = torch.as_tensor(background, dtype=torch.float32, device=self.device)
            xyzph = samples.xyzph.detach().to(device=self.device, dtype=torch.float32)
            mask = samples.mask.detach().to(device=self.device)
            active = mask & torch.isfinite(xyzph).all(dim=-1) & (xyzph[..., 3] > 0)
            projected_signal = torch.zeros_like(bkg)
            if bool(active.any()):
                batch_idx, emitter_idx = active.nonzero(as_tuple=True)
                emitter_xyzph = xyzph[batch_idx, emitter_idx]
                center_x = torch.floor(emitter_xyzph[:, 0]).to(dtype=torch.long)
                center_y = torch.floor(emitter_xyzph[:, 1]).to(dtype=torch.long)
                cell_center_x = center_x.to(dtype=torch.float32) + 0.5
                cell_center_y = center_y.to(dtype=torch.float32) + 0.5
                local_x_nm = (emitter_xyzph[:, 0] - cell_center_x) * float(self.config.pixel_size_x_nm)
                local_y_nm = (emitter_xyzph[:, 1] - cell_center_y) * float(self.config.pixel_size_y_nm)
                origin = _origin_for_batch(
                    roi_origin_xy_px,
                    batch_index=batch_idx,
                    device=self.device,
                )
                full_xy = torch.stack([cell_center_x, cell_center_y], dim=1) + origin
                coeffs_nm, _, _ = evaluate_zernike_from_roi_positions_torch(
                    full_xy,
                    gamma.to(device=self.device, dtype=torch.float32).reshape(-1),
                    self.nat_config,
                    local_x_nm=local_x_nm,
                    local_y_nm=local_y_nm,
                    dtype=torch.float32,
                    device=self.device,
                )
                base_coeffs = self._sample_base_coeffs(full_xy, batch_index=batch_idx, domain_names=domain_names)
                if base_coeffs is not None:
                    coeffs_nm = coeffs_nm + base_coeffs
                patches = self._render_vector_patches(
                    local_x_nm=local_x_nm,
                    local_y_nm=local_y_nm,
                    z_nm=emitter_xyzph[:, 2],
                    photons=emitter_xyzph[:, 3].clamp_min(1e-6),
                    coeffs_nm=coeffs_nm,
                )
                projected_signal = _project_patches_to_frames(
                    patches,
                    batch_index=batch_idx,
                    center_x_px=center_x,
                    center_y_px=center_y,
                    output_shape=(int(bkg.shape[0]), int(bkg.shape[1]), int(bkg.shape[2])),
                )
            return projected_signal + bkg

    def _sample_base_coeffs(
        self,
        full_xy: torch.Tensor,
        *,
        batch_index: torch.Tensor,
        domain_names: list[str] | tuple[str, ...] | None,
    ) -> torch.Tensor | None:
        if not self.base_maps_by_domain:
            return None
        if domain_names is None:
            if len(self.base_maps_by_domain) == 1:
                domain_names = [next(iter(self.base_maps_by_domain)) for _ in range(int(batch_index.max().item()) + 1)]
            else:
                raise ValueError("domain_names are required when multiple base coeff-map domains are configured")
        coeffs = torch.empty((full_xy.shape[0], len(self.nat_config.aberrations)), device=self.device, dtype=torch.float32)
        for domain in sorted(set(str(domain_names[int(index)]) for index in batch_index.detach().cpu().tolist())):
            if domain not in self.base_maps_by_domain:
                raise KeyError(f"missing base coeff maps for domain {domain!r}")
            selector = torch.as_tensor([str(domain_names[int(index)]) == domain for index in batch_index.detach().cpu().tolist()], device=self.device)
            maps = self.base_maps_by_domain[domain]
            xy = full_xy[selector]
            x_idx = torch.round(xy[:, 0]).to(dtype=torch.long).clamp_(0, int(maps.shape[-1]) - 1)
            y_idx = torch.round(xy[:, 1]).to(dtype=torch.long).clamp_(0, int(maps.shape[-2]) - 1)
            coeffs[selector] = maps[:, y_idx, x_idx].transpose(0, 1)
        return coeffs

    def coefficients_at(
        self,
        *,
        gamma: torch.Tensor,
        full_xy_px: torch.Tensor,
        domain_name: str | None = None,
    ) -> torch.Tensor:
        gamma_t = gamma.to(device=self.device, dtype=torch.float32).reshape(-1)
        xy = torch.as_tensor(full_xy_px, dtype=torch.float32, device=self.device).reshape(-1, 2)
        delta, _, _ = evaluate_zernike_from_roi_positions_torch(
            xy,
            gamma_t,
            self.nat_config,
            dtype=torch.float32,
            device=self.device,
        )
        if not self.base_maps_by_domain:
            return delta
        domain = str(domain_name or next(iter(self.base_maps_by_domain)))
        batch_index = torch.zeros((xy.shape[0],), dtype=torch.long, device=self.device)
        base = self._sample_base_coeffs(xy, batch_index=batch_index, domain_names=[domain])
        return delta if base is None else delta + base

    def _render_vector_patches(
        self,
        *,
        local_x_nm: torch.Tensor,
        local_y_nm: torch.Tensor,
        z_nm: torch.Tensor,
        photons: torch.Tensor,
        coeffs_nm: torch.Tensor,
    ) -> torch.Tensor:
        coeffs_rad = coeffs_nm * (2.0 * math.pi / max(float(self.config.wavelength_nm), 1e-6)) * self.ctx.normfac[None, :]
        psf = render_vector_psf_bank(
            self.ctx,
            coeffs_rad,
            z_nm.to(device=self.device, dtype=torch.float32) * 1e-9,
            out_size=int(self.config.patch_size_px),
            batch_size=int(self.config.renderer_batch_size),
            return_torch=True,
        )
        psf = _fourier_shift_patches(
            psf,
            shift_x_px=local_x_nm / max(float(self.config.pixel_size_x_nm), 1e-6),
            shift_y_px=local_y_nm / max(float(self.config.pixel_size_y_nm), 1e-6),
        ).clamp_min(0.0)
        psf = psf / (psf.sum(dim=(-2, -1), keepdim=True) + 1e-12)
        return psf * photons.to(device=self.device, dtype=torch.float32).reshape(-1, 1, 1)


def poisson_nll(observed: torch.Tensor, expected: torch.Tensor, *, eps: float = 1e-6) -> torch.Tensor:
    expected_safe = expected.clamp_min(float(eps))
    observed_safe = observed.clamp_min(0.0)
    return expected_safe - observed_safe * torch.log(expected_safe) + torch.lgamma(observed_safe + 1.0)


def _importance_wake_loss_and_metrics(
    log_p_per_sample: torch.Tensor,
    *,
    metadata: dict[str, object],
    sample_count: int,
) -> dict[str, object]:
    log_q_raw = metadata.get("log_q_h_given_x")
    group_raw = metadata.get("posterior_group_id")
    log_q_values, log_q_missing_count = _metadata_float_list(log_q_raw, sample_count=sample_count)
    group_ids = _metadata_int_list(group_raw, sample_count=sample_count)
    log_q = torch.as_tensor(log_q_values, device=log_p_per_sample.device, dtype=log_p_per_sample.dtype)
    group_to_indices: dict[int, list[int]] = {}
    for index, group_id in enumerate(group_ids):
        group_to_indices.setdefault(int(group_id), []).append(int(index))

    group_losses: list[torch.Tensor] = []
    group_ess: list[torch.Tensor] = []
    group_sizes: list[int] = []
    log_p_ranges: list[torch.Tensor] = []
    log_q_ranges: list[torch.Tensor] = []
    importance_ranges: list[torch.Tensor] = []
    weight_maxes: list[torch.Tensor] = []
    for indices in group_to_indices.values():
        group_sizes.append(len(indices))
        index_tensor = torch.as_tensor(indices, device=log_p_per_sample.device, dtype=torch.long)
        group_log_p = log_p_per_sample.index_select(0, index_tensor)
        group_log_q = log_q.index_select(0, index_tensor)
        importance = group_log_p - group_log_q
        weights = torch.softmax(importance.detach(), dim=0)
        group_losses.append(-(weights * group_log_p).sum())
        group_ess.append(1.0 / weights.square().sum().clamp_min(1e-12))
        log_p_ranges.append(group_log_p.detach().max() - group_log_p.detach().min())
        log_q_ranges.append(group_log_q.detach().max() - group_log_q.detach().min())
        importance_ranges.append(importance.detach().max() - importance.detach().min())
        weight_maxes.append(weights.detach().max())

    if group_losses:
        loss = torch.stack(group_losses).mean()
    else:
        loss = -log_p_per_sample.mean()
    metrics: dict[str, float] = {
        "roi_projection_log_p_mean": float(log_p_per_sample.detach().mean().cpu().item()) if log_p_per_sample.numel() else 0.0,
        "roi_projection_log_q_mean": float(log_q.detach().mean().cpu().item()) if log_q.numel() else 0.0,
        "roi_projection_log_q_missing_count": float(log_q_missing_count),
        "roi_projection_importance_group_count": float(len(group_sizes)),
        "roi_projection_importance_group_size_min": float(min(group_sizes)) if group_sizes else 0.0,
        "roi_projection_importance_group_size_max": float(max(group_sizes)) if group_sizes else 0.0,
        "roi_projection_importance_group_size_mean": float(sum(group_sizes) / len(group_sizes)) if group_sizes else 0.0,
        "roi_projection_importance_incomplete_group_fraction": float(
            sum(1 for size in group_sizes if size != max(group_sizes)) / max(1, len(group_sizes))
        )
        if group_sizes
        else 0.0,
        "roi_projection_wake_loss": float(loss.detach().cpu().item()),
    }
    if group_ess:
        metrics["roi_projection_importance_ess_mean"] = float(torch.stack(group_ess).mean().cpu().item())
    if log_p_ranges:
        metrics.update(
            {
                "roi_projection_log_p_range_mean": float(torch.stack(log_p_ranges).mean().cpu().item()),
                "roi_projection_log_p_range_max": float(torch.stack(log_p_ranges).max().cpu().item()),
                "roi_projection_log_q_range_mean": float(torch.stack(log_q_ranges).mean().cpu().item()),
                "roi_projection_log_q_range_max": float(torch.stack(log_q_ranges).max().cpu().item()),
                "roi_projection_importance_range_mean": float(torch.stack(importance_ranges).mean().cpu().item()),
                "roi_projection_importance_range_max": float(torch.stack(importance_ranges).max().cpu().item()),
                "roi_projection_importance_weight_max_mean": float(torch.stack(weight_maxes).mean().cpu().item()),
                "roi_projection_importance_weight_max": float(torch.stack(weight_maxes).max().cpu().item()),
            }
        )
    return {"loss": loss, "metrics": metrics}


def _metadata_float_list(value: object, *, sample_count: int) -> tuple[list[float], int]:
    if isinstance(value, torch.Tensor):
        raw = value.detach().cpu().reshape(-1).tolist()
    elif isinstance(value, np.ndarray):
        raw = value.reshape(-1).tolist()
    elif isinstance(value, (list, tuple)):
        raw = list(value)
    else:
        raw = []
    missing = max(0, int(sample_count) - len(raw)) + sum(1 for item in raw[: int(sample_count)] if item is None)
    if len(raw) < int(sample_count):
        raw.extend([0.0] * (int(sample_count) - len(raw)))
    return [0.0 if v is None else float(v) for v in raw[: int(sample_count)]], int(missing)


def _metadata_int_list(value: object, *, sample_count: int) -> list[int]:
    if isinstance(value, torch.Tensor):
        raw = value.detach().cpu().reshape(-1).tolist()
    elif isinstance(value, np.ndarray):
        raw = value.reshape(-1).tolist()
    elif isinstance(value, (list, tuple)):
        raw = list(value)
    else:
        raw = list(range(int(sample_count)))
    if len(raw) < int(sample_count):
        raw.extend(range(len(raw), int(sample_count)))
    return [int(v) for v in raw[: int(sample_count)]]


def _nat_aberration_noll_indices(config: NATFieldConfig) -> tuple[int, ...]:
    mode_to_noll: dict[tuple[int, int], int] = {}
    for noll_index in range(1, 128):
        mode_to_noll[noll_to_nm(noll_index)] = noll_index
    return tuple(mode_to_noll[mode] for mode in config.aberrations)


def _load_base_coeff_maps(
    entries: tuple[tuple[str, str], ...],
    *,
    expected_modes: tuple[tuple[int, int], ...],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    result: dict[str, torch.Tensor] = {}
    for domain, path in entries:
        payload = np.load(str(path))
        maps = np.asarray(payload["zernike_maps_nm"], dtype=np.float32)
        mode_order = [tuple(int(v) for v in row) for row in np.asarray(payload["mode_order"])]
        mode_to_index = {mode: index for index, mode in enumerate(mode_order)}
        indices = [mode_to_index[mode] for mode in expected_modes]
        ordered = torch.as_tensor(maps[indices], dtype=torch.float32, device=device)
        result[str(domain)] = ordered
    return result


def _origin_for_batch(
    roi_origin_xy_px: torch.Tensor | None,
    *,
    batch_index: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    if roi_origin_xy_px is None:
        return torch.zeros((int(batch_index.shape[0]), 2), dtype=torch.float32, device=device)
    origins = torch.as_tensor(roi_origin_xy_px, dtype=torch.float32, device=device)
    if origins.ndim != 2 or int(origins.shape[1]) != 2:
        raise ValueError(f"roi_origin_xy_px must have shape (B,2), got {tuple(origins.shape)}")
    return origins[batch_index.to(device=device, dtype=torch.long)]


def _fourier_shift_patches(patches: torch.Tensor, *, shift_x_px: torch.Tensor, shift_y_px: torch.Tensor) -> torch.Tensor:
    if patches.numel() == 0:
        return patches
    shift_x = shift_x_px.to(device=patches.device, dtype=patches.dtype).reshape(-1)
    shift_y = shift_y_px.to(device=patches.device, dtype=patches.dtype).reshape(-1)
    if shift_x.shape[0] != patches.shape[0] or shift_y.shape[0] != patches.shape[0]:
        raise ValueError("Subpixel shifts must match patch count.")
    max_shift = max(float(shift_x.abs().max().detach().cpu().item()), float(shift_y.abs().max().detach().cpu().item()))
    if max_shift <= 1e-8:
        return patches
    height = int(patches.shape[-2])
    width = int(patches.shape[-1])
    freq_y = torch.fft.fftfreq(height, d=1.0, device=patches.device).to(dtype=patches.dtype)
    freq_x = torch.fft.fftfreq(width, d=1.0, device=patches.device).to(dtype=patches.dtype)
    phase_arg = -2.0 * math.pi * (
        shift_y[:, None, None] * freq_y[None, :, None] + shift_x[:, None, None] * freq_x[None, None, :]
    )
    phase = torch.exp(torch.complex(torch.zeros_like(phase_arg), phase_arg))
    return torch.fft.ifft2(torch.fft.fft2(patches) * phase).real


def _project_patches_to_frames(
    patches: torch.Tensor,
    *,
    batch_index: torch.Tensor,
    center_x_px: torch.Tensor,
    center_y_px: torch.Tensor,
    output_shape: tuple[int, int, int],
) -> torch.Tensor:
    batch, height, width = (int(output_shape[0]), int(output_shape[1]), int(output_shape[2]))
    projected = torch.zeros((batch, height, width), device=patches.device, dtype=patches.dtype)
    if patches.numel() == 0:
        return projected
    patch_size = int(patches.shape[-1])
    radius = patch_size // 2
    yy, xx = torch.meshgrid(
        torch.arange(patch_size, device=patches.device),
        torch.arange(patch_size, device=patches.device),
        indexing="ij",
    )
    image_x = center_x_px.to(device=patches.device, dtype=torch.long)[:, None, None] - radius + xx[None, :, :]
    image_y = center_y_px.to(device=patches.device, dtype=torch.long)[:, None, None] - radius + yy[None, :, :]
    batch_ix = batch_index.to(device=patches.device, dtype=torch.long)[:, None, None].expand_as(image_x)
    valid = (batch_ix >= 0) & (batch_ix < batch) & (image_x >= 0) & (image_x < width) & (image_y >= 0) & (image_y < height)
    if not bool(valid.any()):
        return projected
    flat_index = ((batch_ix * height + image_y) * width + image_x)[valid]
    projected.reshape(-1).scatter_add_(0, flat_index.reshape(-1), patches[valid].reshape(-1))
    return projected


def _crop(raw: torch.Tensor, projected: torch.Tensor, over_cut_px: int) -> tuple[torch.Tensor, torch.Tensor]:
    cut = int(over_cut_px)
    if cut <= 0:
        return raw, projected
    if raw.shape[-1] <= 2 * cut or raw.shape[-2] <= 2 * cut:
        raise ValueError("over_cut_px is too large for ROI dimensions")
    return raw[..., cut:-cut, cut:-cut], projected[..., cut:-cut, cut:-cut]
