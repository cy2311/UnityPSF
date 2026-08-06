from __future__ import annotations

import math

import torch

from unity_psf.localization.smlm_output import SMLMOutputChannels, decode_smlm_output
from unity_psf.localization.smlm_targets import (
    SMLMTargetConvention,
    absolute_pxyz_to_local_targets,
    target_pixel_indices,
)


class ActiveSMLMLoss:
    def __init__(
        self,
        *,
        detection_weight: float = 1.0,
        pxyz_weight: float = 1.0,
        background_weight: float = 0.1,
        sigma_weight: float = 0.01,
        photon_scale: float | None = 1.0,
        z_scale: float | None = None,
        z_activation: str = "tanh",
        sigma_min: float = 1e-3,
        record_components: bool = False,
    ) -> None:
        self.detection_weight = float(detection_weight)
        self.pxyz_weight = float(pxyz_weight)
        self.background_weight = float(background_weight)
        self.sigma_weight = float(sigma_weight)
        self.photon_scale = None if photon_scale is None else float(photon_scale)
        self.z_scale = None if z_scale is None else float(z_scale)
        self.z_activation = str(z_activation).strip().lower()
        self.sigma_min = float(sigma_min)
        self.record_components = bool(record_components)
        self.last_components: dict[str, float] | None = None

    def forward(
        self,
        y_out: torch.Tensor,
        detect_tar: torch.Tensor,
        pxyz_tar: torch.Tensor,
        mask_tar: torch.Tensor,
        bkg_tar: torch.Tensor,
    ) -> torch.Tensor:
        output = decode_smlm_output(y_out)
        detect_loss = _bernoulli_probability_loss(output.detection_prob, detect_tar)
        pxyz_loss = _masked_pxyz_gaussian_nll(
            output.raw,
            pxyz_tar,
            mask_tar,
            photon_scale=self.photon_scale,
            z_scale=self.z_scale,
            z_activation=self.z_activation,
            sigma_min=self.sigma_min,
        )
        bg_loss = torch.nn.functional.mse_loss(output.bg, bkg_tar, reduction="none").flatten(start_dim=1).mean(dim=1)
        sigma_loss = output.pxyz_sigma.clamp_min(self.sigma_min).square().flatten(start_dim=1).mean(dim=1)
        total = (
            self.detection_weight * detect_loss
            + self.pxyz_weight * pxyz_loss
            + self.background_weight * bg_loss
            + self.sigma_weight * sigma_loss
        )
        if self.record_components:
            self.last_components = {
                "loss_detect": float(detect_loss.detach().mean().cpu().item()),
                "loss_pxyz": float(pxyz_loss.detach().mean().cpu().item()),
                "loss_bg": float(bg_loss.detach().mean().cpu().item()),
                "loss_sigma": float(sigma_loss.detach().mean().cpu().item()),
                "loss_total": float(total.detach().mean().cpu().item()),
            }
        return total


class ActiveSMLMGMMTargetAdapter:
    def __init__(
        self,
        *,
        photon_scale: float | None = 1.0,
        z_scale: float | None = None,
        disable_attr: int | tuple[int, ...] | list[int] | None = None,
        target_order: str = "legacy_iwae",
    ) -> None:
        self.photon_scale = None if photon_scale is None else float(photon_scale)
        self.z_scale = None if z_scale is None else float(z_scale)
        self.target_order = _normalize_gmm_target_order(target_order)
        if disable_attr is None:
            self.disable_attr = ()
        elif isinstance(disable_attr, int):
            self.disable_attr = (int(disable_attr),)
        else:
            self.disable_attr = tuple(int(item) for item in disable_attr)

    def to_gmm_order(self, pxyz_tar: torch.Tensor) -> torch.Tensor:
        if pxyz_tar.shape[-1] != 4:
            raise ValueError(f"pxyz_tar must have last dimension 4, got {tuple(pxyz_tar.shape)}")
        if self.photon_scale is not None and self.photon_scale <= 0.0:
            raise ValueError("photon_scale must be positive or None")
        if self.z_scale is not None and self.z_scale <= 0.0:
            raise ValueError("z_scale must be positive or None")
        if self.target_order == "legacy_iwae":
            converted = pxyz_tar.clone()
        else:
            photons = pxyz_tar[..., 3]
            z = pxyz_tar[..., 2]
            if self.photon_scale is not None:
                photons = photons / self.photon_scale
            if self.z_scale is not None:
                z = z / self.z_scale
            converted = torch.stack((photons, pxyz_tar[..., 0], pxyz_tar[..., 1], z), dim=-1)
        if self.disable_attr:
            converted = converted.clone()
            converted[..., list(self.disable_attr)] = 0.0
        return converted

    def v03_to_gmm_order(self, pxyz_tar: torch.Tensor) -> torch.Tensor:
        photons = pxyz_tar[..., 3]
        z = pxyz_tar[..., 2]
        if self.photon_scale is not None:
            photons = photons / self.photon_scale
        if self.z_scale is not None:
            z = z / self.z_scale
        converted = torch.stack((photons, pxyz_tar[..., 0], pxyz_tar[..., 1], z), dim=-1)
        if self.disable_attr:
            converted = converted.clone()
            converted[..., list(self.disable_attr)] = 0.0
        return converted


class ActiveSMLMGMMLoss:
    def __init__(
        self,
        *,
        xyoffset: tuple[float, float] = (0.0, 0.0),
        ch_weight: tuple[float, float] = (1.0, 1.0),
        photon_scale: float | None = 1.0,
        z_scale: float | None = None,
        disable_attr: int | tuple[int, ...] | list[int] | None = None,
        gmm_target_chunk: int = 16,
        gmm_component_chunk: int = 4096,
        gmm_backend: str = "manual_chunked",
        target_order: str = "legacy_iwae",
        eps: float = 1e-8,
        record_components: bool = False,
    ) -> None:
        self.offset_x = float(xyoffset[0])
        self.offset_y = float(xyoffset[1])
        self.ch_weight = (float(ch_weight[0]), float(ch_weight[1]))
        self.target_adapter = ActiveSMLMGMMTargetAdapter(
            photon_scale=photon_scale,
            z_scale=z_scale,
            disable_attr=disable_attr,
            target_order=target_order,
        )
        self.gmm_target_chunk = int(gmm_target_chunk) if gmm_target_chunk is not None else 0
        self.gmm_component_chunk = int(gmm_component_chunk) if gmm_component_chunk is not None else 0
        self.gmm_backend = str(gmm_backend or "manual_chunked").strip().lower()
        self.eps = float(eps)
        self.record_components = bool(record_components)
        self.last_components: dict[str, float] | None = None

    def forward(
        self,
        y_out: torch.Tensor,
        detect_tar: torch.Tensor,
        pxyz_tar: torch.Tensor,
        mask_tar: torch.Tensor,
        bkg_tar: torch.Tensor,
    ) -> torch.Tensor:
        _validate_gmm_shapes(y_out, detect_tar, pxyz_tar, mask_tar, bkg_tar)
        p = y_out[:, SMLMOutputChannels.p]
        pxyz_mu = y_out[:, SMLMOutputChannels.pxyz_mu]
        pxyz_sig = y_out[:, SMLMOutputChannels.pxyz_sigma]
        bkg = y_out[:, SMLMOutputChannels.bg]
        pxyz_tar_gmm = self.target_adapter.to_gmm_order(pxyz_tar).to(dtype=y_out.dtype, device=y_out.device)

        loss_count, loss_loc = self._compute_gmm_terms(p, pxyz_mu, pxyz_sig, pxyz_tar_gmm, mask_tar)
        loss_gmm = loss_count + loss_loc
        loss_bkg = torch.nn.functional.mse_loss(bkg, bkg_tar.to(dtype=y_out.dtype, device=y_out.device), reduction="none").sum(dim=(-1, -2))
        weights = y_out.new_tensor(self.ch_weight)
        loss = 2.0 * torch.stack((loss_gmm, loss_bkg), dim=1) * weights.unsqueeze(0)
        total = loss.sum(dim=1)
        if self.record_components:
            self.last_components = {
                "loss_gmm": float(loss_gmm.detach().mean().cpu().item()),
                "loss_bkg": float(loss_bkg.detach().mean().cpu().item()),
                "loss_total": float(total.detach().mean().cpu().item()),
            }
        return loss

    def _compute_gmm_terms(
        self,
        p: torch.Tensor,
        pxyz_mu: torch.Tensor,
        pxyz_sig: torch.Tensor,
        pxyz_tar: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        nlocs_mean = p.sum(dim=(-1, -2))
        nlocs_var = (p - p**2).sum(dim=(-1, -2)).clamp_min(self.eps)
        nlocs_tar = mask.sum(dim=-1).to(dtype=p.dtype, device=p.device)
        count_log_prob = torch.distributions.Normal(nlocs_mean, nlocs_var.sqrt()).log_prob(nlocs_tar) * nlocs_tar
        if not bool(mask.any()):
            return -count_log_prob, torch.zeros_like(count_log_prob)
        if self.gmm_backend in {"mixture_same_family", "mixture", "torch"}:
            loc_log_prob = self._mixture_same_family_log_gmm(p, pxyz_mu, pxyz_sig, pxyz_tar, mask)
        elif self.gmm_backend in {"manual_chunked", "manual", "chunked"}:
            loc_log_prob = self._manual_log_gmm(p, pxyz_mu, pxyz_sig, pxyz_tar, mask)
        else:
            raise ValueError(f"unsupported active_smlm_gmm_loss backend: {self.gmm_backend}")
        return -count_log_prob, -loc_log_prob

    def _mixture_same_family_log_gmm(
        self,
        p: torch.Tensor,
        pxyz_mu: torch.Tensor,
        pxyz_sig: torch.Tensor,
        pxyz_tar: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = int(p.shape[0])
        prob_flat = p.reshape(batch_size, -1).clamp_min(0.0)
        mix = torch.distributions.Categorical(logits=torch.log(prob_flat + self.eps))
        mu, sig = self._component_params(pxyz_mu, pxyz_sig)
        comp = torch.distributions.Independent(torch.distributions.Normal(mu, sig), 1)
        gmm = torch.distributions.MixtureSameFamily(mix, comp)
        log_gmm = gmm.log_prob(pxyz_tar.transpose(0, 1)).transpose(0, 1)
        return (log_gmm * mask.to(dtype=p.dtype, device=p.device)).sum(dim=-1)

    def _manual_log_gmm(
        self,
        p: torch.Tensor,
        pxyz_mu: torch.Tensor,
        pxyz_sig: torch.Tensor,
        pxyz_tar: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = int(p.shape[0])
        target_count = int(pxyz_tar.shape[1])
        if int(mask.shape[1]) != target_count:
            raise ValueError(f"mask shape mismatch: mask={tuple(mask.shape)} pxyz_tar={tuple(pxyz_tar.shape)}")
        active_cols = mask.any(dim=0) if target_count else None
        max_targets = 0
        if active_cols is not None and bool(active_cols.any()):
            max_targets = int(active_cols.nonzero(as_tuple=False).max().item()) + 1
        if max_targets <= 0:
            return p.new_zeros((batch_size,))

        prob_flat = torch.nan_to_num(p.reshape(batch_size, -1).clamp_min(0.0), nan=0.0, posinf=0.0, neginf=0.0)
        log_weights = torch.log(prob_flat + self.eps)
        log_weights = log_weights - torch.logsumexp(log_weights, dim=1, keepdim=True)
        mu, sig = self._component_params(pxyz_mu, pxyz_sig)

        target_chunk = max(1, int(self.gmm_target_chunk) if self.gmm_target_chunk else max_targets)
        comp_count = int(mu.shape[1])
        comp_chunk = max(1, int(self.gmm_component_chunk) if self.gmm_component_chunk else comp_count)
        log_norm = p.new_tensor(2.0 * math.pi).log()
        loc_log_prob = p.new_zeros((batch_size,))
        neg_inf = torch.full((batch_size, target_chunk), -torch.inf, dtype=p.dtype, device=p.device)

        for target_start in range(0, max_targets, target_chunk):
            target_end = min(max_targets, target_start + target_chunk)
            mask_chunk = mask[:, target_start:target_end]
            if not bool(mask_chunk.any()):
                continue
            target_chunk_values = pxyz_tar[:, target_start:target_end]
            running = neg_inf[:, : int(target_end - target_start)]
            for comp_start in range(0, comp_count, comp_chunk):
                comp_end = min(comp_count, comp_start + comp_chunk)
                mu_chunk = mu[:, comp_start:comp_end]
                sig_chunk = sig[:, comp_start:comp_end]
                weighted = log_weights[:, None, comp_start:comp_end].expand(
                    batch_size,
                    int(target_end - target_start),
                    int(comp_end - comp_start),
                ).clone()
                for dim in range(4):
                    sig_d = sig_chunk[:, None, :, dim]
                    diff_d = target_chunk_values[:, :, None, dim] - mu_chunk[:, None, :, dim]
                    weighted.add_(-0.5 * ((diff_d / sig_d) ** 2 + log_norm + 2.0 * torch.log(sig_d)))
                running = torch.logaddexp(running, torch.logsumexp(weighted, dim=-1))
            loc_log_prob = loc_log_prob + (running * mask_chunk.to(dtype=p.dtype, device=p.device)).sum(dim=-1)
        return loc_log_prob

    def _component_params(self, pxyz_mu: torch.Tensor, pxyz_sig: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, channels, height, width = pxyz_mu.shape
        if channels != 4:
            raise ValueError(f"pxyz_mu must have 4 channels, got {channels}")
        dtype = pxyz_mu.dtype
        device = pxyz_mu.device
        x_center = torch.arange(width, dtype=dtype, device=device).repeat(height).view(height, width) + 0.5
        y_center = torch.arange(height, dtype=dtype, device=device).repeat_interleave(width).view(height, width) + 0.5
        mu = pxyz_mu.clone()
        mu[:, 1] += x_center.unsqueeze(0) + self.offset_x
        mu[:, 2] += y_center.unsqueeze(0) + self.offset_y
        mu = mu.permute(0, 2, 3, 1).reshape(batch_size, -1, channels)
        sig = pxyz_sig.permute(0, 2, 3, 1).reshape(batch_size, -1, channels).clamp_min(self.eps)
        return mu, sig


def _bernoulli_probability_loss(prob: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    prob = prob.clamp(1e-6, 1.0 - 1e-6)
    target = target.to(dtype=prob.dtype, device=prob.device)
    loss = -(target * torch.log(prob) + (1.0 - target) * torch.log1p(-prob))
    return loss.flatten(start_dim=1).mean(dim=1)


def _normalize_gmm_target_order(value: str) -> str:
    key = str(value or "legacy_iwae").strip().lower()
    if key in {"legacy_iwae", "iwae", "old", "phot_xyz", "phot,x,y,z", "photons_x_y_z"}:
        return "legacy_iwae"
    if key in {"v03", "xyzph", "x,y,z,phot", "x_y_z_photons"}:
        return "v03"
    raise ValueError(f"unsupported active_smlm_gmm_loss target_order: {value!r}")


def _masked_pxyz_gaussian_nll(
    raw_output: torch.Tensor,
    pxyz_tar: torch.Tensor,
    mask_tar: torch.Tensor,
    *,
    photon_scale: float | None,
    z_scale: float | None,
    z_activation: str,
    sigma_min: float,
) -> torch.Tensor:
    batch_size = int(raw_output.shape[0])
    if int(pxyz_tar.shape[1]) == 0:
        return raw_output.new_zeros((batch_size,))
    losses = []
    for batch_idx in range(batch_size):
        valid = mask_tar[batch_idx].to(dtype=torch.bool, device=raw_output.device)
        if int(valid.sum().item()) == 0:
            losses.append(raw_output.new_zeros(()))
            continue
        targets = pxyz_tar[batch_idx].to(dtype=raw_output.dtype, device=raw_output.device)[valid]
        x, y = target_pixel_indices(targets, height=int(raw_output.shape[-2]), width=int(raw_output.shape[-1]))
        pred_mu = raw_output[batch_idx, SMLMOutputChannels.pxyz_mu, y, x].transpose(0, 1)
        pred_sigma = raw_output[batch_idx, SMLMOutputChannels.pxyz_sigma, y, x].transpose(0, 1).clamp_min(sigma_min)
        target_scaled = absolute_pxyz_to_local_targets(
            targets,
            x=x,
            y=y,
            convention=SMLMTargetConvention(photon_scale=photon_scale, z_scale=z_scale, z_activation=z_activation),
        )
        target_scaled = torch.stack(
            (target_scaled[:, 3], target_scaled[:, 0], target_scaled[:, 1], target_scaled[:, 2]),
            dim=1,
        )
        nll = 0.5 * (((pred_mu - target_scaled) / pred_sigma) ** 2) + torch.log(pred_sigma) + 0.5 * math.log(2.0 * math.pi)
        losses.append(nll.mean())
    return torch.stack(losses)


def _validate_gmm_shapes(
    y_out: torch.Tensor,
    detect_tar: torch.Tensor,
    pxyz_tar: torch.Tensor,
    mask_tar: torch.Tensor,
    bkg_tar: torch.Tensor,
) -> None:
    if y_out.ndim != 4 or int(y_out.shape[1]) != 10:
        raise ValueError(f"active_smlm_gmm_loss expects output shape (N,10,H,W), got {tuple(y_out.shape)}")
    batch_size, _, height, width = y_out.shape
    if tuple(detect_tar.shape) != (batch_size, height, width):
        raise ValueError(f"detect_tar shape mismatch: {tuple(detect_tar.shape)}")
    if pxyz_tar.ndim != 3 or int(pxyz_tar.shape[0]) != batch_size or int(pxyz_tar.shape[2]) != 4:
        raise ValueError(f"pxyz_tar must have shape (N,M,4), got {tuple(pxyz_tar.shape)}")
    if tuple(mask_tar.shape) != (batch_size, int(pxyz_tar.shape[1])):
        raise ValueError(f"mask_tar shape mismatch: {tuple(mask_tar.shape)}")
    if tuple(bkg_tar.shape) != (batch_size, height, width):
        raise ValueError(f"bkg_tar shape mismatch: {tuple(bkg_tar.shape)}")


__all__ = ["ActiveSMLMGMMTargetAdapter", "ActiveSMLMGMMLoss", "ActiveSMLMLoss"]
