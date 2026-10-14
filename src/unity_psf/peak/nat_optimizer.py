from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn as nn

from unity_psf.optics.nat_field import (
    NATFieldConfig,
    default_order1_config,
    evaluate_zernike_coefficients_torch,
    evaluate_zernike_from_roi_positions_torch,
)


NATRenderer = Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    torch.Tensor,
]


def _solve_least_squares(
    design: torch.Tensor,
    targets: torch.Tensor,
    *,
    ridge: float = 0.0,
) -> torch.Tensor:
    if targets.ndim == 1:
        targets = targets[:, None]
    if float(ridge) > 0.0:
        gram = design.T @ design
        reg = torch.eye(gram.shape[0], dtype=gram.dtype, device=gram.device) * float(ridge)
        return torch.linalg.solve(gram + reg, design.T @ targets)
    return torch.linalg.pinv(design) @ targets


def _build_nat_design_matrix_torch(
    xn: torch.Tensor,
    yn: torch.Tensor,
    config: NATFieldConfig,
) -> torch.Tensor:
    n_points = int(xn.shape[0])
    n_modes = len(config.aberrations)
    design = torch.zeros((n_points * n_modes, len(config.gammas)), dtype=xn.dtype, device=xn.device)
    mode_index = {mode: idx for idx, mode in enumerate(config.aberrations)}
    row_offsets = torch.arange(n_points, dtype=torch.long, device=xn.device) * n_modes
    for gamma_index, gamma in enumerate(config.gammas):
        unit_gamma = torch.zeros((len(config.gammas),), dtype=xn.dtype, device=xn.device)
        unit_gamma[gamma_index] = 1.0
        coeffs = evaluate_zernike_coefficients_torch(xn, yn, unit_gamma, config, dtype=xn.dtype, device=xn.device)
        for mode, idx in mode_index.items():
            del mode
            design[row_offsets + int(idx), gamma_index] = coeffs[:, idx]
    return design


def invert_nat_order1_torch(
    xn: torch.Tensor,
    yn: torch.Tensor,
    zernike_coefficients: torch.Tensor,
    config: NATFieldConfig,
    *,
    ridge: float = 0.0,
) -> torch.Tensor:
    x = xn.reshape(-1).to(dtype=torch.float32)
    y = yn.reshape(-1).to(device=x.device, dtype=x.dtype)
    coeffs = zernike_coefficients.to(device=x.device, dtype=x.dtype)
    design = _build_nat_design_matrix_torch(x, y, config)
    return _solve_least_squares(design, coeffs.reshape(-1), ridge=ridge).reshape(-1)


@dataclass(frozen=True)
class NATFieldObservation:
    coefficients_nm: torch.Tensor


class NATFieldModel(nn.Module):
    def __init__(
        self,
        config: NATFieldConfig | None = None,
        *,
        gamma_init: torch.Tensor | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.config = default_order1_config() if config is None else config
        init = torch.zeros((len(self.config.gammas),), dtype=dtype) if gamma_init is None else gamma_init.to(dtype=dtype)
        if init.numel() != len(self.config.gammas):
            raise ValueError("gamma_init length must match config.gammas.")
        self.gamma = nn.Parameter(init.reshape(-1))
        if device is not None:
            self.to(device=device)

    def forward_roi_positions(
        self,
        roixy_px: torch.Tensor,
        *,
        local_x_nm: torch.Tensor | None = None,
        local_y_nm: torch.Tensor | None = None,
    ) -> NATFieldObservation:
        coeffs, _, _ = evaluate_zernike_from_roi_positions_torch(
            roixy_px,
            self.gamma,
            self.config,
            local_x_nm=local_x_nm,
            local_y_nm=local_y_nm,
            dtype=self.gamma.dtype,
            device=self.gamma.device,
        )
        return NATFieldObservation(coefficients_nm=coeffs)


@dataclass(frozen=True)
class NATLocalInit:
    x_nm: torch.Tensor
    y_nm: torch.Tensor
    z_nm: torch.Tensor
    photons: torch.Tensor
    background: torch.Tensor


@dataclass(frozen=True)
class NATFitSnapshot:
    x_nm: torch.Tensor
    y_nm: torch.Tensor
    z_nm: torch.Tensor
    photons: torch.Tensor
    background: torch.Tensor
    gamma: torch.Tensor
    loss_history: tuple[float, ...]


class NATLocalParameters(nn.Module):
    def __init__(self, init: NATLocalInit) -> None:
        super().__init__()
        self.x_nm = nn.Parameter(init.x_nm.detach().clone().float())
        self.y_nm = nn.Parameter(init.y_nm.detach().clone().float())
        self.z_nm = nn.Parameter(init.z_nm.detach().clone().float())
        self.log_photons = nn.Parameter(torch.log(init.photons.detach().clone().float().clamp_min(1e-6)))
        self.log_background = nn.Parameter(torch.log(init.background.detach().clone().float().clamp_min(1e-6)))

    @property
    def photons(self) -> torch.Tensor:
        return torch.exp(self.log_photons)

    @property
    def background(self) -> torch.Tensor:
        return torch.exp(self.log_background)

    def clamp_min_(
        self,
        *,
        min_photons: torch.Tensor | float | None = None,
        min_background: torch.Tensor | float | None = None,
        z_min_nm: float | None = None,
        z_max_nm: float | None = None,
    ) -> None:
        with torch.no_grad():
            if min_photons is not None:
                floor = torch.as_tensor(min_photons, device=self.log_photons.device, dtype=self.log_photons.dtype).clamp_min(1e-12)
                self.log_photons.data = torch.maximum(self.log_photons.data, torch.log(floor))
            if min_background is not None:
                floor = torch.as_tensor(min_background, device=self.log_background.device, dtype=self.log_background.dtype).clamp_min(1e-12)
                self.log_background.data = torch.maximum(self.log_background.data, torch.log(floor))
            if z_min_nm is not None or z_max_nm is not None:
                self.z_nm.data = self.z_nm.data.clamp(
                    min=None if z_min_nm is None else float(z_min_nm),
                    max=None if z_max_nm is None else float(z_max_nm),
                )


def poisson_nll_counts(observed: torch.Tensor, expected: torch.Tensor, *, eps: float = 1e-6) -> torch.Tensor:
    expected_safe = expected.clamp_min(float(eps))
    return (expected_safe - observed * torch.log(expected_safe) + torch.lgamma(observed + 1.0)).mean()


def initialize_local_from_spots(
    spots: torch.Tensor,
    *,
    z_nm: float = 0.0,
    pixel_size_x_nm: float = 1.0,
    pixel_size_y_nm: float = 1.0,
    min_photons: float = 1.0,
    min_background: float = 1e-3,
) -> NATLocalInit:
    if spots.ndim != 3:
        raise ValueError("spots must have shape (N, H, W).")
    device = spots.device
    dtype = spots.dtype
    count = spots.shape[0]
    yy, xx = torch.meshgrid(
        torch.arange(spots.shape[1], device=device, dtype=dtype),
        torch.arange(spots.shape[2], device=device, dtype=dtype),
        indexing="ij",
    )
    weights = spots.clamp_min(0.0)
    sums = weights.sum(dim=(-2, -1)).clamp_min(1e-6)
    center_x = (spots.shape[2] - 1.0) / 2.0
    center_y = (spots.shape[1] - 1.0) / 2.0
    x_px = (weights * xx).sum(dim=(-2, -1)) / sums - center_x
    y_px = (weights * yy).sum(dim=(-2, -1)) / sums - center_y
    rim = torch.cat([spots[:, 0], spots[:, -1], spots[:, 1:-1, 0], spots[:, 1:-1, -1]], dim=1)
    background = rim.median(dim=1).values.clamp_min(float(min_background))
    photons = (spots.sum(dim=(-2, -1)) - background * spots.shape[-1] * spots.shape[-2]).clamp_min(float(min_photons))
    return NATLocalInit(
        x_nm=(x_px * float(pixel_size_x_nm)).float(),
        y_nm=(y_px * float(pixel_size_y_nm)).float(),
        z_nm=torch.full((count,), float(z_nm), dtype=torch.float32, device=device),
        photons=photons.float(),
        background=background.float(),
    )


def render_nat_expected_counts(
    *,
    renderer: NATRenderer,
    nat_model: NATFieldModel,
    roixy_px: torch.Tensor,
    local: NATLocalParameters,
) -> torch.Tensor:
    observation = nat_model.forward_roi_positions(
        roixy_px.to(device=nat_model.gamma.device, dtype=nat_model.gamma.dtype),
        local_x_nm=local.x_nm,
        local_y_nm=local.y_nm,
    )
    signal = renderer(
        local.x_nm,
        local.y_nm,
        local.z_nm,
        local.photons,
        observation.coefficients_nm,
    )
    return signal + local.background[:, None, None]


def render_nat_expected_counts_direct(
    *,
    renderer: NATRenderer,
    config: NATFieldConfig,
    gamma: torch.Tensor,
    roixy_px: torch.Tensor,
    x_nm: torch.Tensor,
    y_nm: torch.Tensor,
    z_nm: torch.Tensor,
    photons: torch.Tensor,
    background: torch.Tensor,
) -> torch.Tensor:
    coefficients, _, _ = evaluate_zernike_from_roi_positions_torch(
        roixy_px.to(device=gamma.device, dtype=gamma.dtype),
        gamma,
        config,
        local_x_nm=x_nm,
        local_y_nm=y_nm,
        dtype=gamma.dtype,
        device=gamma.device,
    )
    signal = renderer(x_nm, y_nm, z_nm, photons, coefficients)
    return signal + background[:, None, None]


def _solve_lm_step(grad: torch.Tensor, hessian: torch.Tensor, damping: float) -> torch.Tensor:
    diag = torch.diag(hessian).abs().clamp_min(1e-6)
    system = hessian + float(damping) * torch.diag(diag)
    rhs = -grad
    try:
        return torch.linalg.solve(system, rhs)
    except RuntimeError:
        return torch.linalg.pinv(system) @ rhs


def _is_cuda_oom_error(error: RuntimeError) -> bool:
    return "out of memory" in str(error).lower()


def _expand_per_emitter_values(
    value: torch.Tensor | float | None,
    *,
    count: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor | None:
    if value is None:
        return None
    flat = torch.as_tensor(value, device=device, dtype=dtype).reshape(-1)
    if flat.numel() == 0:
        return None
    if flat.numel() == 1:
        return flat.expand(count)
    if flat.numel() >= count:
        return flat[:count]
    pad = flat[-1:].expand(count - flat.numel())
    return torch.cat([flat, pad], dim=0)


def _clamp_theta_batch(
    theta: torch.Tensor,
    *,
    photon_min: torch.Tensor,
    photon_max: torch.Tensor,
    background_min: torch.Tensor,
    background_max: torch.Tensor,
    z_min_nm: float | None,
    z_max_nm: float | None,
    x_bound_nm: float | None,
    y_bound_nm: float | None,
) -> torch.Tensor:
    out = theta.clone()
    if x_bound_nm is not None:
        out[:, 0] = out[:, 0].clamp(-float(x_bound_nm), float(x_bound_nm))
    if y_bound_nm is not None:
        out[:, 1] = out[:, 1].clamp(-float(y_bound_nm), float(y_bound_nm))
    if z_min_nm is not None or z_max_nm is not None:
        out[:, 2] = out[:, 2].clamp(
            min=None if z_min_nm is None else float(z_min_nm),
            max=None if z_max_nm is None else float(z_max_nm),
        )
    out[:, 3] = out[:, 3].clamp(photon_min, photon_max)
    out[:, 4] = out[:, 4].clamp(background_min, background_max)
    return out


def _solve_lm_step_batch(grad: torch.Tensor, hessian: torch.Tensor, damping: torch.Tensor) -> torch.Tensor:
    diag = torch.diagonal(hessian, dim1=-2, dim2=-1).abs().clamp_min(1e-6)
    system = hessian + damping[:, None, None] * torch.diag_embed(diag)
    rhs = -grad.unsqueeze(-1)
    if hasattr(torch.linalg, "solve_ex"):
        solution, info = torch.linalg.solve_ex(system, rhs, check_errors=False)
        step = solution.squeeze(-1)
        failed = info != 0
        if bool(failed.any().item()):
            step = step.clone()
            step[failed] = (torch.linalg.pinv(system[failed]) @ rhs[failed]).squeeze(-1)
        return step
    try:
        return torch.linalg.solve(system, rhs).squeeze(-1)
    except RuntimeError:
        return (torch.linalg.pinv(system) @ rhs).squeeze(-1)


def _local_loss_batch(
    theta_value: torch.Tensor,
    spots: torch.Tensor,
    roixy_px: torch.Tensor,
    *,
    renderer: NATRenderer,
    config: NATFieldConfig,
    gamma: torch.Tensor,
) -> torch.Tensor:
    expected = render_nat_expected_counts_direct(
        renderer=renderer,
        config=config,
        gamma=gamma,
        roixy_px=roixy_px,
        x_nm=theta_value[:, 0],
        y_nm=theta_value[:, 1],
        z_nm=theta_value[:, 2],
        photons=theta_value[:, 3],
        background=theta_value[:, 4],
    )
    expected_safe = expected.clamp_min(1e-6)
    loss = expected_safe - spots * torch.log(expected_safe) + torch.lgamma(spots + 1.0)
    return loss.mean(dim=(-2, -1))


def _local_grad_hessian_batch(
    theta_value: torch.Tensor,
    spots: torch.Tensor,
    roixy_px: torch.Tensor,
    *,
    renderer: NATRenderer,
    config: NATFieldConfig,
    gamma: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    theta_req = theta_value.detach().clone().requires_grad_(True)
    losses = _local_loss_batch(
        theta_req,
        spots,
        roixy_px,
        renderer=renderer,
        config=config,
        gamma=gamma,
    )
    grad = torch.autograd.grad(losses.sum(), theta_req, create_graph=True)[0]
    hessian_columns: list[torch.Tensor] = []
    for param_index in range(theta_req.shape[1]):
        hessian_column = torch.autograd.grad(
            grad[:, param_index].sum(),
            theta_req,
            retain_graph=param_index + 1 < theta_req.shape[1],
        )[0]
        hessian_columns.append(hessian_column)
    hessian = torch.stack(hessian_columns, dim=-1)
    return grad.detach(), hessian.detach()


def fit_local_parameters(
    *,
    spots: torch.Tensor,
    roixy_px: torch.Tensor,
    renderer: NATRenderer,
    nat_model: NATFieldModel,
    local: NATLocalParameters,
    steps: int,
    lr: float,
    z_lr_multiplier: float = 1.0,
    min_photons: torch.Tensor | float | None = None,
    min_background: torch.Tensor | float | None = None,
    z_min_nm: float | None = None,
    z_max_nm: float | None = None,
) -> list[float]:
    base_lr = float(lr)
    optimizer = torch.optim.Adam(
        [
            {"params": [local.x_nm, local.y_nm, local.log_photons, local.log_background], "lr": base_lr},
            {"params": [local.z_nm], "lr": base_lr * max(float(z_lr_multiplier), 0.0)},
        ]
    )
    losses: list[float] = []
    target = spots.to(device=nat_model.gamma.device, dtype=nat_model.gamma.dtype)
    for _ in range(max(0, int(steps))):
        expected = render_nat_expected_counts(renderer=renderer, nat_model=nat_model, roixy_px=roixy_px, local=local)
        loss = poisson_nll_counts(target, expected)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        local.clamp_min_(
            min_photons=min_photons,
            min_background=min_background,
            z_min_nm=z_min_nm,
            z_max_nm=z_max_nm,
        )
        losses.append(float(loss.detach().cpu()))
    return losses


def fit_local_parameters_lm(
    *,
    spots: torch.Tensor,
    roixy_px: torch.Tensor,
    renderer: NATRenderer,
    config: NATFieldConfig,
    gamma: torch.Tensor,
    local: NATLocalParameters,
    steps: int,
    damping_init: float = 1.0,
    damping_factor: float = 10.0,
    min_photons: torch.Tensor | float | None = None,
    min_background: torch.Tensor | float | None = None,
    z_min_nm: float | None = None,
    z_max_nm: float | None = None,
    x_bound_nm: float | None = None,
    y_bound_nm: float | None = None,
    chunk_size: int | None = None,
) -> list[float]:
    target = spots.to(device=gamma.device, dtype=gamma.dtype)
    roi_all = roixy_px.to(device=gamma.device, dtype=gamma.dtype)
    theta_all = torch.stack(
        [
            local.x_nm.detach().to(device=gamma.device, dtype=gamma.dtype),
            local.y_nm.detach().to(device=gamma.device, dtype=gamma.dtype),
            local.z_nm.detach().to(device=gamma.device, dtype=gamma.dtype),
            local.photons.detach().to(device=gamma.device, dtype=gamma.dtype),
            local.background.detach().to(device=gamma.device, dtype=gamma.dtype),
        ],
        dim=1,
    )
    count = int(target.shape[0])
    if count == 0:
        return []
    if chunk_size is not None and int(chunk_size) <= 0:
        raise ValueError("chunk_size must be positive when provided.")

    photon_min_all = theta_all[:, 3].detach().clamp_min(1e-6) / 10.0
    photon_max_all = theta_all[:, 3].detach().clamp_min(1e-6) * 2.0
    background_min_all = theta_all[:, 4].detach().clamp_min(1e-6) / 10.0
    background_max_all = torch.maximum(
        theta_all[:, 3].detach().clamp_min(1e-6) / float(2 * target.shape[-1] * target.shape[-2]),
        theta_all[:, 4].detach().clamp_min(1e-6) * 2.0,
    )
    photon_floor = _expand_per_emitter_values(min_photons, count=count, device=gamma.device, dtype=gamma.dtype)
    background_floor = _expand_per_emitter_values(min_background, count=count, device=gamma.device, dtype=gamma.dtype)
    if photon_floor is not None:
        photon_min_all = torch.maximum(photon_min_all, photon_floor)
    if background_floor is not None:
        background_min_all = torch.maximum(background_min_all, background_floor)

    gamma_static = gamma.detach()

    current_loss_all = torch.empty((count,), device=gamma.device, dtype=gamma.dtype)
    damping_all = torch.full((count,), float(damping_init), device=gamma.device, dtype=gamma.dtype)
    requested_chunk = count if chunk_size is None else min(int(chunk_size), count)
    auto_chunk = chunk_size is None
    start = 0
    while start < count:
        active_chunk = min(requested_chunk, count - start)
        while True:
            end = start + active_chunk
            theta_chunk = theta_all[start:end].detach().clone()
            spot_chunk = target[start:end]
            roi_chunk = roi_all[start:end]
            damping_chunk = damping_all[start:end].detach().clone()
            photon_min_chunk = photon_min_all[start:end]
            photon_max_chunk = photon_max_all[start:end]
            background_min_chunk = background_min_all[start:end]
            background_max_chunk = background_max_all[start:end]
            try:
                current_loss_chunk = _local_loss_batch(
                    theta_chunk,
                    spot_chunk,
                    roi_chunk,
                    renderer=renderer,
                    config=config,
                    gamma=gamma_static,
                ).detach()
                for _ in range(max(0, int(steps))):
                    grad_chunk, hessian_chunk = _local_grad_hessian_batch(
                        theta_chunk,
                        spot_chunk,
                        roi_chunk,
                        renderer=renderer,
                        config=config,
                        gamma=gamma_static,
                    )
                    step_chunk = _solve_lm_step_batch(grad_chunk, hessian_chunk, damping_chunk)
                    trial_chunk = _clamp_theta_batch(
                        theta_chunk + step_chunk,
                        photon_min=photon_min_chunk,
                        photon_max=photon_max_chunk,
                        background_min=background_min_chunk,
                        background_max=background_max_chunk,
                        z_min_nm=z_min_nm,
                        z_max_nm=z_max_nm,
                        x_bound_nm=x_bound_nm,
                        y_bound_nm=y_bound_nm,
                    )
                    trial_loss_chunk = _local_loss_batch(
                        trial_chunk,
                        spot_chunk,
                        roi_chunk,
                        renderer=renderer,
                        config=config,
                        gamma=gamma_static,
                    ).detach()
                    accepted = trial_loss_chunk <= current_loss_chunk
                    theta_chunk = torch.where(accepted[:, None], trial_chunk.detach(), theta_chunk)
                    current_loss_chunk = torch.where(accepted, trial_loss_chunk, current_loss_chunk)
                    damping_chunk = torch.where(
                        accepted,
                        (damping_chunk / float(damping_factor)).clamp_min(1e-9),
                        damping_chunk * float(damping_factor),
                    )
                theta_all[start:end] = theta_chunk.detach()
                current_loss_all[start:end] = current_loss_chunk
                damping_all[start:end] = damping_chunk.detach()
                start = end
                break
            except RuntimeError as error:
                if not auto_chunk or not _is_cuda_oom_error(error) or active_chunk == 1:
                    raise
                if gamma.device.type == "cuda":
                    torch.cuda.empty_cache()
                active_chunk = max(1, active_chunk // 2)
                requested_chunk = active_chunk
    losses = [float(value) for value in current_loss_all.detach().cpu().tolist()]

    with torch.no_grad():
        local.x_nm.data = theta_all[:, 0].to(device=local.x_nm.device, dtype=local.x_nm.dtype)
        local.y_nm.data = theta_all[:, 1].to(device=local.y_nm.device, dtype=local.y_nm.dtype)
        local.z_nm.data = theta_all[:, 2].to(device=local.z_nm.device, dtype=local.z_nm.dtype)
        local.log_photons.data = torch.log(theta_all[:, 3].to(device=local.log_photons.device, dtype=local.log_photons.dtype).clamp_min(1e-6))
        local.log_background.data = torch.log(theta_all[:, 4].to(device=local.log_background.device, dtype=local.log_background.dtype).clamp_min(1e-6))
    return losses


def fit_global_gamma(
    *,
    spots: torch.Tensor,
    roixy_px: torch.Tensor,
    renderer: NATRenderer,
    nat_model: NATFieldModel,
    local: NATLocalParameters,
    steps: int,
    lr: float,
    gamma_l2: float = 0.0,
    gamma_train_mask: torch.Tensor | None = None,
    gamma_fixed_values: torch.Tensor | None = None,
    gamma_positive_index: int | None = None,
    gamma_symmetry_vector: torch.Tensor | None = None,
    max_gamma_norm_nm: float | None = None,
) -> list[float]:
    optimizer = torch.optim.Adam([nat_model.gamma], lr=float(lr))
    losses: list[float] = []
    target = spots.to(device=nat_model.gamma.device, dtype=nat_model.gamma.dtype)
    train_mask = None
    fixed_values = None
    if gamma_train_mask is not None:
        train_mask = gamma_train_mask.to(device=nat_model.gamma.device, dtype=torch.bool).reshape(-1)
        if train_mask.shape[0] != nat_model.gamma.shape[0]:
            raise ValueError("gamma_train_mask length must match nat_model.gamma.")
    if gamma_fixed_values is not None:
        fixed_values = gamma_fixed_values.to(device=nat_model.gamma.device, dtype=nat_model.gamma.dtype).reshape(-1)
        if fixed_values.shape[0] != nat_model.gamma.shape[0]:
            raise ValueError("gamma_fixed_values length must match nat_model.gamma.")
    if train_mask is not None and fixed_values is not None:
        nat_model.gamma.data[~train_mask] = fixed_values[~train_mask]
    for _ in range(max(0, int(steps))):
        expected = render_nat_expected_counts(renderer=renderer, nat_model=nat_model, roixy_px=roixy_px, local=local)
        loss = poisson_nll_counts(target, expected)
        if float(gamma_l2) > 0.0:
            loss = loss + float(gamma_l2) * torch.mean(nat_model.gamma.square())
        optimizer.zero_grad()
        loss.backward()
        if train_mask is not None and nat_model.gamma.grad is not None:
            nat_model.gamma.grad[~train_mask] = 0.0
        optimizer.step()
        if train_mask is not None and fixed_values is not None:
            nat_model.gamma.data[~train_mask] = fixed_values[~train_mask]
        project_gamma_norm_(nat_model.gamma, max_norm_nm=max_gamma_norm_nm, train_mask=train_mask)
        if gamma_positive_index is not None:
            enforce_gamma_positive_gauge_(
                nat_model.gamma,
                local,
                positive_index=int(gamma_positive_index),
                symmetry_vector=gamma_symmetry_vector,
            )
        losses.append(float(loss.detach().cpu()))
    return losses


def fit_global_gamma_lm(
    *,
    spots: torch.Tensor,
    roixy_px: torch.Tensor,
    renderer: NATRenderer,
    nat_model: NATFieldModel,
    local: NATLocalParameters,
    steps: int,
    damping_init: float = 1e3,
    damping_factor: float = 10.0,
    gamma_l2: float = 0.0,
    gamma_train_mask: torch.Tensor | None = None,
    gamma_fixed_values: torch.Tensor | None = None,
    gamma_positive_index: int | None = None,
    gamma_symmetry_vector: torch.Tensor | None = None,
    max_gamma_norm_nm: float | None = None,
) -> list[float]:
    target = spots.to(device=nat_model.gamma.device, dtype=nat_model.gamma.dtype)
    train_mask = torch.ones_like(nat_model.gamma, dtype=torch.bool)
    if gamma_train_mask is not None:
        train_mask = gamma_train_mask.to(device=nat_model.gamma.device, dtype=torch.bool).reshape(-1)
    fixed_values = None if gamma_fixed_values is None else gamma_fixed_values.to(device=nat_model.gamma.device, dtype=nat_model.gamma.dtype).reshape(-1)
    if fixed_values is not None:
        nat_model.gamma.data[~train_mask] = fixed_values[~train_mask]
    train_values = nat_model.gamma.detach()[train_mask].clone()
    damping = float(damping_init)
    losses: list[float] = []

    def full_gamma(values: torch.Tensor) -> torch.Tensor:
        gamma = nat_model.gamma.detach().clone()
        gamma[train_mask] = values
        if fixed_values is not None:
            gamma[~train_mask] = fixed_values[~train_mask]
        return gamma

    initial_gamma = full_gamma(train_values)
    project_gamma_norm_(initial_gamma, max_norm_nm=max_gamma_norm_nm, train_mask=train_mask)
    train_values = initial_gamma[train_mask].clone()

    def loss_fn(values: torch.Tensor) -> torch.Tensor:
        gamma = full_gamma(values)
        expected = render_nat_expected_counts_direct(
            renderer=renderer,
            config=nat_model.config,
            gamma=gamma,
            roixy_px=roixy_px.to(device=gamma.device, dtype=gamma.dtype),
            x_nm=local.x_nm.detach().to(device=gamma.device, dtype=gamma.dtype),
            y_nm=local.y_nm.detach().to(device=gamma.device, dtype=gamma.dtype),
            z_nm=local.z_nm.detach().to(device=gamma.device, dtype=gamma.dtype),
            photons=local.photons.detach().to(device=gamma.device, dtype=gamma.dtype),
            background=local.background.detach().to(device=gamma.device, dtype=gamma.dtype),
        )
        loss = poisson_nll_counts(target, expected)
        if float(gamma_l2) > 0.0:
            loss = loss + float(gamma_l2) * torch.mean(gamma.square())
        return loss

    current_loss = loss_fn(train_values)
    for _ in range(max(0, int(steps))):
        values_req = train_values.detach().clone().requires_grad_(True)
        loss = loss_fn(values_req)
        grad = torch.autograd.grad(loss, values_req, create_graph=True)[0]
        hessian = torch.autograd.functional.hessian(loss_fn, values_req)
        step = _solve_lm_step(grad.detach(), hessian.detach(), damping)
        trial = train_values + step
        trial_gamma = full_gamma(trial)
        project_gamma_norm_(trial_gamma, max_norm_nm=max_gamma_norm_nm, train_mask=train_mask)
        trial = trial_gamma[train_mask]
        trial_loss = loss_fn(trial)
        if float(trial_loss.detach().item()) <= float(current_loss.detach().item()):
            train_values = trial.detach()
            current_loss = trial_loss.detach()
            damping = max(damping / float(damping_factor), 1e-9)
        else:
            damping = damping * float(damping_factor)
        losses.append(float(current_loss.detach().cpu()))

    with torch.no_grad():
        nat_model.gamma.data = full_gamma(train_values).to(device=nat_model.gamma.device, dtype=nat_model.gamma.dtype)
        if gamma_positive_index is not None:
            enforce_gamma_positive_gauge_(
                nat_model.gamma,
                local,
                positive_index=int(gamma_positive_index),
                symmetry_vector=gamma_symmetry_vector,
            )
    return losses


def enforce_gamma_positive_gauge_(
    gamma: torch.Tensor,
    local: NATLocalParameters,
    *,
    positive_index: int,
    symmetry_vector: torch.Tensor | None,
) -> bool:
    index = int(positive_index)
    if index < 0 or index >= gamma.shape[0]:
        raise ValueError("positive_index must refer to a gamma coefficient.")
    if float(gamma.detach()[index].item()) >= 0.0:
        return False
    if symmetry_vector is None:
        raise ValueError("symmetry_vector is required for positive gamma gauge enforcement.")
    vector = symmetry_vector.to(device=gamma.device, dtype=gamma.dtype).reshape(-1)
    if vector.shape[0] != gamma.shape[0]:
        raise ValueError("symmetry_vector length must match gamma.")
    with torch.no_grad():
        gamma.data = gamma.data * vector
        local.z_nm.data = -local.z_nm.data
    return True


def project_gamma_norm_(
    gamma: torch.Tensor,
    *,
    max_norm_nm: float | None,
    train_mask: torch.Tensor | None = None,
) -> bool:
    if max_norm_nm is None:
        return False
    limit = float(max_norm_nm)
    if limit <= 0.0:
        raise ValueError("max_gamma_norm_nm must be positive when configured.")
    mask = torch.ones_like(gamma, dtype=torch.bool)
    if train_mask is not None:
        mask = train_mask.to(device=gamma.device, dtype=torch.bool).reshape(-1)
        if mask.shape != gamma.shape:
            raise ValueError("train_mask shape must match gamma.")
    norm = torch.linalg.vector_norm(gamma.detach()[mask])
    if float(norm.item()) <= limit:
        return False
    with torch.no_grad():
        gamma.data[mask] = gamma.data[mask] * (limit / norm)
    return True


def fit_nat_alternating(
    *,
    spots: torch.Tensor,
    roixy_px: torch.Tensor,
    renderer: NATRenderer,
    config: NATFieldConfig | None = None,
    gamma_init: torch.Tensor | None = None,
    local_init: NATLocalInit | None = None,
    rounds: int = 3,
    local_steps: int = 20,
    global_steps: int = 20,
    local_warmup_rounds: int = 0,
    local_warmup_steps: int = 0,
    optimizer_kind: str = "adam",
    local_lr: float = 1e-2,
    local_z_lr_multiplier: float = 1.0,
    gamma_lr: float = 1e-3,
    gamma_l2: float = 0.0,
    gamma_train_mask: torch.Tensor | None = None,
    gamma_fixed_values: torch.Tensor | None = None,
    gamma_positive_index: int | None = None,
    gamma_symmetry_vector: torch.Tensor | None = None,
    max_gamma_norm_nm: float | None = None,
    min_photons: torch.Tensor | float | None = None,
    min_background: torch.Tensor | float | None = None,
    z_min_nm: float | None = None,
    z_max_nm: float | None = None,
    x_bound_nm: float | None = None,
    y_bound_nm: float | None = None,
    local_lm_chunk_size: int | None = None,
    progress_callback: Callable[[dict[str, int | float | str]], None] | None = None,
) -> NATFitSnapshot:
    if spots.ndim != 3:
        raise ValueError("spots must have shape (N, H, W).")
    if roixy_px.ndim != 2 or roixy_px.shape != (spots.shape[0], 2):
        raise ValueError("roixy_px must have shape (N, 2), matching spots.")

    cfg = default_order1_config() if config is None else config
    device = spots.device
    nat_model = NATFieldModel(cfg, gamma_init=gamma_init, device=device, dtype=spots.dtype)
    local = NATLocalParameters(initialize_local_from_spots(spots) if local_init is None else local_init).to(device=device)
    local.clamp_min_(min_photons=min_photons, min_background=min_background, z_min_nm=z_min_nm, z_max_nm=z_max_nm)
    history: list[float] = []
    if gamma_train_mask is not None and gamma_fixed_values is not None:
        train_mask = gamma_train_mask.to(device=device, dtype=torch.bool).reshape(-1)
        fixed_values = gamma_fixed_values.to(device=device, dtype=spots.dtype).reshape(-1)
        nat_model.gamma.data[~train_mask] = fixed_values[~train_mask]
    if gamma_positive_index is not None:
        enforce_gamma_positive_gauge_(
            nat_model.gamma,
            local,
            positive_index=int(gamma_positive_index),
            symmetry_vector=gamma_symmetry_vector,
        )

    use_lm = str(optimizer_kind).strip().lower() == "lm"
    warmup_round_count = max(0, int(local_warmup_rounds))
    warmup_step_count = int(local_warmup_steps) if int(local_warmup_steps) > 0 else int(local_steps)
    def report_progress(phase: str, round_index: int, round_count: int) -> None:
        if progress_callback is None:
            return
        latest_loss = float(history[-1]) if history else 0.0
        progress_callback(
            {
                "phase": str(phase),
                "round_index": int(round_index),
                "round_count": int(round_count),
                "loss": latest_loss,
            }
        )

    for warmup_index in range(warmup_round_count):
        if use_lm:
            history.extend(
                fit_local_parameters_lm(
                    spots=spots,
                    roixy_px=roixy_px,
                    renderer=renderer,
                    config=cfg,
                    gamma=nat_model.gamma.detach(),
                    local=local,
                    steps=warmup_step_count,
                    min_photons=min_photons,
                    min_background=min_background,
                    z_min_nm=z_min_nm,
                    z_max_nm=z_max_nm,
                    x_bound_nm=x_bound_nm,
                    y_bound_nm=y_bound_nm,
                    chunk_size=local_lm_chunk_size,
                )
            )
        else:
            history.extend(
                fit_local_parameters(
                    spots=spots,
                    roixy_px=roixy_px,
                    renderer=renderer,
                    nat_model=nat_model,
                    local=local,
                    steps=warmup_step_count,
                    lr=local_lr,
                    z_lr_multiplier=local_z_lr_multiplier,
                    min_photons=min_photons,
                    min_background=min_background,
                    z_min_nm=z_min_nm,
                    z_max_nm=z_max_nm,
                )
            )
        report_progress("warmup", warmup_index + 1, warmup_round_count)
    round_count = max(0, int(rounds))
    for round_index in range(round_count):
        if use_lm:
            history.extend(
                fit_local_parameters_lm(
                    spots=spots,
                    roixy_px=roixy_px,
                    renderer=renderer,
                    config=cfg,
                    gamma=nat_model.gamma.detach(),
                    local=local,
                    steps=local_steps,
                    min_photons=min_photons,
                    min_background=min_background,
                    z_min_nm=z_min_nm,
                    z_max_nm=z_max_nm,
                    x_bound_nm=x_bound_nm,
                    y_bound_nm=y_bound_nm,
                    chunk_size=local_lm_chunk_size,
                )
            )
            history.extend(
                fit_global_gamma_lm(
                    spots=spots,
                    roixy_px=roixy_px,
                    renderer=renderer,
                    nat_model=nat_model,
                    local=local,
                    steps=global_steps,
                    gamma_l2=gamma_l2,
                    gamma_train_mask=gamma_train_mask,
                    gamma_fixed_values=gamma_fixed_values,
                    gamma_positive_index=gamma_positive_index,
                    gamma_symmetry_vector=gamma_symmetry_vector,
                    max_gamma_norm_nm=max_gamma_norm_nm,
                )
            )
        else:
            history.extend(
                fit_local_parameters(
                    spots=spots,
                    roixy_px=roixy_px,
                    renderer=renderer,
                    nat_model=nat_model,
                    local=local,
                    steps=local_steps,
                    lr=local_lr,
                    z_lr_multiplier=local_z_lr_multiplier,
                    min_photons=min_photons,
                    min_background=min_background,
                    z_min_nm=z_min_nm,
                    z_max_nm=z_max_nm,
                )
            )
            history.extend(
                fit_global_gamma(
                    spots=spots,
                    roixy_px=roixy_px,
                    renderer=renderer,
                    nat_model=nat_model,
                    local=local,
                    steps=global_steps,
                    lr=gamma_lr,
                    gamma_l2=gamma_l2,
                    gamma_train_mask=gamma_train_mask,
                    gamma_fixed_values=gamma_fixed_values,
                    gamma_positive_index=gamma_positive_index,
                    gamma_symmetry_vector=gamma_symmetry_vector,
                    max_gamma_norm_nm=max_gamma_norm_nm,
                )
            )
        report_progress("alternating", round_index + 1, round_count)

    return NATFitSnapshot(
        x_nm=local.x_nm.detach().clone(),
        y_nm=local.y_nm.detach().clone(),
        z_nm=local.z_nm.detach().clone(),
        photons=local.photons.detach().clone(),
        background=local.background.detach().clone(),
        gamma=nat_model.gamma.detach().clone(),
        loss_history=tuple(history),
    )


__all__ = [
    "NATFitSnapshot",
    "NATLocalInit",
    "NATLocalParameters",
    "NATRenderer",
    "fit_global_gamma",
    "fit_global_gamma_lm",
    "fit_local_parameters",
    "fit_local_parameters_lm",
    "fit_nat_alternating",
    "enforce_gamma_positive_gauge_",
    "initialize_local_from_spots",
    "poisson_nll_counts",
    "project_gamma_norm_",
    "render_nat_expected_counts",
]
