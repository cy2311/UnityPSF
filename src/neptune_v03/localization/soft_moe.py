from __future__ import annotations

import copy

import torch
import torch.nn as nn

from neptune_v03.localization.film import FiLMConditionedDoubleUNet, split_conditioned_input
from neptune_v03.localization.smlm_unet import DoubleUNet


def domain_index_from_condition(condition: torch.Tensor, *, zernike_dim: int, domain_count: int) -> torch.Tensor:
    if condition.dim() != 2:
        raise ValueError(f"Expected condition as (N,C), got {tuple(condition.shape)}")
    if condition.shape[1] < int(zernike_dim) + int(domain_count):
        raise ValueError(
            f"Expected condition with zernike_dim + domain_count = {int(zernike_dim) + int(domain_count)}, "
            f"got {condition.shape[1]}"
        )
    onehot = condition[:, int(zernike_dim) : int(zernike_dim) + int(domain_count)]
    return torch.argmax(onehot, dim=1).to(dtype=torch.long)


class SoftMoEFiLMExperts(nn.Module):
    def __init__(
        self,
        *,
        base_model: DoubleUNet,
        condition_dim: int,
        domain_count: int = 2,
        hidden_dim: int = 32,
    ) -> None:
        super().__init__()
        if not isinstance(base_model, DoubleUNet):
            raise TypeError(f"base_model must be DoubleUNet, got {type(base_model)!r}")
        self.condition_dim = int(condition_dim)
        self.domain_count = int(domain_count)
        self.zernike_dim = self.condition_dim - self.domain_count
        if self.domain_count <= 0:
            raise ValueError("domain_count must be positive")
        if self.zernike_dim <= 0:
            raise ValueError("condition_dim must include zernike condition plus domain one-hot")
        self.nch_in = int(base_model.nch_in)
        self.experts = nn.ModuleList(
            [
                FiLMConditionedDoubleUNet.from_base(
                    copy.deepcopy(base_model),
                    condition_dim=self.condition_dim,
                    hidden_dim=int(hidden_dim),
                    copy_base=False,
                )
                for _ in range(self.domain_count)
            ]
        )
        gate_hidden = max(int(hidden_dim), self.condition_dim, 8)
        self.gate = nn.Sequential(
            nn.Linear(self.condition_dim, gate_hidden),
            nn.SiLU(),
            nn.Linear(gate_hidden, self.domain_count),
        )
        self._init_gate_near_domain_routing()
        self.hard_route_until_epoch = -1
        self.current_epoch = 0
        self.gate_entropy_weight = 0.0
        self.gate_supervision_weight = 0.0

    def _init_gate_near_domain_routing(self) -> None:
        final = self.gate[-1]
        if not isinstance(final, nn.Linear):
            return
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)
        with torch.no_grad():
            for domain in range(self.domain_count):
                col = self.zernike_dim + domain
                if col < final.weight.shape[1]:
                    final.weight[domain, col] = 4.0

    def gate_weights(self, condition: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.gate(condition), dim=1)

    def gate_regularization_loss(self, condition: torch.Tensor) -> torch.Tensor:
        weights = self.gate_weights(condition)
        eps = torch.finfo(weights.dtype).eps
        entropy = -(weights * torch.log(weights.clamp_min(eps))).sum(dim=1).mean()
        domain_idx = domain_index_from_condition(condition, zernike_dim=self.zernike_dim, domain_count=self.domain_count)
        supervised = torch.nn.functional.nll_loss(torch.log(weights.clamp_min(eps)), domain_idx.to(weights.device))
        return float(self.gate_entropy_weight) * entropy + float(self.gate_supervision_weight) * supervised

    def forward(self, x) -> torch.Tensor:
        parameter = next(self.parameters())
        image, condition = split_conditioned_input(
            x,
            image_channels=self.nch_in,
            condition_dim=self.condition_dim,
            device=parameter.device,
            dtype=parameter.dtype,
        )
        image = image.to(device=parameter.device, dtype=parameter.dtype)
        condition = condition.to(device=parameter.device, dtype=parameter.dtype)
        if int(self.current_epoch) < int(self.hard_route_until_epoch):
            domain_idx = domain_index_from_condition(
                condition,
                zernike_dim=self.zernike_dim,
                domain_count=self.domain_count,
            ).to(image.device)
            weights = torch.nn.functional.one_hot(domain_idx, num_classes=self.domain_count).to(dtype=image.dtype)
        else:
            weights = self.gate_weights(condition)

        output = None
        for expert_idx, expert in enumerate(self.experts):
            expert_output = expert((image, condition))
            weight = weights[:, expert_idx].view(weights.shape[0], *([1] * (expert_output.dim() - 1)))
            contribution = expert_output * weight
            output = contribution if output is None else output + contribution
        if output is None:
            raise RuntimeError("No expert outputs produced")
        return output
