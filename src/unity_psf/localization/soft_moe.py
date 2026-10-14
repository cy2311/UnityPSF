from __future__ import annotations

import copy
import math

import torch
import torch.nn as nn

from unity_psf.localization.film import FiLMConditionedDoubleUNet, split_conditioned_input
from unity_psf.localization.smlm_unet import DoubleUNet


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

    def _init_gate_near_domain_routing(self) -> None:
        first = self.gate[0]
        final = self.gate[-1]
        assert isinstance(first, nn.Linear) and isinstance(final, nn.Linear)
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)
        with torch.no_grad():
            for domain in range(self.domain_count):
                first.weight[domain].zero_()
                first.bias[domain] = 0.0
                first.weight[domain, self.zernike_dim + domain] = 1.0
                final.weight[domain, domain] = 4.0 * (1.0 + math.exp(-1.0))

    def gate_weights(self, condition: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.gate(condition), dim=1)

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
        weights = self.gate_weights(condition)

        first_output = self.experts[0]((image, condition))
        first_weight = weights[:, 0].view(weights.shape[0], *([1] * (first_output.dim() - 1)))
        output = first_output * first_weight
        for expert_idx, expert in enumerate(self.experts[1:], start=1):
            expert_output = expert((image, condition))
            weight = weights[:, expert_idx].view(weights.shape[0], *([1] * (expert_output.dim() - 1)))
            output = output + expert_output * weight
        return output
