from __future__ import annotations

import torch
from torch import distributions


class LocalSetGMMLoss:
    def __init__(self, eps: float = 1e-8):
        self.eps = float(eps)

    def __call__(self, output: dict[str, torch.Tensor], target: torch.Tensor, mask: torch.BoolTensor) -> torch.Tensor:
        prob = output["prob"]
        mu = output["mu"]
        sig = output["sig"]
        batch_size, num_slots = prob.shape
        eps = self.eps

        nlocs_mean = prob.sum(-1)
        nlocs_var = torch.clamp((prob - prob**2).sum(-1), min=eps)
        nlocs_tar = mask.sum(-1).float()
        log_prob = distributions.Normal(loc=nlocs_mean, scale=nlocs_var.sqrt()).log_prob(nlocs_tar) * torch.clamp(
            nlocs_tar, min=1.0
        )

        prob_flat = torch.clamp(prob, min=0.0)
        prob_flat = torch.nan_to_num(prob_flat, nan=0.0, posinf=0.0, neginf=0.0)
        mix_logits = torch.log(prob_flat + eps)
        mix = distributions.Categorical(logits=mix_logits)
        comp = distributions.Independent(
            distributions.Normal(mu, torch.clamp(sig, min=eps)),
            1,
        )
        gmm = distributions.MixtureSameFamily(mix, comp)

        if bool(mask.any()):
            target = target.to(device=prob.device, dtype=prob.dtype)
            max_targets = target.shape[1]
            log_gmm_acc = torch.zeros((batch_size,), device=prob.device, dtype=prob.dtype)
            for start in range(0, max_targets, max_targets):
                end = min(max_targets, start + max_targets)
                mask_chunk = mask[:, start:end]
                if not bool(mask_chunk.any()):
                    continue
                tar_chunk = target[:, start:end]
                log_chunk = gmm.log_prob(tar_chunk.transpose(0, 1)).transpose(0, 1)
                log_gmm_acc = log_gmm_acc + (log_chunk * mask_chunk.float()).sum(-1)
            log_prob = log_prob + log_gmm_acc

        return -log_prob

