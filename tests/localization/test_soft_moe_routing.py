from __future__ import annotations

import torch

from unity_psf.localization.model import build_localization_model_registry


def _params() -> dict[str, object]:
    return {
        "nch_in": 3,
        "condition_dim": 4,
        "domain_count": 2,
        "film_hidden_dim": 4,
        "depth_shared": 1,
        "depth_union": 1,
        "nfeatures_init": 4,
        "nfeatures_inter": 4,
        "norm_start_level": -1,
        "norm_groups": 0,
        "activation": "ELU",
        "dropout_start_level": None,
        "p_dropout": -0.1,
        "pool_mode": "StrideConv",
        "upsample_mode": "nearest",
        "inter_activation": None,
        "norm_head_groups": 0,
        "final_activation": None,
        "kaiming_normal": True,
        "depthwise": True,
        "z_mu_activation": "tanh",
    }


def test_soft_moe_combines_conditioned_experts() -> None:
    torch.manual_seed(7)
    model = build_localization_model_registry()["active_smlm_soft_moe_double_unet"](_params())
    model.eval()
    images = torch.randn(4, 3, 8, 8)
    conditions = torch.tensor(
        [[0.1, -0.2, 1.0, 0.0], [0.2, 0.3, 0.0, 1.0], [-0.4, 0.5, 1.0, 0.0], [0.6, -0.1, 0.0, 1.0]],
        dtype=torch.float32,
    )

    with torch.no_grad():
        actual = model((images, conditions))
        weights = model.gate_weights(conditions)
        expected = sum(
            expert((images, conditions)) * weights[:, index].view(-1, 1, 1, 1)
            for index, expert in enumerate(model.experts)
        )

    assert torch.allclose(actual, expected)
    assert torch.allclose(weights.sum(dim=1), torch.ones(images.shape[0]))


def test_soft_moe_initial_gate_routes_by_domain_and_remains_trainable() -> None:
    conditions = torch.tensor(
        [[-0.8, 0.6, 1.0, 0.0], [0.7, -0.5, 0.0, 1.0]],
        dtype=torch.float32,
    )
    for seed in (0, 1, 2, 3):
        torch.manual_seed(seed)
        model = build_localization_model_registry()["active_smlm_soft_moe_double_unet"](_params())
        weights = model.gate_weights(conditions)

        assert weights[0, 0] > 0.95
        assert weights[1, 1] > 0.95

        model.zero_grad()
        weights[:, 0].sum().backward()
        assert model.gate[0].weight.grad is not None
        assert model.gate[0].weight.grad.abs().sum() > 0
        assert model.gate[-1].weight.grad is not None
        assert model.gate[-1].weight.grad.abs().sum() > 0
