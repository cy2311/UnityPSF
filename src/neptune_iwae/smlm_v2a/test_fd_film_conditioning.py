import copy

import torch

from neptune.smlm_v2a.fd_conditioning.film_patch import patch_model_for_fd_film_conditioning
from neptune.smlm_v2a.fd_conditioning.fullres import FullResZernikeConditioning
from neptune.smlm_v2a.models.smlm_model import DoubleUNet


def test_fd_film_patch_preserves_initial_double_unet_output():
    torch.manual_seed(0)
    base_model = DoubleUNet(
        nch_in=3,
        depth_shared=1,
        depth_union=1,
        nfeatures_init=8,
        nfeatures_inter=8,
        norm_start_level=-1,
        norm_groups=-1,
        dropout_start_level=None,
        p_dropout=0.0,
        depthwise=False,
    )
    patched_model = copy.deepcopy(base_model)
    patched_model = patch_model_for_fd_film_conditioning(patched_model, condition_dim=8, hidden_dim=16)
    x = torch.randn(2, 3, 40, 40)
    cond = torch.randn(2, 8)

    with torch.no_grad():
        baseline = base_model(x)
        conditioned = patched_model([x, cond])
        fallback = patched_model(x)

    assert torch.allclose(conditioned, baseline, atol=1e-6, rtol=1e-6)
    assert torch.allclose(fallback, baseline, atol=1e-6, rtol=1e-6)


def test_fullres_condition_vector_contains_xy_and_coeff_summary():
    full_maps = torch.stack(
        [
            torch.full((4, 4), 10.0),
            torch.full((4, 4), 12.0),
            torch.full((4, 4), 14.0),
            torch.full((4, 4), 16.0),
            torch.full((4, 4), 18.0),
            torch.full((4, 4), 20.0),
        ],
        dim=0,
    )
    provider = FullResZernikeConditioning(
        full_maps_nm=full_maps,
        mode_order=[(2, 0), (3, 1), (3, -1), (4, 0), (3, -3), (3, 3)],
    )
    cond = provider.condition_vector_from_xy(x0=0, y0=0, height=2, width=2)

    assert cond.shape == (8,)
    assert torch.allclose(cond[:2], torch.tensor([-2.0 / 3.0, -2.0 / 3.0]), atol=1e-6, rtol=1e-6)
    assert torch.allclose(cond[2:], torch.zeros(6), atol=1e-6, rtol=1e-6)
