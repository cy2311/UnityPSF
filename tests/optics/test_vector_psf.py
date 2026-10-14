from __future__ import annotations

import warnings

import torch

from unity_psf.optics.vector_psf import LocalVectorPSFTorchFit


def test_prechirpz_accepts_tensor_scale_without_copy_construction_warning() -> None:
    model = LocalVectorPSFTorchFit.__new__(LocalVectorPSFTorchFit)
    model.device = torch.device("cpu")
    model.complex_type = torch.complex64
    scale = torch.tensor(1.0, requires_grad=True)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        a, b, d = model.prechirpz(scale, 1.0, 4, 4)

    assert a.shape == (1, 4)
    assert b.shape == (1, 4)
    assert d.shape == (1, 7)
    assert not any("copy construct from a tensor" in str(item.message) for item in caught)
