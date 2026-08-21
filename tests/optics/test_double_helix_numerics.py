from __future__ import annotations

import pytest
import torch

from unity_psf.optics.psf.double_helix.numerics import (
    legendre_polynomial,
    normalized_cross_correlation,
)


@pytest.mark.parametrize(
    ("degree", "expected"),
    [
        (0, [1.0, 1.0, 1.0]),
        (1, [-0.5, 0.0, 0.5]),
        (2, [-0.125, -0.5, -0.125]),
        (3, [0.4375, 0.0, -0.4375]),
    ],
)
def test_legendre_polynomial_preserves_values_dtype_device_and_gradients(
    degree: int,
    expected: list[float],
) -> None:
    values = torch.tensor([-0.5, 0.0, 0.5], dtype=torch.float64, requires_grad=True)

    result = legendre_polynomial(degree, values)

    assert result.dtype == values.dtype
    assert result.device == values.device
    torch.testing.assert_close(result, torch.tensor(expected, dtype=torch.float64))
    if degree == 0:
        assert not result.requires_grad
    else:
        result.sum().backward()
        assert values.grad is not None
        assert torch.isfinite(values.grad).all()


def test_legendre_polynomial_rejects_negative_degree() -> None:
    with pytest.raises(ValueError, match="degree must be non-negative"):
        legendre_polynomial(-1, torch.ones(1))


def test_normalized_cross_correlation_uses_last_two_spatial_dimensions() -> None:
    first = torch.tensor(
        [[[1.0, 2.0], [3.0, 4.0]], [[4.0, 3.0], [2.0, 1.0]]],
        dtype=torch.float64,
        requires_grad=True,
    )
    second = first.detach().flip(-1).requires_grad_()

    result = normalized_cross_correlation(first, second)

    assert result.shape == (2,)
    torch.testing.assert_close(result, torch.tensor([0.6, 0.6], dtype=torch.float64))
    result.sum().backward()
    assert first.grad is not None
    assert second.grad is not None


def test_normalized_cross_correlation_returns_zero_for_constant_inputs() -> None:
    result = normalized_cross_correlation(torch.ones(2, 3, 3), torch.ones(2, 3, 3))

    torch.testing.assert_close(result, torch.zeros(2))


def test_normalized_cross_correlation_preserves_nan_and_empty_spatial_behavior() -> None:
    nan_result = normalized_cross_correlation(
        torch.tensor([[[float("nan")]]]),
        torch.ones(1, 1, 1),
    )
    empty_result = normalized_cross_correlation(torch.empty(1, 0, 3), torch.empty(1, 0, 3))

    assert torch.isnan(nan_result).all()
    torch.testing.assert_close(empty_result, torch.zeros(1))
