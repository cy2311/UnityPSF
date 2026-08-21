from __future__ import annotations

import pytest

from unity_psf.infer_recon.filter.filter import (
    _first_optional_float,
    _optional_float,
    _psf_xy_nm_from_row,
)


@pytest.mark.parametrize(
    ("value", "expected"),
    [(None, None), ("", None), ("nan", None), ("inf", None), ("3.5", 3.5)],
)
def test_quality_field_parsers_keep_missing_and_nonfinite_semantics(value, expected) -> None:
    assert _optional_float(value) == expected


def test_quality_field_parsers_preserve_alias_precedence_and_anisotropic_psf() -> None:
    row = {"PSFxnm": "120", "PSFynm": "80", "LLrel": "2.5"}
    assert _first_optional_float(row, ("llrel", "LLrel")) == 2.5
    assert _psf_xy_nm_from_row(row) == pytest.approx((120.0**2 / 2.0 + 80.0**2 / 2.0) ** 0.5)
