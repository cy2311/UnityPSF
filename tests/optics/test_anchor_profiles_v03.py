from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from unity_psf.cli.train_selection import build_plan, parse_routes
from unity_psf.localization.conditioning import build_astigmatism_anchor_conditioning_maps
from unity_psf.localization.data.online import (
    OnlineBatchProviderConfig,
    _default_condition_provider,
    build_online_batch_provider,
)
from unity_psf.optics.profiles import (
    CIRCULAR_2D_50NM_ANCHOR_PROFILE,
    ASTIGMATISM_660NM_ANCHOR_PROFILE,
)


def test_default_anchor_profiles_are_explicit_and_physical() -> None:
    assert CIRCULAR_2D_50NM_ANCHOR_PROFILE.name == "circular_2d_50nm"
    assert CIRCULAR_2D_50NM_ANCHOR_PROFILE.radius_nm == 50.0
    assert ASTIGMATISM_660NM_ANCHOR_PROFILE.anchor_nm == 100.0

    circular = _default_condition_provider(
        OnlineBatchProviderConfig(
            batch_size=1,
            width=8,
            height=8,
            conditioning_profile="circular_2d_50nm",
        )
    )
    assert circular.anchor_profile == "circular_2d_50nm"
    assert circular.anchor_radius_nm == 50.0
    assert torch.count_nonzero(circular.full_maps_nm) == 0

    astig = build_astigmatism_anchor_conditioning_maps(width=8, height=8)
    index = astig.mode_order.index((2, 2))
    assert astig.anchor_profile == "astigmatism_660nm"
    assert astig.anchor_nm == 100.0
    assert torch.all(astig.full_maps_nm[index] == 100.0)

    astig_config = OnlineBatchProviderConfig(
        batch_size=1,
        width=8,
        height=8,
        conditioning_profile="astigmatism_660nm",
        simulation_backend="native",
        npupil=8,
        vector_psf_size=5,
        vector_batch_size=1,
    )
    astig_batch = next(iter(build_online_batch_provider(astig_config)(1))).inputs
    assert astig_batch.metadata["anchor_nm"] == 100.0


def test_circular_anchor_is_recorded_in_simulator_metadata() -> None:
    config = OnlineBatchProviderConfig(
        batch_size=1,
        channels=3,
        width=8,
        height=8,
        emitters_per_sample=1,
        simulation_backend="native",
        npupil=8,
        vector_psf_size=5,
        vector_batch_size=1,
        conditioning_profile="circular_2d_50nm",
        circular_psf_radius_nm=50.0,
    )
    batch = next(iter(build_online_batch_provider(config)(1))).inputs
    assert batch.metadata["anchor_profile"] == "circular_2d_50nm"
    assert batch.metadata["circular_psf_radius_nm"] == 50.0
    assert batch.metadata["circular_psf_sigma_px_xy"] == [50.0 / 101.11, 50.0 / 98.83]


def test_circular_anchor_cannot_be_added_to_an_absolute_coefficient_map(tmp_path: Path) -> None:
    path = tmp_path / "absolute.npz"
    np.savez(
        path,
        zernike_maps_nm=np.zeros((6, 8, 8), dtype=np.float32),
        mode_order=np.asarray([(2, 0), (3, 1), (3, -1), (4, 0), (3, -3), (3, 3)]),
    )
    config = OnlineBatchProviderConfig(
        batch_size=1,
        conditioning_profile="circular_2d_50nm",
        circular_psf_radius_nm=50.0,
        dual_domain_coeff_maps=({"name": "left", "coeff_maps_npz": str(path)},),
        domain_count=1,
    )
    with pytest.raises(ValueError, match="circular_psf_radius_nm.*absolute coefficient maps"):
        build_online_batch_provider(config)


def test_cli_defaults_to_circular_2d_and_100nm_astigmatism_anchor(tmp_path: Path) -> None:
    plan = build_plan(
        routes=parse_routes(["emitter_2d:left", "astigmatism:right"]),
        maps={},
        dh_ppg_map=None,
        epochs=1,
        steps_per_epoch=1,
        batch_size=1,
        run_root=tmp_path,
        run_id="anchors",
        smoke=True,
    )

    emitter = plan["configs"]["emitter_2d:left"]
    emitter_online = emitter["train"]["online_generation"]
    assert emitter_online["dual_domain_coeff_maps"] == []
    assert emitter_online["conditioning_profile"] == "circular_2d_50nm"
    assert emitter_online["circular_psf_radius_nm"] == 50.0
    assert emitter["metadata"]["physical_initialization"] == "anchor_only"
    assert emitter["metadata"]["anchor_profile"] == "circular_2d_50nm"
    assert emitter["metadata"]["anchor_radius_nm"] == 50.0

    astig = plan["configs"]["astigmatism:right"]
    assert astig["train"]["online_generation"]["conditioning_profile"] == "astigmatism_660nm"
    assert astig["metadata"]["anchor_nm"] == 100.0
