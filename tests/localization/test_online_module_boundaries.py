from __future__ import annotations

import ast
from dataclasses import fields
from importlib import import_module
from pathlib import Path

import pytest

from unity_psf.localization.data import online
from unity_psf.localization.data.online_cached import CachedWindowEpochBatches
from unity_psf.localization.data.online_cached_sequences import CachedSequenceStore
from unity_psf.localization.data.online_provider import build_online_batch_provider
from unity_psf.localization.data.online_targets import target_contract_for_config
from unity_psf.localization.data.online_types import OnlineBatchProviderConfig


def test_online_config_keeps_complete_field_schema() -> None:
    expected = """
        batch_size channels height width emitters_per_sample seed steps_per_epoch
        background signal simulation_backend psf_type pixel_size_nm_x pixel_size_nm_y
        wavelength_nm na refmed refcov refimm objstage0 otf_rescale_xy npupil
        vector_psf_size vector_batch_size zemit0 emitter_density_um2 lifetime_avg
        warmup_frames photon_range photon_mean photon_sigma background_range
        background_scale z_range conditioning_mode nat_simulation_mode nat_grid_size
        nat_grid_z_steps append_domain_onehot condition_feature_dim condition_dim
        condition_fields conditioning_profile domain_count domain_balance_mode
        dual_domain_coeff_maps fixed_astig_anchor_nm circular_psf_radius_nm
        pupil_carrier_complex batch_strategy sequence_window_chunks sequence_count
        camera_qe camera_spurious_charge camera_baseline camera_e_per_adu
        camera_readout pxyz_target_order photon_scale z_scale z_scale_unit
        simulation_output_device cached_window_order cached_window_max_gpu_sequences
        cached_window_cache_epochs cached_window_cache_frames cached_window_cache_storage
        cached_window_prefetch_batches ppg_map_path field_origin_sampling_mode
        field_origin_stride_px empirical_psf_path empirical_psf_channel
        empirical_psf_focus_index camera_read_sigma
    """.split()
    assert [field.name for field in fields(OnlineBatchProviderConfig)] == expected
    assert OnlineBatchProviderConfig(batch_size=1).z_scale_unit is None
    assert OnlineBatchProviderConfig.__dataclass_params__.frozen


def test_explicit_legacy_exports_resolve_to_canonical_owners() -> None:
    assert online.OnlineBatchProviderConfig is OnlineBatchProviderConfig
    assert online.build_online_batch_provider is build_online_batch_provider
    assert online._CachedWindowEpochBatches is CachedWindowEpochBatches
    for name in online.__all__:
        exported = getattr(online, name)
        owner = import_module(exported.__module__)
        assert owner is not online
        assert getattr(owner, exported.__name__) is exported
    assert CachedWindowEpochBatches._sequence is CachedSequenceStore.sequence
    assert CachedWindowEpochBatches._refresh_condition_state is CachedSequenceStore.refresh_condition_state


def test_canonical_modules_do_not_import_facade_or_private_online_interfaces() -> None:
    directory = Path(online.__file__).parent
    for path in directory.glob("online_*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if not isinstance(node, ast.ImportFrom):
                continue
            module = node.module or ""
            assert module not in {"online", "unity_psf.localization.data.online"}, path.name
            if module.startswith("online_"):
                assert all(not alias.name.startswith("_") for alias in node.names), path.name
    assert len(Path(online.__file__).read_text().splitlines()) <= 180
    for filename in ("online_native.py", "online_cached.py"):
        assert len((directory / filename).read_text().splitlines()) <= 450


@pytest.mark.parametrize("strategy", ["triplet", "sequence_window", "cached_window"])
@pytest.mark.parametrize("z_scale,z_unit,z_nm", [(0.7, "um", 700.0), (700.0, "nm", 700.0)])
@pytest.mark.parametrize("precompute", ["0", "1"])
def test_online_scale_contract_survives_strategy_split(
    monkeypatch: pytest.MonkeyPatch, strategy: str, z_scale: float, z_unit: str, z_nm: float, precompute: str,
) -> None:
    monkeypatch.setenv("UNITY_V04_CACHED_WINDOW_PRECOMPUTE", precompute)
    config = OnlineBatchProviderConfig(
        batch_size=1,
        height=8,
        width=8,
        emitters_per_sample=1,
        seed=29,
        simulation_backend="native",
        conditioning_profile="zero",
        conditioning_mode="film",
        npupil=8,
        vector_psf_size=5,
        vector_batch_size=1,
        batch_strategy=strategy,
        sequence_count=1,
        photon_range=(100.0, 500.0),
        z_range=(-0.3, 0.3),
        photon_scale=1500.0,
        z_scale=z_scale,
        z_scale_unit=z_unit,
    )
    contract = target_contract_for_config(config).as_dict()
    assert contract["z_scale_nm"] == z_nm
    batches = list(build_online_batch_provider(config)(1))
    assert batches[0].inputs.metadata["target_contract"] == contract
