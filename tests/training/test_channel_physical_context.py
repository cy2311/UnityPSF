from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from unity_psf.contracts.modality import ExpertInstanceSpec, MeasurementChannelSpec
from unity_psf.localization.conditioning import ConditioningProviderStore
from unity_psf.runtime.layout import ensure_run_layout, write_run_manifest
from unity_psf.training.channel_context import (
    ASTIGMATISM_660NM_ANCHOR_PROFILE,
    ChannelTrainingContext,
    atomic_write_json,
    sha256_file,
)
from unity_psf.training.high_fidelity.peak_bootstrap import peak_bootstrap_config as _peak_bootstrap_config, single_channel_peak_domain as _single_channel_peak_domain
from unity_psf.optics.profiles import resolve_astigmatism_anchor_profile


def _runtime_config(*, crop: tuple[int, int, int, int] | None = None) -> dict[str, object]:
    return {
        "expert_instance": {
            "expert_type": "astigmatism",
            "instance_id": "main",
            "channel_id": "main",
            "prototype_ref": None,
        },
        "input_frame_spec": {"input_frame_channels": 3, "frame_order": "temporal"},
        "channel_layout": {
            "frame_size": [32, 32],
            "channels": [
                {
                    "channel_id": "main",
                    "crop": None if crop is None else list(crop),
                    "anchor_profile": None,
                    "calibration_ref": None,
                }
            ]
        },
    }


def _coeff_map(path: Path, value: float) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        zernike_maps_nm=np.full((6, 8, 8), value, dtype=np.float32),
        mode_order=np.asarray([[2, 0], [3, 1], [3, -1], [4, 0], [3, -3], [3, 3]], dtype=np.int64),
    )
    return path


def test_context_has_explicit_single_instance_defaults_and_channel_crop(tmp_path: Path) -> None:
    layout = ensure_run_layout(tmp_path, "run")

    context = ChannelTrainingContext.from_runtime_config(
        _runtime_config(crop=(12, 4, 8, 8)),
        layout=layout,
    )

    assert context.instance == ExpertInstanceSpec("astigmatism", "main", "main")
    assert context.channel == MeasurementChannelSpec("main", crop=(12, 4, 8, 8))
    assert context.raw_crop == (12, 4, 8, 8)
    assert context.anchor_profile == ASTIGMATISM_660NM_ANCHOR_PROFILE.name
    assert context.physical_state_path == layout.metadata_dir / "current_physical_state.json"
    assert isinstance(context.condition_store, ConditioningProviderStore)
    assert context.condition_store.snapshot()[1] is None


def test_context_owns_one_condition_store_and_updates_only_that_instance(tmp_path: Path) -> None:
    layout = ensure_run_layout(tmp_path, "run")
    first = _coeff_map(tmp_path / "first.npz", 1.0)
    second = _coeff_map(tmp_path / "second.npz", 2.0)
    config = _runtime_config()
    config["batch_provider"] = {"params": {"dual_domain_coeff_maps": [{"name": "main", "coeff_maps_npz": str(first)}]}}

    context = ChannelTrainingContext.from_runtime_config(config, layout=layout)

    version_before = context.condition_store.version
    context.update_coefficient_map(second)

    version, providers = context.condition_store.snapshot()
    assert version == version_before + 1
    assert providers is not None
    assert len(providers) == 1
    assert providers[0][0] == "main"
    assert context.coefficient_map_path == second


def test_physical_state_manifest_records_initial_and_latest_hash(tmp_path: Path) -> None:
    layout = ensure_run_layout(tmp_path, "run")
    write_run_manifest(layout, {"stage": "high_fidelity_localization"})
    first = _coeff_map(tmp_path / "first.npz", 1.0)
    second = _coeff_map(tmp_path / "second.npz", 2.0)
    context = ChannelTrainingContext.from_runtime_config(_runtime_config(), layout=layout)

    context.update_coefficient_map(first)
    context.write_physical_state(source="initial")
    initial_manifest = json.loads((layout.metadata_dir / "run_manifest.json").read_text(encoding="utf-8"))
    initial_hash = initial_manifest["initial_physical_state_hash"]
    assert initial_hash == initial_manifest["latest_physical_state_hash"]

    context.update_coefficient_map(second)
    context.write_physical_state(source="gamma_feedback", condition_store_version=1)
    latest_manifest = json.loads((layout.metadata_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert latest_manifest["initial_physical_state_hash"] == initial_hash
    assert latest_manifest["latest_physical_state_hash"] != initial_hash
    assert latest_manifest["current_physical_state"]["condition_store_version"] == 1


def test_atomic_write_keeps_previous_state_when_replace_is_interrupted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "state.json"
    atomic_write_json(path, {"version": 1})
    previous = path.read_bytes()

    def fail_replace(_source: str | bytes | os.PathLike[str], _target: str | bytes | os.PathLike[str]) -> None:
        raise OSError("simulated interruption")

    monkeypatch.setattr(os, "replace", fail_replace)
    with pytest.raises(OSError, match="simulated interruption"):
        atomic_write_json(path, {"version": 2})

    assert path.read_bytes() == previous
    assert json.loads(path.read_text(encoding="utf-8")) == {"version": 1}


def test_context_restores_checkpoint_map_and_normalizes_channel_binding(tmp_path: Path) -> None:
    layout = ensure_run_layout(tmp_path, "run")
    write_run_manifest(layout, {"initial_physical_state_hash": "original"})
    coeff = _coeff_map(tmp_path / "restored.npz", 3.0)
    context = ChannelTrainingContext.from_runtime_config(_runtime_config(), layout=layout)
    state = {
        "schema_version": "unitypsf.channel_physical_state.v1",
        "source": "gamma_feedback",
        "condition_store_version": 4,
        "expert_instance": {"expert_type": "astigmatism", "instance_id": "main", "channel_id": "main"},
        "coeff_maps": [{"name": "main", "coeff_maps_npz": str(coeff), "sha256": sha256_file(coeff)}],
    }

    context.restore_physical_state(state)

    restored = json.loads(context.physical_state_path.read_text(encoding="utf-8"))
    assert restored["coeff_maps"][0]["name"] == "main"
    assert context.condition_store.snapshot()[1][0][0] == "main"
    assert context.initial_physical_state_hash == "original"


def test_context_rejects_raw_crop_outside_declared_frame(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="out of frame bounds|exceeds frame_size"):
        ChannelTrainingContext.from_runtime_config(
            _runtime_config(crop=(30, 0, 8, 8)),
            layout=ensure_run_layout(tmp_path, "invalid-crop"),
        )


def test_anchor_profile_accepts_existing_99nm_aliases() -> None:
    assert resolve_astigmatism_anchor_profile("astigmatism_660nm_anchor99").anchor_nm == 99.0
    assert resolve_astigmatism_anchor_profile("astigmatism_660nm_99nm").wavelength_nm == 660.0


def test_single_astigmatism_peak_bootstrap_selects_only_main_crop_and_profile() -> None:
    train_cfg = {
        "model": {"name": "astigmatism_expert"},
        "expert": {"name": "astigmatism", "channel_id": "main"},
        "channel_layout": {
            "frame_size": [64, 128],
            "measurement_channels": [{"id": "main", "crop": [40, 3, 16, 16]}],
        },
    }
    domain = _single_channel_peak_domain(
        train_cfg,
        [
            {"name": "left", "crop_left": 0, "crop_top": 0, "crop_width": 32, "crop_height": 32},
            {"name": "right", "crop_left": 32, "crop_top": 0, "crop_width": 32, "crop_height": 32},
        ],
    )

    assert domain == {"name": "main", "crop_left": 40, "crop_top": 3, "crop_width": 16, "crop_height": 16}
    peak_config = _peak_bootstrap_config(
        {},
        domain=domain,
        name="main",
        tiff_path=Path("raw.tif"),
        anchor_profile=ASTIGMATISM_660NM_ANCHOR_PROFILE,
    )
    assert peak_config.vectorfit_astig_anchor_nm == 99.0
