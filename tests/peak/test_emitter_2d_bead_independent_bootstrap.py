from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
import tifffile

from unity_psf.optics import build_named_nat_config
from unity_psf.peak.contract import PeakBootstrapConfig
from unity_psf.peak.export import _nat_config_from_summary
from unity_psf.peak.harvest import run_peak_harvest
from unity_psf.peak.nat_optimizer import project_gamma_norm_
from unity_psf.peak.run_peak_nat_zmap import peak_config_from_mapping
from unity_psf.peak.vector_nat_fit import initial_gamma_from_zernike_coefficients


CONFIG_DIR = Path(__file__).parents[2] / "configs" / "calibration"
PROJECT_ROOT = Path(__file__).parents[2]


def _load(channel_id: str):
    path = CONFIG_DIR / f"origami_emitter_2d_{channel_id}_peak_nat.json"
    return path, json.loads(path.read_text(encoding="utf-8"))


@pytest.mark.parametrize(
    ("channel_id", "crop"),
    (("left", [0, 600]), ("right", [600, 1200])),
)
def test_origami_bootstrap_is_sample_driven_and_bead_independent(
    channel_id: str,
    crop: list[int],
) -> None:
    path, payload = _load(channel_id)
    config = peak_config_from_mapping(payload)

    source = str(config.tiff_path).lower()
    serialized = path.read_text(encoding="utf-8").lower()
    assert "origami_2d/spool_2d-640nm100mw" in source
    assert source.endswith("_mmstack_default.ome.tif")
    assert "bead" not in serialized
    assert "psf_aligned" not in serialized
    assert "zernike_maps_nm.npz" not in serialized
    assert [config.crop_x0, config.crop_x1] == crop
    assert [config.crop_y0, config.crop_y1] == [0, 1200]
    assert config.local_z_range_nm == (0.0, 0.0)
    assert config.vectorfit_astig_gauge is False
    assert config.vectorfit_phasor_z_init is False


def test_explicit_zero_emitter_anchor_initializes_conventional_psf_gamma() -> None:
    _, payload = _load("left")
    config = peak_config_from_mapping(payload)
    nat_config = build_named_nat_config(
        config.nat_config_kind,
        img_size_x=600,
        img_size_y=1200,
        pixel_size_x_nm=101.11,
        pixel_size_y_nm=98.83,
    )

    gamma = initial_gamma_from_zernike_coefficients(config, nat_config)

    assert gamma.shape == (len(nat_config.gammas),)
    assert torch.count_nonzero(gamma).item() == 0
    assert config.initial_zernike_coefficients_nm == {
        "2,0": 0.0,
        "2,2": 0.0,
        "2,-2": 0.0,
    }


@pytest.mark.parametrize("channel_id", ("left", "right"))
def test_origami_emitter_nat_uses_diffraction_limited_gamma_constraint(channel_id: str) -> None:
    _, payload = _load(channel_id)
    config = peak_config_from_mapping(payload)

    assert config.max_gamma_norm_nm == pytest.approx(47.0)
    assert config.nat_config_kind == "order1"


def test_gamma_projection_limits_only_trainable_field_terms() -> None:
    gamma = torch.tensor([12.0, 3000.0, 4000.0])
    train_mask = torch.tensor([False, True, True])

    projected = project_gamma_norm_(gamma, max_norm_nm=47.0, train_mask=train_mask)

    assert projected is True
    assert gamma[0].item() == pytest.approx(12.0)
    assert torch.linalg.vector_norm(gamma[train_mask]).item() == pytest.approx(47.0, abs=1e-5)


def test_export_uses_explicit_peak_nat_model_over_stale_diagnostics_summary() -> None:
    _, payload = _load("left")
    config = peak_config_from_mapping(payload)

    nat_config = _nat_config_from_summary(
        summary={"config": {"nat_config_kind": "order1_21"}},
        peak_config=config,
        width=600,
        height=1200,
    )

    assert len(nat_config.gammas) == 8


def test_initial_anchor_rejects_modes_without_a_constant_nat_term() -> None:
    _, payload = _load("left")
    payload["initial_zernike_coefficients_nm"] = {"3,1": 12.0}
    config = peak_config_from_mapping(payload)
    nat_config = build_named_nat_config("order1", img_size_x=600, img_size_y=1200)

    with pytest.raises(ValueError, match="constant NAT term"):
        initial_gamma_from_zernike_coefficients(config, nat_config)


def test_origami_peak_nat_slurm_runs_both_channels_on_independent_gpus() -> None:
    script = (
        PROJECT_ROOT / "scripts" / "train" / "unitypsf_origami_emitter_2d_peak_nat_2gpu.sbatch"
    ).read_text(encoding="utf-8")
    lowered = script.lower()

    assert "#SBATCH --gres=gpu:2" in script
    assert "origami_emitter_2d_left_peak_nat.json" in script
    assert "origami_emitter_2d_right_peak_nat.json" in script
    assert "origami-emitter-2d-left" in script
    assert "origami-emitter-2d-right" in script
    assert 'CUDA_VISIBLE_DEVICES="$LEFT_GPU"' in script
    assert 'CUDA_VISIBLE_DEVICES="$RIGHT_GPU"' in script
    assert "torch.cuda.is_available()" in script
    assert "calibration refuses CPU fallback" in script
    assert "allow-cpu" not in lowered
    assert "bead" not in lowered
    assert "psf_aligned" not in lowered


def test_peak_harvest_rejects_patches_crossing_channel_crop_boundary(tmp_path: Path) -> None:
    stack = np.zeros((1, 32, 64), dtype=np.float32)
    stack[0, 16, 16] = 1000.0
    stack[0, 16, 30] = 900.0
    tiff_path = tmp_path / "dual_channel.tif"
    tifffile.imwrite(tiff_path, stack)
    config = PeakBootstrapConfig(
        sample="synthetic",
        side="left",
        frame_range=(0, 1),
        tiff_path=tiff_path,
        crop_x0=0,
        crop_x1=32,
        crop_y0=0,
        crop_y1=32,
        max_emitters=10,
        max_candidates=10,
        min_distance_px=1.0,
        threshold_sigma=1.0,
        patch_size_px=7,
    )

    result = run_peak_harvest(config=config, output_dir=tmp_path / "harvest")
    harvest = torch.load(result["harvest_path"], map_location="cpu", weights_only=False)

    assert result["kept_count"] == 1
    assert harvest["payload"]["x_px"].tolist() == pytest.approx([16.5])
