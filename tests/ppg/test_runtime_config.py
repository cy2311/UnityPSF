from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from ppg.infer_recon_sample import _load_model
from ppg.lunar_background import LunarBackgroundEstimate
from ppg.spatial_gamma import coefficient_field_from_gamma
from ppg.train_dh_physical import (
    apply_lunar_background_estimate,
    resolve_training_epochs,
)
from unity_psf.data.normalization import LunarInputNormalization
from unity_psf.localization import build_localization_runtime_config
from unity_psf.models.psf_moe.experts.double_helix import DoubleHelixExpert


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs/modalities/double_helix/ppg_physical.yaml"
ARTIFACT_PATH = ROOT / "configs/calibration/ppg/ppg_map_initial.npz"


def test_default_runtime_uses_only_the_ppg_map() -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    online = config["train"]["online_generation"]
    model_params = config["train"]["model"]["params"]
    physical_update = config["train"]["ppg_physical_update"]

    runtime = build_localization_runtime_config(
        config,
        config_base_dir=CONFIG_PATH.parent,
        seed=4311,
    )
    provider = runtime["batch_provider"]["params"]
    model = runtime["model"]["params"]

    assert runtime["batch_provider"]["name"] == "online_train_batch"
    assert runtime["loss"]["name"] == "active_smlm_gmm_loss"
    assert runtime["loss"]["params"]["target_order"] == "legacy_iwae"
    assert runtime["resolved_contract"]["model"]["output"] == "smlm_10ch"
    assert provider["simulation_backend"] == "ppg_direct"
    assert provider["psf_type"] == "ppg_direct"
    assert Path(provider["ppg_map_path"]).is_file()
    assert provider["vector_psf_size"] == 21
    assert provider["dual_domain_coeff_maps"] == ()
    assert provider["pupil_carrier_complex"] is None
    assert provider["conditioning_mode"] == "film"
    assert provider["condition_feature_dim"] == 6
    assert provider["condition_dim"] == 6
    assert provider["condition_fields"] == (
        "field_x", "field_y", "ppg_pca_0", "ppg_pca_1", "ppg_pca_2", "ppg_pca_3"
    )
    assert model["input_normalization"] == {
        "name": "lunar",
        "background_range": [110.0, 110.0],
        "photon_range": [0.0, 31000.0],
    }
    assert model["nch_in"] == 3
    assert model["depth_shared"] == 2
    assert model["depth_union"] == 2
    assert model_params["input_normalization"] == "lunar"
    assert "input_offset" not in model
    assert "input_scale" not in model
    assert "gain" not in physical_update
    assert "sample_signal_gain" not in physical_update
    assert "baseline_percentile" not in physical_update
    assert "target_background" not in physical_update
    assert physical_update["background_estimation"] == {
        "method": "lunar_gaussian_dark_pixels",
        "percentile": 10.0,
        "max_size_gb": 2.0,
        "chunk_frames": 64,
        "camera_backward": "identity",
    }
    assert not {
        "wavelength_nm",
        "NA",
        "refmed",
        "refcov",
        "refimm",
        "npupil",
        "vector_batch_size",
        "zemit0",
        "domain_count",
        "append_domain_onehot",
    }.intersection(online)


def test_training_uses_sample_background_for_simulation_and_normalization() -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    estimate = LunarBackgroundEstimate(
        background_range=(99.0, 115.0),
        frame_range=(0, 4096),
        percentile=10.0,
    )

    apply_lunar_background_estimate(config, estimate)

    assert config["train"]["online_generation"]["background_range"] == [99.0, 115.0]
    resolved = config["train"]["ppg_physical_update"]["resolved_background"]
    assert resolved == {
        "background_range": [99.0, 115.0],
        "background_mean": 107.0,
        "frame_range": [0, 4096],
        "percentile": 10.0,
        "camera_backward": "identity",
    }


def _small_dh_expert(*, lunar: bool) -> DoubleHelixExpert:
    return DoubleHelixExpert(
        nch_in=3,
        depth_shared=1,
        depth_union=1,
        nfeatures_init=4,
        nfeatures_inter=4,
        input_normalization={
            "name": "lunar",
            "background_range": [110.0, 110.0],
            "photon_range": [0.0, 31000.0],
        }
        if lunar
        else None,
    )


def test_dh_runtime_applies_lunar_input_normalization_and_records_it() -> None:
    model = _small_dh_expert(lunar=True)
    metadata = model.checkpoint_metadata()
    captured: list[torch.Tensor] = []
    handle = model.network.register_forward_pre_hook(
        lambda _module, args: captured.append(args[0].detach().clone())
    )

    raw = torch.full((1, 3, 8, 8), 110.0)
    raw[:, :, :, 4:] = 730.0
    model(raw)
    handle.remove()

    torch.testing.assert_close(captured[0][:, :, :, :4], torch.zeros((1, 3, 8, 4)))
    torch.testing.assert_close(captured[0][:, :, :, 4:], torch.ones((1, 3, 8, 4)))
    assert metadata.model_config["input_normalization"] == {
        "name": "lunar",
        "background_range": [110.0, 110.0],
        "photon_range": [0.0, 31000.0],
    }


def test_ppg_checkpoint_reload_restores_input_normalization(tmp_path: Path) -> None:
    source = _small_dh_expert(lunar=True)
    checkpoint = tmp_path / "double_helix.resume.ckpt"
    torch.save(
        {
            "training_state": {"double_helix": {"epoch": 7}},
            "experts": {
                "double_helix": {
                    "model_config": dict(source.checkpoint_metadata().model_config),
                    "model_state_dict": source.state_dict(),
                }
            },
        },
        checkpoint,
    )

    restored, epoch = _load_model(checkpoint, torch.device("cpu"))

    assert epoch == 7
    assert restored.input_normalization == LunarInputNormalization(
        background_range=(110.0, 110.0),
        photon_range=(0.0, 31000.0),
    )


def test_ppg_inference_rejects_checkpoint_without_lunar_normalization(
    tmp_path: Path,
) -> None:
    source = _small_dh_expert(lunar=False)
    checkpoint = tmp_path / "legacy.ckpt"
    torch.save(
        {
            "training_state": {"double_helix": {"epoch": 7}},
            "experts": {
                "double_helix": {
                    "model_config": dict(source.checkpoint_metadata().model_config),
                    "model_state_dict": source.state_dict(),
                }
            },
        },
        checkpoint,
    )

    with pytest.raises(RuntimeError, match="LUNAR input normalization"):
        _load_model(checkpoint, torch.device("cpu"))


def test_training_epochs_have_one_config_source_with_optional_override() -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))

    assert resolve_training_epochs(config, override=None) == 300
    assert resolve_training_epochs(config, override=12) == 12


def test_default_artifact_is_a_self_consistent_gamma36_map() -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    with np.load(ARTIFACT_PATH, allow_pickle=False) as payload:
        assert str(payload["schema"].item()) == "ppg.pca_map.v1"
        assert payload["gamma"].shape == (9, 4)
        assert payload["roi_shape_yx"].tolist() == [21, 21]
        assert payload["coefficient_field"].shape == (4, 40, 40)
        assert float(payload["boundary_anchor_strength"]) == 1.0
        assert (
            config["train"]["ppg_physical_update"]["gamma_boundary_anchor_strength"]
            == 1.0
        )
        np.testing.assert_allclose(
            payload["coefficient_field"],
            coefficient_field_from_gamma(
                payload["gamma"], payload["xy_grid_normalized"]
            ),
            rtol=1e-7,
            atol=1e-9,
        )


def test_slurm_entrypoints_keep_logs_out_of_artifact_directories() -> None:
    train_script = (ROOT / "scripts/ppg/train_ppg_physical_300.sbatch").read_text(
        encoding="utf-8"
    )
    infer_script = (ROOT / "scripts/ppg/infer_recon_after_training.sbatch").read_text(
        encoding="utf-8"
    )

    assert "#SBATCH --output=logs/ppg_phys300-%j.out" in train_script
    assert "#SBATCH --output=logs/ppg_infer300-%j.out" in infer_script
    assert "ROOT=\"$(cd \"$SCRIPT_DIR/../..\" && pwd)\"" in train_script
    assert "$ROOT/configs/modalities/double_helix/ppg_physical.yaml" in train_script
    assert "$ROOT/ppg/train_dh_physical.py" not in train_script
    assert "gmm-physical-300-lunar-bg10-${SLURM_JOB_ID}" in train_script
    assert 'RUN_ID="${RUN_ID:-ppg-dh-gmm-physical-300-lunar-bg10-${SLURM_JOB_ID}}"' in train_script
    assert 'if [[ "${RESUME:-0}" == "1" ]]' in train_script
    assert 'RESUME_ARGS+=(--resume)' in train_script
    assert "gmm-physical-300-lunar-bg10-${TRAIN_JOB_ID}" in infer_script
    assert "--baseline-percentile" not in infer_script
    assert "--target-background" not in infer_script
