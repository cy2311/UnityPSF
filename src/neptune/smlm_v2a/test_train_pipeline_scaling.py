from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

from neptune.smlm_v2a.models.smlm_model import DoubleUNet
from neptune.smlm_v2a.training.train_pipeline import _build_param, _resolve_residual_refinement_config


def test_build_param_respects_fixed_scaling_from_config(tmp_path):
    cfg = {
        "camera": {
            "baseline": 398.6,
            "qe": 0.9,
            "spurious_charge": 0.002,
            "e_per_adu": 1.020784562122306,
        },
        "optical": {
            "pixel_size_nm_x": 101.11,
            "pixel_size_nm_y": 98.83,
        },
        "simulation": {
            "background_uniform": [20, 200],
            "emitter": {
                "emitter_extent": [[-0.5, 63.5], [-0.5, 63.5], [-0.6, 0.6]],
                "z_range": [-0.6, 0.6],
            },
            "n_max": 96,
        },
        "train": {
            "scaling": {
                "input_offset": 516.111694,
                "input_scale": 521.062988,
                "photon_max": 31000.0,
                "z_max": 0.6,
                "bg_max": 240.0,
            }
        },
    }
    scaling_info = SimpleNamespace(
        photon_max=27332.0,
        z_max=0.5999,
        bg_max=211.0,
    )

    param = _build_param(
        cfg,
        scaling_info,
        output_dir=Path(tmp_path),
        device=torch.device("cpu"),
        epochs=300,
        batch_size=64,
        window_size=3,
    )

    assert param.Scaling.input_offset == 516.111694
    assert param.Scaling.input_scale == 521.062988
    assert param.Scaling.phot_max == 31000.0
    assert param.Scaling.z_max == 0.6
    assert param.Scaling.bkg_max == 240.0


def test_build_param_can_disable_depthwise_unet(tmp_path):
    cfg = {
        "camera": {
            "baseline": 398.6,
            "qe": 0.9,
            "spurious_charge": 0.002,
            "e_per_adu": 1.020784562122306,
        },
        "optical": {
            "pixel_size_nm_x": 101.11,
            "pixel_size_nm_y": 98.83,
        },
        "simulation": {
            "background_uniform": [20, 200],
            "emitter": {
                "emitter_extent": [[-0.5, 63.5], [-0.5, 63.5], [-0.6, 0.6]],
                "z_range": [-0.6, 0.6],
            },
            "n_max": 96,
        },
        "smlm_overrides": {
            "depthwise": False,
        },
    }
    scaling_info = SimpleNamespace(
        photon_max=31000.0,
        z_max=0.6,
        bg_max=240.0,
    )

    param = _build_param(
        cfg,
        scaling_info,
        output_dir=Path(tmp_path),
        device=torch.device("cpu"),
        epochs=300,
        batch_size=64,
        window_size=3,
    )

    model = DoubleUNet.parse(param)

    assert param.HyperParameter.arch_param.depthwise is False
    assert isinstance(model.unet_shared.encoder[0].conv1, nn.Conv2d)


def test_symmetric_z_range_uses_tanh_for_z_mu(tmp_path):
    cfg = {
        "camera": {
            "baseline": 398.6,
            "qe": 0.9,
            "spurious_charge": 0.002,
            "e_per_adu": 1.020784562122306,
        },
        "optical": {
            "pixel_size_nm_x": 101.11,
            "pixel_size_nm_y": 98.83,
        },
        "simulation": {
            "background_uniform": [20, 200],
            "emitter": {
                "emitter_extent": [[-0.5, 63.5], [-0.5, 63.5], [-0.6, 0.6]],
                "z_range": [-0.6, 0.6],
            },
            "n_max": 96,
        },
    }
    scaling_info = SimpleNamespace(
        photon_max=31000.0,
        z_max=0.6,
        bg_max=240.0,
    )

    param = _build_param(
        cfg,
        scaling_info,
        output_dir=Path(tmp_path),
        device=torch.device("cpu"),
        epochs=300,
        batch_size=64,
        window_size=3,
    )

    model = DoubleUNet.parse(param)

    assert 4 not in model.ch_ix_sigmoid
    assert 4 in model.ch_ix_tanh


def test_z_mu_defaults_to_tanh_even_for_positive_only_z_range(tmp_path):
    cfg = {
        "camera": {
            "baseline": 398.6,
            "qe": 0.9,
            "spurious_charge": 0.002,
            "e_per_adu": 1.020784562122306,
        },
        "optical": {
            "pixel_size_nm_x": 101.11,
            "pixel_size_nm_y": 98.83,
        },
        "simulation": {
            "background_uniform": [20, 200],
            "emitter": {
                "emitter_extent": [[-0.5, 63.5], [-0.5, 63.5], [0.0, 0.6]],
                "z_range": [0.0, 0.6],
            },
            "n_max": 96,
        },
    }
    scaling_info = SimpleNamespace(
        photon_max=31000.0,
        z_max=0.6,
        bg_max=240.0,
    )

    param = _build_param(
        cfg,
        scaling_info,
        output_dir=Path(tmp_path),
        device=torch.device("cpu"),
        epochs=300,
        batch_size=64,
        window_size=3,
    )

    model = DoubleUNet.parse(param)

    assert 4 not in model.ch_ix_sigmoid
    assert 4 in model.ch_ix_tanh


def test_resolve_residual_refinement_config_reads_enabled_settings():
    cfg = {
        "residual_refinement": {
            "enabled": True,
            "rounds": 3,
            "acceptance_threshold": 0.8,
            "training_render_source": "matched_gt",
            "inference_render_source": "prediction",
        }
    }

    rr_cfg = _resolve_residual_refinement_config(cfg)

    assert rr_cfg.enabled is True
    assert rr_cfg.rounds == 3
    assert rr_cfg.acceptance_threshold == 0.8
    assert rr_cfg.training_render_source == "matched_gt"
    assert rr_cfg.inference_render_source == "prediction"


def test_resolve_residual_refinement_config_defaults_to_disabled():
    rr_cfg = _resolve_residual_refinement_config({})

    assert rr_cfg.enabled is False
    assert rr_cfg.rounds == 3
    assert rr_cfg.acceptance_threshold == 0.8
    assert rr_cfg.training_render_source == "matched_gt"
    assert rr_cfg.inference_render_source == "prediction"
