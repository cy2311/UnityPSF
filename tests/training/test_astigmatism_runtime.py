from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import torch
import pytest
import yaml

from unity_psf.contracts.modality import PSFModality
from unity_psf.localization.model import build_localization_model_registry
from unity_psf.localization.runtime_config import (
    build_localization_runtime_config,
    resolve_localization_model_config,
)
from unity_psf.models.psf_moe.experts.astigmatism import AstigmatismExpert
from unity_psf.runtime.layout import ensure_run_layout
from unity_psf.training.loop import TrainingConfig, train_one_epoch
from unity_psf.training.runtime import build_trainer_runtime


CONFIG_PATH = Path(__file__).parents[2] / "configs" / "modalities" / "astigmatism" / "astigmatism_single_channel_smoke.yaml"


def _config() -> dict[str, object]:
    payload = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def test_astigmatism_smoke_config_resolves_explicit_single_channel_contract() -> None:
    config = _config()

    runtime_config = build_localization_runtime_config(config, config_base_dir=CONFIG_PATH.parent, seed=7)

    assert runtime_config["model"]["name"] == "astigmatism_expert"
    assert runtime_config["input_frame_spec"] == {
        "input_frame_channels": 3,
        "frame_order": "temporal",
    }
    assert runtime_config["channel_layout"] == {
        "channels": [{"channel_id": "main", "crop": None, "anchor_profile": None, "calibration_ref": None}],
    }
    assert runtime_config["expert_instance"] == {
        "expert_type": PSFModality.ASTIGMATISM.value,
        "instance_id": "main",
        "channel_id": "main",
        "prototype_ref": None,
    }

    provider_params = runtime_config["batch_provider"]["params"]
    assert provider_params["append_domain_onehot"] is False
    assert provider_params["domain_count"] == 1
    assert len(provider_params["dual_domain_coeff_maps"]) <= 1
    assert provider_params["condition_dim"] == 4
    assert provider_params["condition_fields"] == ("zernike_0", "zernike_1", "field_x", "field_y")
    assert runtime_config["loss"]["name"] == "active_smlm_gmm_loss"
    assert runtime_config["loss"]["params"]["target_order"] == "legacy_iwae"
    assert runtime_config["resolved_contract"]["input_frame_spec"] == runtime_config["input_frame_spec"]
    assert runtime_config["resolved_contract"]["channel_layout"] == runtime_config["channel_layout"]
    assert runtime_config["resolved_contract"]["expert_instance"] == runtime_config["expert_instance"]


def test_astigmatism_runtime_builds_one_model_and_trains_one_online_batch(tmp_path: Path) -> None:
    config = _config()
    runtime_config = build_localization_runtime_config(config, config_base_dir=CONFIG_PATH.parent, seed=9)
    runtime = build_trainer_runtime(
        runtime_config,
        layout=ensure_run_layout(tmp_path, "astigmatism_single_channel"),
        model_registry=build_localization_model_registry(),
    )

    assert isinstance(runtime.model, AstigmatismExpert)
    assert len(runtime.optimizer.param_groups) == 1
    batch = next(iter(runtime.batch_provider(1)))
    images, conditions = batch.inputs.model_input
    assert tuple(images.shape[1:]) == (3, 8, 8)
    assert tuple(conditions.shape) == (1, 4)
    assert batch.inputs.metadata["domain_count"] == 1
    assert batch.inputs.metadata["condition_domain_onehot_slice"] is None
    assert batch.inputs.metadata["condition_feature_order"] == (
        "zernike_0",
        "zernike_1",
        "field_x",
        "field_y",
    )

    before = {name: parameter.detach().clone() for name, parameter in runtime.model.named_parameters()}
    result = train_one_epoch(
        model=runtime.model,
        optimizer=runtime.optimizer,
        scheduler=runtime.scheduler,
        batches=[batch],
        layout=runtime.layout,
        config=TrainingConfig(epoch=1),
        loss_fn=runtime.loss_fn,
    )

    assert result.step_count == 1
    assert torch.isfinite(torch.tensor(result.mean_loss))
    assert any(not torch.equal(before[name], parameter.detach()) for name, parameter in runtime.model.named_parameters())


def test_legacy_soft_moe_config_still_resolves_to_legacy_route() -> None:
    config = {
        "train": {
            "online_generation": {
                "enabled": True,
                "channels": 3,
                "conditioning_mode": "film",
                "expert_mode": "soft_moe",
                "condition_feature_dim": 2,
                "condition_dim": 4,
                "domain_count": 2,
            }
        }
    }

    model_name, model_params = resolve_localization_model_config(config)

    assert model_name == "active_smlm_soft_moe_double_unet"
    assert model_params["condition_dim"] == 4
    assert model_params["domain_count"] == 2


def test_astigmatism_contract_normalizes_expert_alias_and_preserves_frame_size() -> None:
    config = _config()
    config["train"]["expert"]["name"] = "astigmatism_expert"
    config["train"]["channel_layout"] = {
        "frame_size": [8, 8],
        "channels": [{"id": "main", "crop": [0, 0, 8, 8]}],
    }

    runtime_config = build_localization_runtime_config(config, config_base_dir=CONFIG_PATH.parent)

    assert runtime_config["expert_instance"]["expert_type"] == "astigmatism"
    assert runtime_config["channel_layout"]["frame_size"] == [8, 8]


def test_astigmatism_runtime_rejects_model_provider_condition_dimension_mismatch() -> None:
    config = deepcopy(_config())
    config["train"]["online_generation"]["condition_feature_dim"] = 3

    with pytest.raises(ValueError, match="condition_feature_dim"):
        build_localization_runtime_config(config, config_base_dir=CONFIG_PATH.parent)


def test_explicit_train_optimizer_is_used_by_astigmatism_runtime(tmp_path: Path) -> None:
    config = _config()

    runtime_config = build_localization_runtime_config(config, config_base_dir=CONFIG_PATH.parent)
    runtime = build_trainer_runtime(
        runtime_config,
        layout=ensure_run_layout(tmp_path, "explicit_adamw"),
        model_registry=build_localization_model_registry(),
    )

    assert runtime_config["optimizer"] == {"name": "adamw", "params": {"lr": 0.001}}
    assert isinstance(runtime.optimizer, torch.optim.AdamW)


def test_astigmatism_rejects_active_loss_for_legacy_iwae_targets() -> None:
    config = _config()
    config["train"]["loss"] = {"name": "active_smlm_loss", "params": {}}

    with pytest.raises(ValueError, match="legacy_iwae"):
        build_localization_runtime_config(config, config_base_dir=CONFIG_PATH.parent)
