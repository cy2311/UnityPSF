from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from unity_psf.localization.runtime import build_localization_runtime_config
from unity_psf.training.joint_config import bind_instance


ROOT = Path(__file__).parents[2]
LEGACY_CONFIG_PATH = ROOT / "configs/modalities/astigmatism/astigmatism_single_channel_smoke.yaml"
FORMAL_CONFIG_PATH = ROOT / "configs/modalities/emitter_2d/emitter_2d_dual_channel_300epoch.yaml"


def _load(path: Path) -> dict[str, object]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(config, dict)
    return config


def _runtime_snapshot(runtime: dict[str, object]) -> dict[str, object]:
    provider = runtime["batch_provider"]
    assert isinstance(provider, dict)
    params = provider["params"]
    assert isinstance(params, dict)
    return {
        "model": runtime["model"],
        "optimizer": runtime["optimizer"],
        "loss": runtime["loss"],
        "input_frame_spec": runtime["input_frame_spec"],
        "channel_layout": runtime["channel_layout"],
        "expert_instance": runtime["expert_instance"],
        "provider": {
            key: params[key]
            for key in (
                "batch_size",
                "channels",
                "height",
                "width",
                "seed",
                "steps_per_epoch",
                "simulation_backend",
                "simulation_output_device",
                "psf_type",
                "condition_feature_dim",
                "condition_dim",
                "condition_fields",
                "domain_count",
                "append_domain_onehot",
                "batch_strategy",
                "sequence_count",
                "pxyz_target_order",
                "photon_scale",
                "z_scale",
            )
        },
        "resolved_contract": _portable_contract(runtime["resolved_contract"]),
    }


def _portable_contract(value: object) -> object:
    if isinstance(value, dict):
        return {key: _portable_contract_value(key, item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_portable_contract(item) for item in value)
    if isinstance(value, list):
        return [_portable_contract(item) for item in value]
    return value


def _portable_contract_value(key: object, value: object) -> object:
    if key == "coeff_maps_npz" and isinstance(value, str):
        return Path(value).name
    return _portable_contract(value)


def test_legacy_v04_runtime_snapshot_is_stable() -> None:
    runtime = build_localization_runtime_config(
        _load(LEGACY_CONFIG_PATH),
        config_base_dir=LEGACY_CONFIG_PATH.parent,
        seed=31,
    )

    snapshot = _runtime_snapshot(runtime)

    assert snapshot["model"] == {
        "name": "astigmatism_expert",
        "params": {
            "depth_shared": 1,
            "depth_union": 1,
            "nfeatures_init": 4,
            "nfeatures_inter": 4,
            "condition_fields": ["zernike_0", "zernike_1", "field_x", "field_y"],
            "film_hidden_dim": 4,
            "nch_in": 3,
            "condition_dim": 4,
        },
    }
    assert snapshot["provider"] == {
        "batch_size": 1,
        "channels": 3,
        "height": 8,
        "width": 8,
        "seed": 31,
        "steps_per_epoch": 1,
        "simulation_backend": "native",
        "simulation_output_device": "cpu",
        "psf_type": "vector",
        "condition_feature_dim": 4,
        "condition_dim": 4,
        "condition_fields": ("zernike_0", "zernike_1", "field_x", "field_y"),
        "domain_count": 1,
        "append_domain_onehot": False,
        "batch_strategy": "triplet",
        "sequence_count": 64,
        "pxyz_target_order": "legacy_iwae",
        "photon_scale": 10.0,
        "z_scale": 1.0,
    }
    assert snapshot["loss"] == {
        "name": "active_smlm_gmm_loss",
        "params": {"photon_scale": 10.0, "z_scale": 1.0, "target_order": "legacy_iwae"},
    }
    assert snapshot["input_frame_spec"] == {"input_frame_channels": 3, "frame_order": "temporal"}
    assert snapshot["expert_instance"] == {
        "expert_type": "astigmatism",
        "instance_id": "main",
        "channel_id": "main",
        "prototype_ref": None,
    }
    assert snapshot["resolved_contract"] == {
        "model": {"name": "astigmatism_expert", "output": "smlm_10ch"},
        "batch_provider": {
            "name": "online_train_batch",
            "pxyz_target_order": "legacy_iwae",
            "batch_strategy": "triplet",
            "sequence_count": 64,
            "condition_dim": 4,
            "condition_feature_dim": 4,
            "condition_fields": ("zernike_0", "zernike_1", "field_x", "field_y"),
            "domain_count": 1,
            "append_domain_onehot": False,
        },
        "loss": {"name": "active_smlm_gmm_loss", "params": snapshot["loss"]["params"], "legacy_params": {}},
        "training_runtime": {
            "optimizer": snapshot["optimizer"],
            "legacy_optimizer": None,
            "scheduler": {"name": "none", "active": False, "step_unit": None, "params": {}, "inactive_reason": "not_configured"},
            "grad_clip": {"configured_norm": None, "active": False},
            "amp": {"configured": False, "dtype": None, "active": False, "inactive_reason": "not_configured"},
            "max_batches": 1,
        },
        "input_frame_spec": snapshot["input_frame_spec"],
        "channel_layout": snapshot["channel_layout"],
        "expert_instance": snapshot["expert_instance"],
    }


def test_formal_v1_bound_channel_runtime_snapshot_is_stable() -> None:
    config = bind_instance(_load(FORMAL_CONFIG_PATH), "emitter_2d:left", device="cpu")
    runtime = build_localization_runtime_config(config, config_base_dir=FORMAL_CONFIG_PATH.parent, seed=31)

    snapshot = _runtime_snapshot(runtime)

    assert snapshot["model"]["name"] == "emitter_2d_expert"
    assert snapshot["model"]["params"]["disabled_attr"] == [3]
    assert snapshot["provider"] == {
        "batch_size": 24,
        "channels": 3,
        "height": 96,
        "width": 96,
        "seed": 31,
        "steps_per_epoch": 417,
        "simulation_backend": "lut",
        "simulation_output_device": "renderer",
        "psf_type": "vector",
        "condition_feature_dim": 8,
        "condition_dim": 8,
        "condition_fields": (
            "field_x",
            "field_y",
            "zernike_nm_mean:n2_m0",
            "zernike_nm_mean:n3_m1",
            "zernike_nm_mean:n3_m-1",
            "zernike_nm_mean:n4_m0",
            "zernike_nm_mean:n3_m-3",
            "zernike_nm_mean:n3_m3",
        ),
        "domain_count": 1,
        "append_domain_onehot": False,
        "batch_strategy": "cached_window",
        "sequence_count": 64,
        "pxyz_target_order": "legacy_iwae",
        "photon_scale": 31000.0,
        "z_scale": 0.1,
    }
    assert snapshot["loss"] == {
        "name": "active_smlm_gmm_loss",
        "params": {
            "disable_attr": 3,
            "photon_scale": 31000.0,
            "z_scale": 0.1,
            "gmm_target_chunk": 4,
            "gmm_component_chunk": 64,
            "gmm_backend": "mixture_same_family",
            "target_order": "legacy_iwae",
        },
    }
    assert snapshot["channel_layout"]["channels"][0]["channel_id"] == "left"
    assert snapshot["expert_instance"] == {
        "expert_type": "emitter_2d",
        "instance_id": "left",
        "channel_id": "left",
        "prototype_ref": None,
    }
    assert snapshot["resolved_contract"] == {
        "model": {"name": "emitter_2d_expert", "output": "smlm_10ch"},
        "batch_provider": {
            "name": "online_train_batch",
            "pxyz_target_order": "legacy_iwae",
            "batch_strategy": "cached_window",
            "sequence_count": 64,
            "condition_dim": 8,
            "condition_feature_dim": 8,
            "condition_fields": snapshot["provider"]["condition_fields"],
            "domain_count": 1,
            "append_domain_onehot": False,
            "dual_domain_coeff_maps": ({"name": "left", "coeff_maps_npz": "preferred_full_roi_zernike_maps_nm.npz"},),
        },
        "loss": {"name": "active_smlm_gmm_loss", "params": snapshot["loss"]["params"], "legacy_params": {}},
        "training_runtime": {
            "optimizer": snapshot["optimizer"],
            "legacy_optimizer": None,
            "scheduler": {
                "name": "StepLR",
                "active": True,
                "step_unit": "epoch",
                "params": {"step_size": 10, "gamma": 0.9},
                "inactive_reason": None,
                "legacy_source": "smlm_overrides",
            },
            "grad_clip": {"configured_norm": 0.03, "active": True},
            "amp": {"configured": True, "dtype": "float16", "active": True, "inactive_reason": None},
            "max_batches": None,
        },
        "input_frame_spec": snapshot["input_frame_spec"],
        "channel_layout": snapshot["channel_layout"],
        "expert_instance": snapshot["expert_instance"],
    }


def test_legacy_input_frame_alias_is_retained_without_rewriting_the_source_config() -> None:
    config = _load(LEGACY_CONFIG_PATH)
    frame_spec = config["train"]["input_frame_spec"]
    assert isinstance(frame_spec, dict)
    config["train"]["input_frame_spec"] = {"channels": frame_spec["input_frame_channels"], "order": frame_spec["frame_order"]}

    with pytest.warns(DeprecationWarning, match="input_frame_channels"):
        runtime = build_localization_runtime_config(config, config_base_dir=LEGACY_CONFIG_PATH.parent, seed=5)

    assert config["train"]["input_frame_spec"] == {"channels": 3, "order": "temporal"}
    assert runtime["input_frame_spec"] == {"input_frame_channels": 3, "frame_order": "temporal"}


def test_conflicting_legacy_and_canonical_input_frame_fields_keep_value_error_contract() -> None:
    config = deepcopy(_load(LEGACY_CONFIG_PATH))
    config["train"]["input_frame_spec"] = {
        "input_frame_channels": 3,
        "channels": 2,
        "frame_order": "temporal",
    }

    with pytest.raises(ValueError, match="input_frame_channels and legacy channels"):
        build_localization_runtime_config(config, config_base_dir=LEGACY_CONFIG_PATH.parent)


def test_unknown_runtime_schema_is_rejected_without_rewriting_the_input() -> None:
    config = _load(LEGACY_CONFIG_PATH)
    config["schema_version"] = "unitypsf.instance_training.v2"

    with pytest.raises(ValueError, match="unsupported localization runtime schema"):
        build_localization_runtime_config(config, config_base_dir=LEGACY_CONFIG_PATH.parent)

    assert config["schema_version"] == "unitypsf.instance_training.v2"
