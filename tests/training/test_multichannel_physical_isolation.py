from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from unity_psf.localization.runtime_config import build_localization_runtime_config
from unity_psf.runtime.layout import ensure_run_layout
from unity_psf.training.channel_context import ChannelTrainingContext, sha256_file
from unity_psf.training.run_high_fidelity import (
    _physical_checkpoint_extra_fn,
    _select_single_channel_roi_split,
    _single_channel_peak_domain,
)


def _channel_runtime_config(channel_id: str, *, crop: tuple[int, int, int, int]) -> dict[str, object]:
    return {
        "expert_instance": {
            "expert_type": "astigmatism",
            "instance_id": channel_id,
            "channel_id": channel_id,
            "prototype_ref": None,
        },
        "input_frame_spec": {"input_frame_channels": 3, "frame_order": "temporal"},
        "channel_layout": {
            "frame_size": [32, 64],
            "channels": [{"id": channel_id, "crop": list(crop)}],
        },
    }


def _coeff_map(path: Path, value: float) -> Path:
    np.savez(
        path,
        zernike_maps_nm=np.full((6, 8, 8), value, dtype=np.float32),
        mode_order=np.asarray([[2, 0], [3, 1], [3, -1], [4, 0], [3, -3], [3, 3]], dtype=np.int64),
    )
    return path


def test_left_and_right_contexts_have_independent_state_and_checkpoint_extra(tmp_path: Path) -> None:
    left_layout = ensure_run_layout(tmp_path, "run/channels/left")
    right_layout = ensure_run_layout(tmp_path, "run/channels/right")
    left_context = ChannelTrainingContext.from_runtime_config(
        _channel_runtime_config("left", crop=(0, 0, 8, 8)),
        layout=left_layout,
    )
    right_context = ChannelTrainingContext.from_runtime_config(
        _channel_runtime_config("right", crop=(8, 0, 8, 8)),
        layout=right_layout,
    )

    left_map = _coeff_map(tmp_path / "left.npz", 1.0)
    right_map = _coeff_map(tmp_path / "right.npz", 2.0)
    left_context.update_coefficient_map(left_map)
    right_context.update_coefficient_map(right_map)
    left_context.write_physical_state(source="initial")
    right_context.write_physical_state(source="initial")

    assert left_context.physical_state_path.name == "current_physical_state.json"
    assert right_context.physical_state_path.name == "current_physical_state.json"
    assert left_context.physical_state_path != right_context.physical_state_path
    assert left_context.latest_physical_state_hash != right_context.latest_physical_state_hash

    left_extra = _physical_checkpoint_extra_fn(left_layout, physical_context=left_context)()
    right_extra = _physical_checkpoint_extra_fn(right_layout, physical_context=right_context)()
    assert left_extra["physical_state"]["expert_instance"]["channel_id"] == "left"
    assert right_extra["physical_state"]["expert_instance"]["channel_id"] == "right"
    assert left_extra["physical_coeff_maps"][0]["coeff_maps_npz"] == str(left_map)
    assert right_extra["physical_coeff_maps"][0]["coeff_maps_npz"] == str(right_map)
    assert left_extra["physical_coeff_maps"][0]["sha256"] == sha256_file(left_map)

    left_context.physical_state_path.write_bytes(right_context.physical_state_path.read_bytes())
    with pytest.raises(ValueError, match="physical state channel_id does not match"):
        _physical_checkpoint_extra_fn(left_layout, physical_context=left_context)()


def test_single_channel_peak_domain_rejects_ambiguous_fallback() -> None:
    train_cfg = {
        "model": {"name": "astigmatism_expert"},
        "expert": {"name": "astigmatism", "channel_id": "right"},
        "channel_layout": {"channels": [{"id": "right"}]},
    }

    with pytest.raises(ValueError, match="no matching domain"):
        _single_channel_peak_domain(
            train_cfg,
            [
                {"name": "left", "crop_left": 0, "crop_top": 0, "crop_width": 32, "crop_height": 32},
                {"name": "main", "crop_left": 32, "crop_top": 0, "crop_width": 32, "crop_height": 32},
            ],
        )


def test_single_channel_runtime_selects_its_coefficient_map(tmp_path: Path) -> None:
    config_path = Path(__file__).parents[2] / "configs" / "modalities" / "astigmatism" / "astigmatism_single_channel_smoke.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert isinstance(config, dict)
    left = _coeff_map(tmp_path / "left.npz", 1.0)
    right = _coeff_map(tmp_path / "right.npz", 2.0)
    config["train"]["channel_layout"] = {
        "frame_size": [8, 16],
        "channels": [{"id": "right", "crop": [8, 0, 8, 8]}],
    }
    config["train"]["expert"].update({"instance_id": "right", "channel_id": "right"})
    config["train"]["online_generation"]["dual_domain_coeff_maps"] = [
        {"name": "left", "coeff_maps_npz": str(left)},
        {"name": "right", "coeff_maps_npz": str(right)},
    ]

    runtime_config = build_localization_runtime_config(config, config_base_dir=tmp_path, seed=3)

    entries = runtime_config["batch_provider"]["params"]["dual_domain_coeff_maps"]
    assert entries == ({"name": "right", "coeff_maps_npz": str(right.resolve())},)
    assert runtime_config["batch_provider"]["params"]["domain_count"] == 1


@pytest.mark.parametrize(
    ("expert_type", "expected_condition_dim", "expected_domain_onehot"),
    [("emitter_2d", 2, False), ("double_helix", 5, True)],
)
def test_non_astigmatism_single_channel_runtime_uses_one_domain(
    tmp_path: Path,
    expert_type: str,
    expected_condition_dim: int,
    expected_domain_onehot: bool,
) -> None:
    left = _coeff_map(tmp_path / "left.npz", 1.0)
    right = _coeff_map(tmp_path / "right.npz", 2.0)
    config = {
        "train": {
            "channel_layout": {
                "frame_size": [8, 16],
                "channels": [{"id": "right", "crop": [8, 0, 8, 8]}],
            },
            "expert": {
                "expert_type": expert_type,
                "instance_id": "right",
                "channel_id": "right",
            },
            "online_generation": {
                "enabled": True,
                "channels": 3,
                "conditioning_mode": "film",
                "expert_mode": "soft_moe",
                "condition_feature_dim": 4,
                "condition_dim": 6,
                "domain_count": 2,
                "append_domain_onehot": True,
                "dual_domain_coeff_maps": [
                    {"name": "left", "coeff_maps_npz": str(left)},
                    {"name": "right", "coeff_maps_npz": str(right)},
                ],
            },
        }
    }

    runtime_config = build_localization_runtime_config(config, config_base_dir=tmp_path, seed=3)

    provider = runtime_config["batch_provider"]["params"]
    assert provider["dual_domain_coeff_maps"] == ({"name": "right", "coeff_maps_npz": str(right.resolve())},)
    assert provider["domain_count"] == 1
    assert provider["condition_dim"] == expected_condition_dim
    assert provider["append_domain_onehot"] is expected_domain_onehot
    if expert_type == "emitter_2d":
        assert provider["condition_fields"] == ("field_x", "field_y")
        assert runtime_config["model"]["params"]["condition_dim"] == 2
    else:
        assert runtime_config["model"]["params"]["domain_count"] == 1
        assert runtime_config["model"]["params"]["condition_dim"] == 5
    assert runtime_config["expert_instance"] == {
        "expert_type": expert_type,
        "instance_id": "right",
        "channel_id": "right",
        "prototype_ref": None,
    }


def test_astigmatism_channel_runtime_overrides_legacy_soft_moe_dimensions() -> None:
    config = {
        "train": {
            "channel_layout": {
                "frame_size": [8, 16],
                "channels": [{"id": "left", "crop": [0, 0, 8, 8]}],
            },
            "expert": {"expert_type": "astigmatism", "instance_id": "left", "channel_id": "left"},
            "online_generation": {
                "enabled": True,
                "channels": 3,
                "conditioning_mode": "film",
                "expert_mode": "soft_moe",
                "condition_feature_dim": 8,
                "condition_dim": 10,
                "domain_count": 2,
                "append_domain_onehot": True,
            },
        }
    }

    runtime_config = build_localization_runtime_config(config, seed=3)

    provider = runtime_config["batch_provider"]["params"]
    assert runtime_config["model"]["name"] == "astigmatism_expert"
    assert provider["domain_count"] == 1
    assert provider["condition_feature_dim"] == 4
    assert provider["condition_dim"] == 4
    assert provider["append_domain_onehot"] is False


def test_single_channel_gamma_binds_a_unique_domain_to_current_channel() -> None:
    left_bank = object()

    selected = _select_single_channel_roi_split({"left": (left_bank, None)}, "right")

    assert tuple(selected) == ("right",)
    assert selected["right"] == (left_bank, None)


def test_peak_zmap_hash_is_channel_bound_and_verified_on_restore(tmp_path: Path) -> None:
    left_layout = ensure_run_layout(tmp_path, "run/channels/left")
    right_layout = ensure_run_layout(tmp_path, "run/channels/right")
    left_context = ChannelTrainingContext.from_runtime_config(
        _channel_runtime_config("left", crop=(0, 0, 8, 8)),
        layout=left_layout,
    )
    right_context = ChannelTrainingContext.from_runtime_config(
        _channel_runtime_config("right", crop=(8, 0, 8, 8)),
        layout=right_layout,
    )
    left_zmap = tmp_path / "left_zmap.npz"
    right_zmap = tmp_path / "right_zmap.npz"
    np.savez(left_zmap, zmap_nm=np.zeros((8, 8), dtype=np.float32))
    np.savez(right_zmap, zmap_nm=np.ones((8, 8), dtype=np.float32))
    left_context.peak_zmap_path = left_zmap
    right_context.peak_zmap_path = right_zmap

    left_context.write_physical_state(source="initial")
    right_context.write_physical_state(source="initial")
    left_state = json_load(left_context.physical_state_path)
    right_state = json_load(right_context.physical_state_path)

    assert left_state["peak_zmap_sha256"] == sha256_file(left_zmap)
    assert right_state["peak_zmap_sha256"] == sha256_file(right_zmap)
    assert left_state["peak_zmap_sha256"] != right_state["peak_zmap_sha256"]

    left_zmap.write_bytes(right_zmap.read_bytes())
    with pytest.raises(ValueError, match="peak zmap artifact hash mismatch"):
        ChannelTrainingContext.from_runtime_config(
            {
                **_channel_runtime_config("left", crop=(0, 0, 8, 8)),
                "metadata": {"peak_zmap_bootstrap": {"domains": {"left": {"zmap_path": str(left_zmap)}}}},
            },
            layout=left_layout,
        ).restore_physical_state(left_state)


def test_physical_state_identity_rejects_a_different_expert_type(tmp_path: Path) -> None:
    layout = ensure_run_layout(tmp_path, "run/channels/left")
    context = ChannelTrainingContext.from_runtime_config(
        _channel_runtime_config("left", crop=(0, 0, 8, 8)),
        layout=layout,
    )
    context.write_physical_state(source="initial")
    state = json_load(context.physical_state_path)
    state["expert_instance"]["expert_type"] = "double_helix"

    with pytest.raises(ValueError, match="physical state expert_type does not match"):
        context.validate_physical_state_identity(state)


def test_restore_rejects_a_coefficient_map_bound_to_another_channel(tmp_path: Path) -> None:
    layout = ensure_run_layout(tmp_path, "run/channels/left")
    coeff = _coeff_map(tmp_path / "right.npz", 2.0)
    context = ChannelTrainingContext.from_runtime_config(
        _channel_runtime_config("left", crop=(0, 0, 8, 8)),
        layout=layout,
    )

    state = {
        "schema_version": "unitypsf.channel_physical_state.v1",
        "expert_instance": {"expert_type": "astigmatism", "instance_id": "left", "channel_id": "left"},
        "coeff_maps": [{"name": "right", "coeff_maps_npz": str(coeff), "sha256": sha256_file(coeff)}],
    }

    with pytest.raises(ValueError, match="coefficient map channel_id does not match"):
        context.restore_physical_state(state)


def test_checkpoint_extra_rejects_multiple_channel_maps(tmp_path: Path) -> None:
    layout = ensure_run_layout(tmp_path, "run/channels/left")
    context = ChannelTrainingContext.from_runtime_config(
        _channel_runtime_config("left", crop=(0, 0, 8, 8)),
        layout=layout,
    )
    left = _coeff_map(tmp_path / "left.npz", 1.0)
    right = _coeff_map(tmp_path / "right.npz", 2.0)
    context.update_coefficient_map(left)
    context.write_physical_state(source="initial")
    state = json_load(context.physical_state_path)
    state["coeff_maps"].append({"name": "right", "coeff_maps_npz": str(right), "sha256": sha256_file(right)})
    context.physical_state_path.write_text(json.dumps(state), encoding="utf-8")

    with pytest.raises(ValueError, match="exactly one coefficient map"):
        _physical_checkpoint_extra_fn(layout, physical_context=context)()


def test_restore_rejects_a_new_state_with_missing_peak_zmap_hash(tmp_path: Path) -> None:
    layout = ensure_run_layout(tmp_path, "run/channels/left")
    context = ChannelTrainingContext.from_runtime_config(
        _channel_runtime_config("left", crop=(0, 0, 8, 8)),
        layout=layout,
    )
    zmap = tmp_path / "left-zmap.npz"
    np.savez(zmap, zmap_nm=np.zeros((8, 8), dtype=np.float32))
    context.peak_zmap_path = zmap
    context.write_physical_state(source="initial")
    state = json_load(context.physical_state_path)
    state["peak_zmap_sha256"] = None

    with pytest.raises(ValueError, match="peak zmap hash is required"):
        context.restore_physical_state(state)


def json_load(path: Path) -> dict[str, object]:
    import json

    return json.loads(path.read_text(encoding="utf-8"))
