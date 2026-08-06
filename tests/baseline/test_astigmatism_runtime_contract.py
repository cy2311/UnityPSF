from __future__ import annotations

from pathlib import Path

import torch
import yaml

from unity_psf.localization.model import build_localization_model_registry
from unity_psf.localization.runtime_config import resolve_localization_model_config
from unity_psf.localization.smlm_output import SMLMOutputChannels, decode_smlm_output
from unity_psf.runtime.layout import ensure_run_layout
from unity_psf.training.loop import TrainingBatch, TrainingConfig, load_training_checkpoint, train_one_epoch


FIXTURE_PATH = Path(__file__).parents[1] / "fixtures" / "astigmatism_baseline.yaml"


def _fixture() -> dict[str, object]:
    payload = yaml.safe_load(FIXTURE_PATH.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _build_current_soft_moe(fixture: dict[str, object]) -> torch.nn.Module:
    model_spec = fixture["model"]
    assert isinstance(model_spec, dict)
    params = model_spec["params"]
    assert isinstance(params, dict)
    registry = build_localization_model_registry()
    model = registry[str(model_spec["name"])](dict(params))
    return model


def test_baseline_config_resolves_to_current_soft_moe_route() -> None:
    fixture = _fixture()
    model_spec = fixture["model"]
    assert isinstance(model_spec, dict)
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
    assert model_name == model_spec["name"]
    assert model_params["nch_in"] == 3
    assert model_params["condition_dim"] == 4
    assert model_params["domain_count"] == 2


def test_current_soft_moe_supports_single_and_dual_domain_10ch_output() -> None:
    fixture = _fixture()
    model = _build_current_soft_moe(fixture)
    model.eval()

    input_spec = fixture["input"]
    conditioning_spec = fixture["conditioning"]
    output_spec = fixture["output"]
    assert isinstance(input_spec, dict)
    assert isinstance(conditioning_spec, dict)
    assert isinstance(output_spec, dict)

    torch.manual_seed(17)
    images = torch.randn(
        int(input_spec["batch_size"]),
        int(input_spec["channels"]),
        int(input_spec["height"]),
        int(input_spec["width"]),
    )
    conditions = torch.tensor(conditioning_spec["vectors"], dtype=torch.float32)

    with torch.no_grad():
        dual_output = model((images, conditions))
        single_output = model((images[:1], conditions[:1]))
        repeated_output = model((images, conditions))

    assert getattr(model, "domain_count") == int(conditioning_spec["domain_count"])
    assert len(getattr(model, "experts")) == int(conditioning_spec["domain_count"])
    assert tuple(dual_output.shape) == (
        int(input_spec["batch_size"]),
        int(output_spec["channels"]),
        int(input_spec["height"]),
        int(input_spec["width"]),
    )
    assert tuple(single_output.shape) == (
        1,
        int(output_spec["channels"]),
        int(input_spec["height"]),
        int(input_spec["width"]),
    )
    assert torch.allclose(dual_output, repeated_output)

    decoded = decode_smlm_output(dual_output)
    assert decoded.raw.shape[1] == SMLMOutputChannels.count == int(output_spec["channels"])
    assert decoded.p.shape == (int(input_spec["batch_size"]), int(input_spec["height"]), int(input_spec["width"]))
    assert decoded.pxyz_mu.shape == (
        int(input_spec["batch_size"]),
        4,
        int(input_spec["height"]),
        int(input_spec["width"]),
    )
    assert decoded.pxyz_sigma.shape == decoded.pxyz_mu.shape
    assert decoded.bg.shape == decoded.p.shape


def test_current_training_checkpoint_contains_resume_state(tmp_path: Path) -> None:
    fixture = _fixture()
    model = _build_current_soft_moe(fixture)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
    input_spec = fixture["input"]
    conditioning_spec = fixture["conditioning"]
    checkpoint_spec = fixture["checkpoint"]
    assert isinstance(input_spec, dict)
    assert isinstance(conditioning_spec, dict)
    assert isinstance(checkpoint_spec, dict)

    images = torch.zeros(
        int(input_spec["batch_size"]),
        int(input_spec["channels"]),
        int(input_spec["height"]),
        int(input_spec["width"]),
    )
    conditions = torch.tensor(conditioning_spec["vectors"], dtype=torch.float32)
    targets = torch.zeros(
        int(input_spec["batch_size"]),
        10,
        int(input_spec["height"]),
        int(input_spec["width"]),
    )
    layout = ensure_run_layout(tmp_path, "baseline_checkpoint")

    result = train_one_epoch(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        batches=[TrainingBatch(inputs=(images, conditions), targets=targets)],
        layout=layout,
        config=TrainingConfig(
            epoch=1,
            checkpoint_name="checkpoint_latest.pt",
            scheduler_step_unit="optimizer_step",
        ),
    )

    payload = torch.load(result.checkpoint_path, map_location="cpu")
    assert set(checkpoint_spec["required_keys"]).issubset(payload)
    assert payload["epoch"] == 1
    assert payload["step_count"] == 1
    assert payload["global_step"] == 1

    restored = _build_current_soft_moe(fixture)
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=1e-3)
    restored_scheduler = torch.optim.lr_scheduler.StepLR(restored_optimizer, step_size=1, gamma=0.9)
    resume = load_training_checkpoint(
        result.checkpoint_path,
        model=restored,
        optimizer=restored_optimizer,
        scheduler=restored_scheduler,
        map_location="cpu",
    )

    assert resume.epoch == 1
    assert resume.step_count == 1
    assert resume.global_step == 1
    for name, value in model.state_dict().items():
        assert torch.equal(value, restored.state_dict()[name]), name
