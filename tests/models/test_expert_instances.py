from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import torch

from unity_psf.contracts import (
    ExpertInstanceSpec,
    load_checkpoint,
    save_checkpoint,
    sha256_file,
)
from unity_psf.models.psf_moe.experts.astigmatism import AstigmatismExpert
from unity_psf.models.psf_moe.instances import (
    AstigmatismExpertInstance,
    build_instance_optimizer,
    create_expert_instance_from_prototype,
    parameter_state_hash,
)


def _build_expert(seed: int = 17) -> AstigmatismExpert:
    torch.manual_seed(seed)
    return AstigmatismExpert(
        nch_in=3,
        depth_shared=1,
        depth_union=1,
        nfeatures_init=4,
        nfeatures_inter=4,
        condition_dim=4,
        film_hidden_dim=4,
    )


def _write_prototype(path: Path, *, metadata=None) -> tuple[AstigmatismExpert, str]:
    expert = _build_expert()
    save_checkpoint(path, expert.state_dict(), metadata=metadata or expert.checkpoint_metadata())
    return expert, sha256_file(path)


def test_factory_creates_independent_channel_instances_with_lineage(tmp_path: Path) -> None:
    prototype_path = tmp_path / "astigmatism_base.ckpt"
    prototype, parent_hash = _write_prototype(prototype_path)
    left = create_expert_instance_from_prototype(
        prototype_path,
        ExpertInstanceSpec("astigmatism", "astig-left", "left"),
        device="cpu",
    )
    right = create_expert_instance_from_prototype(
        prototype_path,
        ExpertInstanceSpec("astigmatism", "astig-right", "right"),
        device="cpu",
    )

    assert isinstance(left, AstigmatismExpertInstance)
    assert left.expert_type.value == "astigmatism"
    assert left.instance_id == "astig-left"
    assert left.channel_id == "left"
    assert left.parent_checkpoint_hash == parent_hash
    assert left.metadata.checkpoint_role == "instance"
    assert left.metadata.parent_checkpoint_hash == parent_hash
    assert left.metadata.channel_spec.channel_id == "left"
    with pytest.raises(AttributeError):
        left.instance_id = "mutated"  # type: ignore[misc]

    assert parameter_state_hash(left) == parameter_state_hash(right) == parameter_state_hash(prototype)
    for name, left_parameter in left.named_parameters():
        right_parameter = dict(right.named_parameters())[name]
        assert left_parameter is not right_parameter
        assert left_parameter.data_ptr() != right_parameter.data_ptr()
    assert next(left.parameters()).device.type == "cpu"


def test_instance_optimizer_step_cannot_change_sibling_or_prototype(tmp_path: Path) -> None:
    prototype_path = tmp_path / "astigmatism_base.ckpt"
    prototype, _ = _write_prototype(prototype_path)
    left = create_expert_instance_from_prototype(
        prototype_path,
        ExpertInstanceSpec("astigmatism", "astig-left", "left"),
    )
    right = create_expert_instance_from_prototype(
        prototype_path,
        ExpertInstanceSpec("astigmatism", "astig-right", "right"),
    )
    left_optimizer = build_instance_optimizer(left, optimizer_params={"lr": 1e-3})
    right_optimizer = build_instance_optimizer(right, optimizer_params={"lr": 1e-3})
    assert {id(parameter) for group in left_optimizer.param_groups for parameter in group["params"]}.isdisjoint(
        {id(parameter) for group in right_optimizer.param_groups for parameter in group["params"]}
    )

    right_before = {name: value.detach().clone() for name, value in right.state_dict().items()}
    prototype_before = {
        name: value.detach().clone() for name, value in load_checkpoint(prototype_path)["model_state_dict"].items()
    }
    images = torch.randn(1, 3, 8, 8)
    conditions = torch.randn(1, 4)
    left_optimizer.zero_grad(set_to_none=True)
    left(images, conditions).square().mean().backward()
    left_optimizer.step()

    assert any(not torch.equal(value, right_before[name]) for name, value in left.state_dict().items())
    for name, value in right.state_dict().items():
        assert torch.equal(value, right_before[name]), name
    prototype_after = load_checkpoint(prototype_path)["model_state_dict"]
    for name, value in prototype_before.items():
        assert torch.equal(value, prototype_after[name]), name


def test_factory_rejects_wrong_expert_type_and_mismatched_model_config(tmp_path: Path) -> None:
    prototype = _build_expert()
    wrong_type_metadata = replace(
        prototype.checkpoint_metadata(),
        expert_type="emitter_2d",
        experts=("emitter_2d",),
    )
    wrong_type_path = tmp_path / "wrong-type.ckpt"
    save_checkpoint(wrong_type_path, prototype.state_dict(), metadata=wrong_type_metadata)
    with pytest.raises(ValueError, match="expert_type"):
        create_expert_instance_from_prototype(
            wrong_type_path,
            ExpertInstanceSpec("astigmatism", "astig-left", "left"),
        )

    mismatched_metadata = replace(
        prototype.checkpoint_metadata(),
        model_config={**prototype.model_config, "nfeatures_init": 8},
    )
    mismatched_path = tmp_path / "mismatched-config.ckpt"
    save_checkpoint(mismatched_path, prototype.state_dict(), metadata=mismatched_metadata)
    with pytest.raises(RuntimeError):
        create_expert_instance_from_prototype(
            mismatched_path,
            ExpertInstanceSpec("astigmatism", "astig-left", "left"),
        )
