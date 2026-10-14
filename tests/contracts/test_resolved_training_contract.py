from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from unity_psf.contracts.training import ResolvedTrainingContract, ResolvedTrainingInstance
from unity_psf.training.entrypoints.canonical_runtime import (
    _load_joint_config,
    _load_resolved_training_contract,
)


def _contract(**changes):
    values = {
        "run_id": "demo",
        "run_dir": "/tmp/demo",
        "epochs": 3,
        "required_gpus": 2,
        "smoke": False,
        "instances": (
            ResolvedTrainingInstance("emitter_2d:left", "emitter_2d", "left", "/tmp/left.yaml", 1, 2, 4),
            ResolvedTrainingInstance("astigmatism:right", "astigmatism", "right", "/tmp/right.yaml", 3, 4, 4),
        ),
        "configs": {"emitter_2d:left": {"train": {"epochs": 3}}, "astigmatism:right": {"train": {"epochs": 3}}},
        "inputs": {"emitter_2d:left:coefficient_map": {"path": "/tmp/map.npz", "sha256": "a" * 64}},
    }
    values.update(changes)
    return ResolvedTrainingContract(**values)


def test_contract_hash_is_stable_and_round_trips() -> None:
    contract = _contract()
    restored = ResolvedTrainingContract.from_dict(contract.to_dict())
    assert restored.contract_hash == contract.contract_hash
    assert restored.modalities == ("emitter_2d", "astigmatism")


def test_contract_hash_changes_for_semantic_training_changes() -> None:
    original = _contract()
    changed = _contract(epochs=4)
    assert original.contract_hash != changed.contract_hash


def test_contract_hash_is_independent_of_route_argument_order() -> None:
    original = _contract()
    reversed_order = _contract(instances=tuple(reversed(original.instances)))
    assert reversed_order.contract_hash == original.contract_hash
    assert [item.key for item in reversed_order.instances] == [
        "emitter_2d:left",
        "astigmatism:right",
    ]


def test_joint_config_carries_contract_identity() -> None:
    contract = _contract()
    joint = contract.joint_config(__import__("pathlib").Path("/tmp/demo/resolved"))
    assert joint["contract_schema"] == "unitypsf.resolved_training_contract.v1"
    assert joint["contract_hash"] == contract.contract_hash
    assert joint["instances"][0]["key"] == "emitter_2d:left"


def test_executor_rejects_resolved_yaml_that_differs_from_contract(tmp_path: Path) -> None:
    resolved = tmp_path / "run" / "resolved"
    resolved.mkdir(parents=True)
    config_path = resolved / "emitter.yaml"
    config = {"train": {"epochs": 1, "batch_size": 1}}
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    contract = ResolvedTrainingContract(
        run_id="run",
        run_dir=str(tmp_path / "run"),
        epochs=1,
        required_gpus=1,
        smoke=True,
        instances=(ResolvedTrainingInstance("emitter_2d:left", "emitter_2d", "left", str(config_path), 1, 2, 1),),
        configs={"emitter_2d:left": config},
        inputs={},
    )
    (resolved / "training_contract.json").write_text(
        json.dumps(contract.to_dict()), encoding="utf-8"
    )
    joint_path = resolved / "joint.yaml"
    joint_path.write_text(yaml.safe_dump(contract.joint_config(resolved)), encoding="utf-8")
    joint = _load_joint_config(joint_path)
    assert _load_resolved_training_contract(joint_path, joint).contract_hash == contract.contract_hash
    config_path.write_text(yaml.safe_dump({"train": {"epochs": 2, "batch_size": 1}}), encoding="utf-8")
    with pytest.raises(ValueError, match="differs from the training contract"):
        _load_resolved_training_contract(joint_path, joint)
