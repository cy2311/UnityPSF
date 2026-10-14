"""Resolved training contract shared by planning, execution, and checkpoints."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping


RESOLVED_TRAINING_CONTRACT_SCHEMA = "unitypsf.resolved_training_contract.v1"
_MODALITY_ORDER = {"emitter_2d": 0, "astigmatism": 1, "double_helix": 2}


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_freeze(item) for item in value)
    return value


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


@dataclass(frozen=True)
class ResolvedTrainingInstance:
    """One channel-bound training instance in a resolved execution plan."""

    key: str
    modality: str
    channel: str
    config: str
    model_seed: int
    data_seed: int
    step_budget: int

    def __post_init__(self) -> None:
        if not self.key or self.key != f"{self.modality}:{self.channel}":
            raise ValueError("training instance key must match modality and channel")
        if self.model_seed < 0 or self.data_seed < 0 or self.step_budget <= 0:
            raise ValueError("training instance seeds must be non-negative and budget positive")


@dataclass(frozen=True)
class ResolvedTrainingContract:
    """The single resolved semantic contract for one UnityPSF training run."""

    run_id: str
    run_dir: str
    epochs: int
    required_gpus: int
    smoke: bool
    instances: tuple[ResolvedTrainingInstance, ...]
    configs: Mapping[str, Mapping[str, Any]]
    inputs: Mapping[str, Mapping[str, str]]
    execution: str = "expert_parallel"
    rank_assignment: str = "one_modality_per_rank"
    schema_version: str = RESOLVED_TRAINING_CONTRACT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != RESOLVED_TRAINING_CONTRACT_SCHEMA:
            raise ValueError(f"unsupported resolved training contract {self.schema_version!r}")
        if not self.run_id or self.run_id in {".", ".."} or Path(self.run_id).name != self.run_id:
            raise ValueError("run_id must be one safe directory name")
        if self.epochs <= 0 or self.required_gpus <= 0:
            raise ValueError("epochs and required_gpus must be positive")
        if self.execution != "expert_parallel" or self.rank_assignment != "one_modality_per_rank":
            raise ValueError("unsupported canonical training execution contract")
        if not self.instances:
            raise ValueError("resolved training contract requires at least one instance")
        keys = tuple(item.key for item in self.instances)
        if len(set(keys)) != len(keys):
            raise ValueError("resolved training instances must be unique")
        ordered_instances = tuple(
            sorted(
                self.instances,
                key=lambda item: (_MODALITY_ORDER.get(item.modality, 99), item.modality, item.channel),
            )
        )
        object.__setattr__(self, "instances", ordered_instances)
        if set(keys) != set(self.configs):
            raise ValueError("resolved configs must cover every training instance")
        modalities = {item.modality for item in self.instances}
        if self.required_gpus != len(modalities):
            raise ValueError("required_gpus must equal the selected modality count")
        object.__setattr__(self, "configs", _freeze(self.configs))
        object.__setattr__(self, "inputs", _freeze(self.inputs))

    @property
    def modalities(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(item.modality for item in self.instances))

    @property
    def contract_hash(self) -> str:
        payload = self.to_dict(include_hash=False)
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
        return hashlib.sha256(encoded).hexdigest()

    def to_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        value = {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "run_dir": self.run_dir,
            "epochs": self.epochs,
            "required_gpus": self.required_gpus,
            "smoke": self.smoke,
            "execution": self.execution,
            "rank_assignment": self.rank_assignment,
            "instances": [asdict(item) for item in self.instances],
            "configs": _plain(self.configs),
            "inputs": _plain(self.inputs),
        }
        if include_hash:
            value["contract_hash"] = self.contract_hash
        return value

    def joint_config(self, resolved_dir: Path) -> dict[str, Any]:
        """Return the only joint execution document accepted by the trainer."""

        return {
            "schema_version": "unitypsf.joint_training.v1",
            "execution": self.execution,
            "rank_assignment": self.rank_assignment,
            "epochs": self.epochs,
            "required_gpus": self.required_gpus,
            "contract_schema": self.schema_version,
            "contract_hash": self.contract_hash,
            "instances": [
                {
                    "key": item.key,
                    "config": item.config,
                    "model_seed": item.model_seed,
                    "data_seed": item.data_seed,
                    "step_budget": item.step_budget,
                }
                for item in self.instances
            ],
            "metadata": {
                "selection_cli": "unity-psf-train",
                "contract": str(resolved_dir / "training_contract.json"),
                "input_manifest": str(resolved_dir / "inputs.json"),
            },
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ResolvedTrainingContract":
        if not isinstance(value, Mapping):
            raise TypeError("resolved training contract must be a mapping")
        instances = tuple(
            ResolvedTrainingInstance(
                key=str(item["key"]), modality=str(item["modality"]), channel=str(item["channel"]),
                config=str(item["config"]), model_seed=int(item["model_seed"]),
                data_seed=int(item["data_seed"]), step_budget=int(item["step_budget"]),
            )
            for item in value.get("instances", ())
        )
        contract = cls(
            schema_version=str(value.get("schema_version", "")),
            run_id=str(value["run_id"]), run_dir=str(value["run_dir"]),
            epochs=int(value["epochs"]), required_gpus=int(value["required_gpus"]),
            smoke=bool(value["smoke"]), instances=instances,
            configs=value.get("configs", {}), inputs=value.get("inputs", {}),
            execution=str(value.get("execution", "expert_parallel")),
            rank_assignment=str(value.get("rank_assignment", "one_modality_per_rank")),
        )
        saved_hash = value.get("contract_hash")
        if saved_hash is not None and str(saved_hash) != contract.contract_hash:
            raise ValueError("resolved training contract hash does not match its contents")
        return contract


__all__ = [
    "RESOLVED_TRAINING_CONTRACT_SCHEMA",
    "ResolvedTrainingContract",
    "ResolvedTrainingInstance",
]
