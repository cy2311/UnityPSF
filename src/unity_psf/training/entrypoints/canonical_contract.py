"""Resolved training contract and joint configuration helpers.

This module owns the file-level contract boundary used by the canonical
expert-parallel training entrypoint. Runtime construction stays in
``canonical_runtime`` so contract validation can be tested independently.
"""

from __future__ import annotations

from copy import deepcopy
from collections import defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from unity_psf.config import resolve_config_reference
from unity_psf.contracts import ChannelLayout, JointExpertKey, PSFModality, ResolvedTrainingContract


def mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def load_joint_config(
    path: Path,
    *,
    expected_execution: str | None = "expert_parallel",
) -> Mapping[str, Any]:
    import yaml

    config = mapping(yaml.safe_load(path.read_text(encoding="utf-8")), "joint config")
    if config.get("schema_version") != "unitypsf.joint_training.v1":
        raise ValueError("unsupported joint training schema")
    execution = str(config.get("execution", "expert_parallel"))
    if expected_execution is not None and execution != expected_execution:
        raise ValueError(f"this entrypoint requires execution: {expected_execution}")
    return config


def load_resolved_training_contract(
    joint_path: Path,
    joint_config: Mapping[str, Any],
) -> ResolvedTrainingContract:
    """Load and verify the resolved contract that owns ``joint_config``."""

    metadata = mapping(joint_config.get("metadata", {}), "joint metadata")
    contract_ref = metadata.get("contract")
    if contract_ref is None:
        raise ValueError("joint config must reference metadata.contract")
    contract_path = config_path(joint_path, contract_ref)
    if not contract_path.is_file():
        raise ValueError(f"resolved training contract is missing: {contract_path}")
    contract = ResolvedTrainingContract.from_dict(
        json.loads(contract_path.read_text(encoding="utf-8"))
    )
    run_dir = Path(contract.run_dir).expanduser().resolve()
    resolved_dir = (run_dir / "resolved").resolve()
    if contract_path.resolve().parent != resolved_dir or joint_path.resolve().parent != resolved_dir:
        raise ValueError(
            "resolved training contract and joint config must belong to the run resolved directory"
        )
    if int(joint_config.get("epochs", -1)) != contract.epochs:
        raise ValueError("joint config epochs do not match resolved training contract")
    if int(joint_config.get("required_gpus", contract.required_gpus)) != contract.required_gpus:
        raise ValueError("joint config required_gpus does not match resolved training contract")
    if str(joint_config.get("execution")) != contract.execution:
        raise ValueError("joint config execution does not match resolved training contract")
    if str(joint_config.get("rank_assignment")) != contract.rank_assignment:
        raise ValueError("joint config rank assignment does not match resolved training contract")
    if joint_config.get("contract_schema") != contract.schema_version:
        raise ValueError("joint config contract schema does not match resolved contract")
    if joint_config.get("contract_hash") != contract.contract_hash:
        raise ValueError("joint config contract hash does not match resolved contract")

    raw_instances = instance_specs(joint_config)
    expected = {item.key: item for item in contract.instances}
    if set(raw_instances) != set(expected):
        raise ValueError("joint config instances do not match resolved training contract")

    import yaml

    serialized_configs = contract.to_dict(include_hash=False)["configs"]
    for key, spec in raw_instances.items():
        item = expected[key]
        instance_config_path = config_path(joint_path, spec["config"])
        if (
            instance_config_path.resolve() != Path(item.config).resolve()
            or int(spec.get("model_seed", -1)) != item.model_seed
            or int(spec.get("data_seed", -1)) != item.data_seed
            or int(spec.get("step_budget", -1)) != item.step_budget
        ):
            raise ValueError(f"joint instance {key!r} does not match resolved training contract")
        if instance_config_path.resolve().parent != resolved_dir:
            raise ValueError(f"resolved instance config {key!r} escapes the run resolved directory")
        if not instance_config_path.is_file():
            raise ValueError(f"resolved instance config is missing: {instance_config_path}")
        resolved_config = yaml.safe_load(instance_config_path.read_text(encoding="utf-8"))
        if json.dumps(resolved_config, sort_keys=True, separators=(",", ":")) != json.dumps(
            serialized_configs[key], sort_keys=True, separators=(",", ":")
        ):
            raise ValueError(f"resolved instance config {key!r} differs from the training contract")

    for name, artifact in contract.inputs.items():
        if not isinstance(artifact, Mapping) or not artifact.get("path") or not artifact.get("sha256"):
            raise ValueError(f"resolved training input {name!r} is malformed")
        artifact_path = Path(str(artifact["path"])).expanduser().resolve()
        if not artifact_path.is_file() or sha256(artifact_path) != str(artifact["sha256"]):
            raise ValueError(f"resolved training input {name!r} does not match its contract hash")
    return contract


def instance_specs(config: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    raw = config.get("instances")
    if not isinstance(raw, list) or not raw:
        raise ValueError("joint config instances must be a non-empty list")
    specs: dict[str, Mapping[str, Any]] = {}
    for item in raw:
        spec = mapping(item, "instance")
        if "key" not in spec or "config" not in spec:
            raise ValueError("every joint instance requires key and config")
        key = JointExpertKey.parse(str(spec["key"])).storage_key
        if key in specs:
            raise ValueError(f"duplicate joint instance key {key!r}")
        if JointExpertKey.parse(key).modality.value not in {
            "emitter_2d", "astigmatism", "double_helix"
        }:
            raise ValueError(f"unsupported trainable modality in joint config: {key!r}")
        if key.startswith("double_helix:") and JointExpertKey.parse(key).channel_id != "main":
            raise ValueError("double_helix training supports only the main channel")
        specs[key] = spec
    return specs


def bind_instance(config: Mapping[str, Any], key: str, *, device: str | None) -> dict[str, Any]:
    bound = deepcopy(dict(config))
    train = dict(mapping(bound.get("train"), "train"))
    parsed = JointExpertKey.parse(key)
    layout = ChannelLayout.from_value(train.get("channel_layout", {"channels": ["main"]}))
    matching = [channel for channel in layout.channels if channel.channel_id == parsed.channel_id]
    if matching:
        channel = matching[0]
    elif len(layout.channels) == 1:
        template = layout.channels[0]
        channel = type(template)(
            channel_id=parsed.channel_id,
            crop=template.crop,
            anchor_profile=template.anchor_profile,
            calibration_ref=template.calibration_ref,
        )
    else:
        raise ValueError(f"config has no measurement channel {parsed.channel_id!r}")
    channel_value = {
        "id": channel.channel_id,
        "crop": channel.crop,
        "anchor_profile": channel.anchor_profile,
        "calibration_ref": channel.calibration_ref,
    }
    train["channel_layout"] = {
        "channels": [{name: value for name, value in channel_value.items() if value is not None}],
        **({"frame_size": list(layout.frame_size)} if layout.frame_size is not None else {}),
    }
    train["expert"] = {
        **dict(mapping(train.get("expert", {}), "train.expert")),
        "expert_type": parsed.modality.value,
        "instance_id": parsed.channel_id,
        "channel_id": parsed.channel_id,
    }
    if device is not None:
        train["device"] = device
    bound["train"] = train
    return bound


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def config_path(joint_path: Path, value: object) -> Path:
    return resolve_config_reference(str(value), source_path=joint_path)


def modality_groups(
    specs: Mapping[str, Mapping[str, Any]],
) -> dict[PSFModality, list[tuple[str, Mapping[str, Any]]]]:
    grouped: dict[PSFModality, list[tuple[str, Mapping[str, Any]]]] = defaultdict(list)
    for key, spec in specs.items():
        grouped[JointExpertKey.parse(key).modality].append((key, spec))
    return dict(grouped)


_mapping = mapping
_load_joint_config = load_joint_config
_load_resolved_training_contract = load_resolved_training_contract
_instance_specs = instance_specs
_bind_instance = bind_instance
_sha256 = sha256
_config_path = config_path
_modality_groups = modality_groups
