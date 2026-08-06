"""CLI for independent left/right training of one PSF expert family."""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import yaml

from unity_psf.config import load_config
from unity_psf.contracts.modality import ChannelLayout, PSFModality
from unity_psf.training.multichannel import (
    ChannelRunSpec,
    MultichannelExecutionResult,
    MultichannelTrainingPlan,
    build_multichannel_plan,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plan or run independent channel instances for one UnityPSF expert."
    )
    parser.add_argument("--config", type=Path, required=True, help="Base YAML training config.")
    parser.add_argument("--run-root", type=Path, required=True, help="Root directory for parent runs.")
    parser.add_argument("--expert-type", help="PSF modality; defaults to train.expert.name.")
    parser.add_argument("--run-name", help="Parent run name; defaults to the modality name.")
    parser.add_argument("--mode", choices=("plan", "local", "slurm"), default="plan")
    parser.add_argument("--seed", type=int, default=0, help="Default seed for every channel.")
    parser.add_argument(
        "--channel-seed",
        action="append",
        default=[],
        metavar="CHANNEL=SEED",
        help="Override one channel seed; repeatable.",
    )
    parser.add_argument(
        "--prototype",
        action="append",
        default=[],
        metavar="CHANNEL=PATH",
        help="Prototype checkpoint for one channel; repeatable.",
    )
    parser.add_argument("--entrypoint", help="Python module to launch for each channel.")
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--workdir", type=Path, default=Path("."))
    parser.add_argument("--slurm-script-root", type=Path, help="Directory for generated channel scripts.")
    parser.add_argument(
        "--slurm-resource",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="SLURM resource such as gpus=1; repeatable.",
    )
    parser.add_argument("--fail-fast", action="store_true", help="Stop local execution after the first failed channel.")
    parser.add_argument("--json-output", type=Path, help="Also write the result JSON to this path.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    modality = _resolve_modality(config, args.expert_type)
    layout = _resolve_channel_layout(config)
    channel_seeds = _parse_channel_assignments(args.channel_seed, value_name="seed", cast=int)
    prototypes = _parse_channel_assignments(args.prototype, value_name="prototype", cast=str)
    train_cfg = _mapping(config.get("train"))
    entrypoint = args.entrypoint or str(train_cfg.get("entrypoint", "unity_psf.training.run_localization"))
    plan = build_multichannel_plan(
        channel_layout=layout,
        expert_type=modality,
        run_root=args.run_root,
        run_name=args.run_name,
        seed=args.seed,
        channel_seeds=channel_seeds,
        prototype_ref=prototypes or None,
        entrypoint=entrypoint,
        config_path=args.config,
    )
    plan = _materialize_channel_configs(plan, config, args.config)

    if args.mode == "plan":
        payload = plan.to_dict()
    elif args.mode == "slurm":
        script_root = args.slurm_script_root or (plan.parent_run_dir / "slurm")
        jobs = plan.write_slurm_scripts(
            script_root,
            workdir=args.workdir,
            resources=_parse_resources(args.slurm_resource),
            python_executable=args.python_executable,
        )
        payload = {
            **plan.to_dict(),
            "jobs": [
                {
                    "channel_id": job.channel_id,
                    "script_path": str(job.script_path),
                    "entrypoint": job.launch_spec.entrypoint,
                    "args": [str(value) for value in job.launch_spec.args],
                }
                for job in jobs
            ],
        }
    else:
        result = plan.execute_local_subprocess(
            python_executable=args.python_executable,
            workdir=args.workdir,
            continue_on_error=not args.fail_fast,
        )
        payload = _execution_result_to_dict(result)

    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


def _resolve_modality(config: Mapping[str, Any], requested: str | None) -> PSFModality:
    if requested is not None:
        return PSFModality.parse(requested)
    expert = _mapping(_mapping(config.get("train")).get("expert"))
    selected = expert.get("expert_type", expert.get("modality", expert.get("name")))
    if selected is None:
        raise ValueError("--expert-type is required when train.expert has no modality name")
    return PSFModality.parse(str(selected))


def _resolve_channel_layout(config: Mapping[str, Any]) -> ChannelLayout:
    train_cfg = _mapping(config.get("train"))
    selected = train_cfg.get("channel_layout")
    if selected is None:
        measurement_cfg = _mapping(config.get("measurement"))
        selected = measurement_cfg.get("channel_layout", measurement_cfg.get("channels"))
    if selected is None:
        raise ValueError("config requires train.channel_layout for multichannel execution")
    return ChannelLayout.from_value(selected)


def _materialize_channel_configs(
    plan: MultichannelTrainingPlan,
    base_config: Mapping[str, Any],
    source_path: Path,
) -> MultichannelTrainingPlan:
    specs: list[ChannelRunSpec] = []
    base_dir = source_path.resolve().parent
    for spec in plan.channel_specs:
        channel_config = _channel_config(base_config, plan.expert_type, spec, base_dir=base_dir)
        config_path = spec.run_dir / "config.yaml"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(yaml.safe_dump(channel_config, sort_keys=False), encoding="utf-8")
        specs.append(replace(spec, config_path=config_path))
    return replace(plan, channel_specs=tuple(specs))


def _channel_config(
    config: Mapping[str, Any],
    modality: PSFModality,
    spec: ChannelRunSpec,
    *,
    base_dir: Path,
) -> dict[str, Any]:
    resolved = deepcopy(dict(config))
    train_cfg = dict(_mapping(resolved.get("train")))
    layout: dict[str, Any] = {"channels": [{"id": spec.channel_id}]}
    if spec.crop is not None:
        layout["channels"][0]["crop"] = list(spec.crop)
    if spec.anchor_profile is not None:
        layout["channels"][0]["anchor_profile"] = spec.anchor_profile
    if spec.calibration_ref is not None:
        layout["channels"][0]["calibration_ref"] = spec.calibration_ref
    original_layout = train_cfg.get("channel_layout")
    if not isinstance(original_layout, Mapping):
        measurement_cfg = _mapping(resolved.get("measurement"))
        original_layout = measurement_cfg.get("channel_layout")
    if isinstance(original_layout, Mapping) and original_layout.get("frame_size") is not None:
        layout["frame_size"] = deepcopy(original_layout["frame_size"])
    train_cfg["channel_layout"] = layout
    expert_cfg = dict(_mapping(train_cfg.get("expert")))
    expert_cfg.update(
        {
            "expert_type": modality.value,
            "instance_id": spec.instance_id,
            "channel_id": spec.channel_id,
        }
    )
    if spec.prototype_ref is not None:
        expert_cfg["prototype_ref"] = spec.prototype_ref
    train_cfg["expert"] = expert_cfg
    _filter_channel_physical_inputs(train_cfg, channel_id=spec.channel_id)
    resolved["train"] = train_cfg
    return _absolutize_paths(resolved, base_dir=base_dir)


def _filter_channel_physical_inputs(train_cfg: dict[str, Any], *, channel_id: str) -> None:
    """Keep raw and physical artifacts scoped to the materialized channel."""

    real_tiff_cfg = train_cfg.get("real_tiff_wake")
    if isinstance(real_tiff_cfg, Mapping) and "domains" in real_tiff_cfg:
        real_tiff_cfg = dict(real_tiff_cfg)
        real_tiff_cfg["domains"] = _select_channel_entries(
            real_tiff_cfg["domains"], channel_id=channel_id, field_name="train.real_tiff_wake.domains"
        )
        train_cfg["real_tiff_wake"] = real_tiff_cfg

    online_cfg = train_cfg.get("online_generation")
    if isinstance(online_cfg, Mapping):
        online_cfg = dict(online_cfg)
        if "dual_domain_coeff_maps" in online_cfg:
            online_cfg["dual_domain_coeff_maps"] = _select_channel_entries(
                online_cfg["dual_domain_coeff_maps"],
                channel_id=channel_id,
                field_name="train.online_generation.dual_domain_coeff_maps",
            )
        lut_cfg = online_cfg.get("lut_simulation")
        if isinstance(lut_cfg, Mapping) and "dual_domain_zmaps" in lut_cfg:
            lut_cfg = dict(lut_cfg)
            lut_cfg["dual_domain_zmaps"] = _select_channel_entries(
                lut_cfg["dual_domain_zmaps"],
                channel_id=channel_id,
                field_name="train.online_generation.lut_simulation.dual_domain_zmaps",
            )
            online_cfg["lut_simulation"] = lut_cfg
        train_cfg["online_generation"] = online_cfg

    gamma_cfg = train_cfg.get("roi_bank_gamma")
    if not isinstance(gamma_cfg, Mapping):
        return
    gamma_cfg = dict(gamma_cfg)
    if "base_coeff_maps" in gamma_cfg:
        gamma_cfg["base_coeff_maps"] = _select_channel_entries(
            gamma_cfg["base_coeff_maps"],
            channel_id=channel_id,
            field_name="train.roi_bank_gamma.base_coeff_maps",
        )
    if "auto_build_domains" in gamma_cfg:
        gamma_cfg["auto_build_domains"] = _select_channel_entries(
            gamma_cfg["auto_build_domains"],
            channel_id=channel_id,
            field_name="train.roi_bank_gamma.auto_build_domains",
        )
    source_cfg = gamma_cfg.get("roi_bank_source")
    if isinstance(source_cfg, Mapping) and "domains" in source_cfg:
        source_cfg = dict(source_cfg)
        source_cfg["domains"] = _select_channel_entries(
            source_cfg["domains"],
            channel_id=channel_id,
            field_name="train.roi_bank_gamma.roi_bank_source.domains",
        )
        gamma_cfg["roi_bank_source"] = source_cfg
    train_cfg["roi_bank_gamma"] = gamma_cfg


def _select_channel_entries(value: Any, *, channel_id: str, field_name: str) -> Any:
    if value is None:
        return None
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{field_name} must be a list")
    if not value:
        return list(value)

    matching = [item for item in value if _entry_channel_id(item) == channel_id]
    if len(value) > 1 and len(matching) != 1:
        names = [_entry_channel_id(item) or "<unnamed>" for item in value]
        raise ValueError(
            f"{field_name} has no unique entry for channel={channel_id!r}; available domains={names!r}"
        )
    selected = deepcopy(matching[0] if matching else value[0])
    return [_rename_channel_entry(selected, channel_id)]


def _entry_channel_id(value: Any) -> str | None:
    if isinstance(value, Mapping):
        selected = value.get("name", value.get("channel_id", value.get("id")))
        return None if selected is None else str(selected)
    if isinstance(value, (list, tuple)) and value:
        return str(value[0])
    return None


def _rename_channel_entry(value: Any, channel_id: str) -> Any:
    if isinstance(value, Mapping):
        renamed = dict(value)
        if "name" in renamed or "channel_id" not in renamed:
            renamed["name"] = channel_id
        else:
            renamed["channel_id"] = channel_id
        return renamed
    if isinstance(value, (list, tuple)) and value:
        renamed = list(value)
        renamed[0] = channel_id
        return renamed
    return value


_PATH_KEYS = frozenset(
    {
        "alternating_coeff_maps_npz",
        "auto_build_source_path",
        "checkpoint_path",
        "coeff_maps_npz",
        "path",
        "phase_figures_dir",
        "phase_output_dir",
        "pupil_carrier_complex_npz",
        "prototype_ref",
        "raw_path",
        "raw_tiff_dir",
        "roi_bank_path",
        "source_path",
        "source_reference",
        "tiff_path",
        "zmap_npz",
    }
)


def _absolutize_paths(value: Any, *, base_dir: Path, key: str | None = None) -> Any:
    if isinstance(value, Mapping):
        return {
            item_key: _absolutize_paths(item_value, base_dir=base_dir, key=str(item_key))
            for item_key, item_value in value.items()
        }
    if isinstance(value, list):
        return [_absolutize_paths(item, base_dir=base_dir, key=key) for item in value]
    if isinstance(value, tuple):
        return tuple(_absolutize_paths(item, base_dir=base_dir, key=key) for item in value)
    if key not in _PATH_KEYS or not isinstance(value, str) or not value.strip():
        return value
    path = Path(value)
    return str(path if path.is_absolute() else (base_dir / path).resolve())


def _execution_result_to_dict(result: MultichannelExecutionResult) -> dict[str, Any]:
    return {
        "parent_run_dir": str(result.parent_run_dir),
        "status": result.status.value,
        "channels": {channel_id: item.to_dict() for channel_id, item in result.channels.items()},
    }


def _parse_channel_assignments(entries: list[str], *, value_name: str, cast):
    parsed: dict[str, Any] = {}
    for entry in entries:
        if "=" not in entry:
            raise ValueError(f"{value_name} assignment must use CHANNEL=VALUE: {entry!r}")
        channel, raw_value = entry.split("=", 1)
        channel = channel.strip()
        if not channel:
            raise ValueError(f"{value_name} assignment has an empty channel")
        try:
            parsed[channel] = cast(raw_value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid {value_name} value for channel {channel!r}: {raw_value!r}") from exc
    return parsed


def _parse_resources(entries: list[str]) -> dict[str, str]:
    return _parse_channel_assignments(entries, value_name="SLURM resource", cast=str)


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


__all__ = ["main", "parse_args"]


if __name__ == "__main__":
    raise SystemExit(main())
