"""Independent per-channel orchestration for every UnityPSF expert family."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
import json
import os
from pathlib import Path
import subprocess
from typing import Any

from unity_psf.contracts.modality import ChannelLayout, PSFModality
from unity_psf.launch import LaunchSpec, write_slurm_script
from unity_psf.runtime.layout import ensure_run_layout


class ChannelRunStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass(frozen=True)
class ChannelRunSpec:
    """Everything needed to launch one independent channel run."""

    expert_type: PSFModality
    instance_id: str
    channel_id: str
    seed: int
    run_root: Path
    run_name: str
    crop: tuple[int, int, int, int] | None = None
    anchor_profile: str | None = None
    calibration_ref: str | None = None
    prototype_ref: str | None = None
    entrypoint: str = "unity_psf.training.run_localization"
    config_path: Path | None = None
    extra_args: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "expert_type", PSFModality.parse(self.expert_type))
        _validate_run_name(self.channel_id, "channel_id")
        _validate_run_name(self.run_name, "run_name")
        object.__setattr__(self, "run_root", Path(self.run_root))
        if self.config_path is not None:
            object.__setattr__(self, "config_path", Path(self.config_path))
        if self.crop is not None:
            if len(self.crop) != 4 or any(int(value) != value for value in self.crop):
                raise ValueError("channel crop must contain four integer values")
            object.__setattr__(self, "crop", tuple(int(value) for value in self.crop))
        if int(self.seed) < 0:
            raise ValueError("channel seed must be non-negative")
        object.__setattr__(self, "seed", int(self.seed))
        object.__setattr__(self, "extra_args", tuple(str(value) for value in self.extra_args))

    @property
    def run_dir(self) -> Path:
        return self.run_root / self.run_name

    @property
    def command_args(self) -> tuple[str, ...]:
        args: list[str] = []
        if self.config_path is not None:
            args.extend(("--config", str(self.config_path)))
        args.extend(("--run-root", str(self.run_root), "--run-name", self.run_name, "--seed", str(self.seed)))
        args.extend(self.extra_args)
        return tuple(args)

    def to_dict(self) -> dict[str, object]:
        return {
            "expert_type": self.expert_type.value,
            "instance_id": self.instance_id,
            "channel_id": self.channel_id,
            "seed": self.seed,
            "run_root": str(self.run_root),
            "run_name": self.run_name,
            "run_dir": str(self.run_dir),
            "crop": None if self.crop is None else list(self.crop),
            "anchor_profile": self.anchor_profile,
            "calibration_ref": self.calibration_ref,
            "prototype_ref": self.prototype_ref,
            "entrypoint": self.entrypoint,
            "config_path": None if self.config_path is None else str(self.config_path),
            "extra_args": list(self.extra_args),
        }


@dataclass(frozen=True)
class ChannelExecutionResult:
    spec: ChannelRunSpec
    status: ChannelRunStatus
    exit_code: int | None = None
    error: str | None = None

    @property
    def run_dir(self) -> Path:
        return self.spec.run_dir

    def to_dict(self) -> dict[str, object]:
        return {
            "channel_id": self.spec.channel_id,
            "instance_id": self.spec.instance_id,
            "status": self.status.value,
            "exit_code": self.exit_code,
            "error": self.error,
            "run_dir": str(self.spec.run_dir),
        }


@dataclass(frozen=True)
class MultichannelExecutionResult:
    parent_run_dir: Path
    status: ChannelRunStatus
    channels: Mapping[str, ChannelExecutionResult]


@dataclass(frozen=True)
class SlurmChannelJob:
    channel_id: str
    script_path: Path
    launch_spec: LaunchSpec


@dataclass(frozen=True)
class MultichannelTrainingPlan:
    """A parent plan containing independent single-channel launch specs."""

    expert_type: PSFModality
    parent_run_root: Path
    parent_run_name: str
    channel_specs: tuple[ChannelRunSpec, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "expert_type", PSFModality.parse(self.expert_type))
        object.__setattr__(self, "parent_run_root", Path(self.parent_run_root))
        _validate_run_name(self.parent_run_name, "parent_run_name")
        if not self.channel_specs:
            raise ValueError("multichannel plan requires at least one channel")
        channel_ids = tuple(spec.channel_id for spec in self.channel_specs)
        if len(channel_ids) != len(set(channel_ids)):
            raise ValueError("channel run specs must have unique channel IDs")
        if any(spec.expert_type is not self.expert_type for spec in self.channel_specs):
            raise ValueError("all channel run specs must use the plan expert_type")
        run_dirs = tuple(spec.run_dir for spec in self.channel_specs)
        if len(run_dirs) != len(set(run_dirs)):
            raise ValueError("channel run specs must have unique run directories")

    @property
    def parent_run_dir(self) -> Path:
        return self.parent_run_root / self.parent_run_name

    @property
    def channel_ids(self) -> tuple[str, ...]:
        return tuple(spec.channel_id for spec in self.channel_specs)

    def to_dict(self) -> dict[str, object]:
        return {
            "expert_type": self.expert_type.value,
            "parent_run_root": str(self.parent_run_root),
            "parent_run_name": self.parent_run_name,
            "parent_run_dir": str(self.parent_run_dir),
            "channels": [spec.to_dict() for spec in self.channel_specs],
        }

    def execute_local(
        self,
        runner: Callable[[ChannelRunSpec], int | None],
        *,
        continue_on_error: bool = True,
    ) -> MultichannelExecutionResult:
        """Run channels sequentially while preserving completed siblings."""

        ensure_run_layout(self.parent_run_root, self.parent_run_name)
        results: dict[str, ChannelExecutionResult] = {}
        self._write_manifest(results, status=ChannelRunStatus.PENDING)
        for spec in self.channel_specs:
            spec.run_dir.mkdir(parents=True, exist_ok=True)
            try:
                results[spec.channel_id] = ChannelExecutionResult(spec, ChannelRunStatus.RUNNING)
                self._write_manifest(results, status=ChannelRunStatus.RUNNING)
                exit_code = runner(spec)
                normalized_exit_code = 0 if exit_code is None else int(exit_code)
                if normalized_exit_code != 0:
                    raise RuntimeError(f"channel runner returned exit code {normalized_exit_code}")
                results[spec.channel_id] = ChannelExecutionResult(
                    spec,
                    ChannelRunStatus.COMPLETED,
                    exit_code=normalized_exit_code,
                )
            except Exception as exc:
                results[spec.channel_id] = ChannelExecutionResult(
                    spec,
                    ChannelRunStatus.FAILED,
                    exit_code=None,
                    error=str(exc),
                )
                self._write_manifest(results, status=ChannelRunStatus.FAILED)
                if not continue_on_error:
                    raise
            self._write_manifest(results, status=self._overall_status(results))
        status = self._overall_status(results)
        self._write_manifest(results, status=status)
        return MultichannelExecutionResult(self.parent_run_dir, status, dict(results))

    def execute_local_subprocess(
        self,
        *,
        python_executable: str = "python",
        workdir: Path | str | None = None,
        env: Mapping[str, str] | None = None,
        continue_on_error: bool = True,
    ) -> MultichannelExecutionResult:
        """Run standard channel entrypoints sequentially in child processes."""

        def runner(spec: ChannelRunSpec) -> int:
            completed = subprocess.run(
                [python_executable, "-m", spec.entrypoint, *spec.command_args],
                cwd=None if workdir is None else str(workdir),
                env=None if env is None else {**os.environ, **env},
                check=False,
            )
            return int(completed.returncode)

        return self.execute_local(runner, continue_on_error=continue_on_error)

    def build_slurm_jobs(
        self,
        *,
        workdir: Path | str = ".",
        resources: Mapping[str, str | int] | None = None,
        python_executable: str = "python",
    ) -> tuple[tuple[ChannelRunSpec, LaunchSpec], ...]:
        jobs = []
        for spec in self.channel_specs:
            launch = LaunchSpec(
                job_name=f"{self.parent_run_name}-{spec.channel_id}",
                entrypoint=spec.entrypoint,
                args=spec.command_args,
                workdir=Path(workdir),
                output_dir=spec.run_dir,
                logs_dir=self.parent_run_dir / "logs" / spec.channel_id,
                resources=dict(resources or {}),
                python_executable=python_executable,
            )
            jobs.append((spec, launch))
        return tuple(jobs)

    def write_slurm_scripts(
        self,
        script_root: Path | str,
        *,
        workdir: Path | str = ".",
        resources: Mapping[str, str | int] | None = None,
        python_executable: str = "python",
    ) -> tuple[SlurmChannelJob, ...]:
        script_root = Path(script_root)
        jobs = []
        for spec, launch in self.build_slurm_jobs(
            workdir=workdir,
            resources=resources,
            python_executable=python_executable,
        ):
            script_path = script_root / f"{spec.channel_id}.sh"
            write_slurm_script(launch, script_path)
            jobs.append(SlurmChannelJob(spec.channel_id, script_path, launch))
        return tuple(jobs)

    def _overall_status(self, results: Mapping[str, ChannelExecutionResult]) -> ChannelRunStatus:
        if any(item.status is ChannelRunStatus.FAILED for item in results.values()):
            return ChannelRunStatus.FAILED
        if len(results) == len(self.channel_specs) and all(
            item.status is ChannelRunStatus.COMPLETED for item in results.values()
        ):
            return ChannelRunStatus.COMPLETED
        return ChannelRunStatus.RUNNING if results else ChannelRunStatus.PENDING

    def _write_manifest(
        self,
        results: Mapping[str, ChannelExecutionResult],
        *,
        status: ChannelRunStatus,
    ) -> Path:
        metadata_dir = self.parent_run_dir / "metadata"
        metadata_dir.mkdir(parents=True, exist_ok=True)
        path = metadata_dir / "multichannel_manifest.json"
        payload = {
            "schema_version": "unitypsf.multichannel_run.v1",
            "expert_type": self.expert_type.value,
            "status": status.value,
            "plan": self.to_dict(),
            "channels": {channel_id: result.to_dict() for channel_id, result in results.items()},
        }
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return path


def build_multichannel_plan(
    *,
    channel_layout: ChannelLayout | Mapping[str, object] | Sequence[object],
    expert_type: PSFModality | str,
    run_root: Path | str,
    run_name: str | None = None,
    seed: int = 0,
    channel_seeds: Mapping[str, int] | None = None,
    prototype_ref: str | Mapping[str, str] | None = None,
    entrypoint: str | Mapping[PSFModality | str, str] = "unity_psf.training.run_localization",
    config_path: Path | str | None = None,
    extra_args: Mapping[str, Sequence[str]] | None = None,
) -> MultichannelTrainingPlan:
    layout = ChannelLayout.from_value(channel_layout)
    modality = PSFModality.parse(expert_type)
    selected_run_name = modality.value if run_name is None else str(run_name)
    channel_seeds = dict(channel_seeds or {})
    extra_args = dict(extra_args or {})
    unknown_seed_channels = set(channel_seeds).difference(layout.channel_ids)
    if unknown_seed_channels:
        raise ValueError(f"unknown channel seed entries: {sorted(unknown_seed_channels)}")
    unknown_extra_channels = set(extra_args).difference(layout.channel_ids)
    if unknown_extra_channels:
        raise ValueError(f"unknown channel extra_args entries: {sorted(unknown_extra_channels)}")
    if isinstance(prototype_ref, Mapping):
        unknown_prototype_channels = set(prototype_ref).difference(layout.channel_ids)
        if unknown_prototype_channels:
            raise ValueError(f"unknown channel prototype entries: {sorted(unknown_prototype_channels)}")
    channels = []
    for channel in layout.channels:
        channel_id = channel.channel_id
        channels.append(
            ChannelRunSpec(
                expert_type=modality,
                instance_id=channel_id,
                channel_id=channel_id,
                seed=int(channel_seeds.get(channel_id, seed)),
                run_root=Path(run_root) / selected_run_name / "channels",
                run_name=channel_id,
                crop=channel.crop,
                anchor_profile=channel.anchor_profile,
                calibration_ref=channel.calibration_ref,
                prototype_ref=_resolve_channel_value(prototype_ref, channel_id),
                entrypoint=_resolve_entrypoint(entrypoint, modality),
                config_path=None if config_path is None else Path(config_path),
                extra_args=tuple(str(value) for value in extra_args.get(channel_id, ())),
            )
        )
    return MultichannelTrainingPlan(
        expert_type=modality,
        parent_run_root=Path(run_root),
        parent_run_name=selected_run_name,
        channel_specs=tuple(channels),
    )


def build_multimodal_training_plans(
    *,
    channel_layouts: Mapping[PSFModality | str, ChannelLayout | Mapping[str, object] | Sequence[object]],
    run_root: Path | str,
    run_name: str | None = None,
    seed: int = 0,
    entrypoints: Mapping[PSFModality | str, str] | None = None,
    channel_seeds: Mapping[PSFModality | str, Mapping[str, int]] | None = None,
    prototype_refs: Mapping[PSFModality | str, str | Mapping[str, str]] | None = None,
) -> tuple[MultichannelTrainingPlan, ...]:
    """Build one independent left/right plan for each requested PSF modality."""

    plans = []
    for raw_modality, channel_layout in channel_layouts.items():
        modality = PSFModality.parse(raw_modality)
        selected_entrypoint = "unity_psf.training.run_localization"
        if entrypoints is not None:
            selected_entrypoint = _resolve_entrypoint(entrypoints, modality)
        selected_run_name = (
            modality.value
            if run_name is None
            else f"{run_name}-{modality.value}"
        )
        selected_channel_seeds = _resolve_modality_mapping(channel_seeds, modality)
        selected_prototype_refs = _resolve_modality_value(prototype_refs, modality)
        plans.append(
            build_multichannel_plan(
                channel_layout=channel_layout,
                expert_type=modality,
                run_root=run_root,
                run_name=selected_run_name,
                seed=seed,
                channel_seeds=selected_channel_seeds,
                prototype_ref=selected_prototype_refs,
                entrypoint=selected_entrypoint,
            )
        )
    return tuple(plans)


def _resolve_channel_value(value: str | Mapping[str, str] | None, channel_id: str) -> str | None:
    if value is None:
        return None
    if isinstance(value, Mapping):
        return None if channel_id not in value else str(value[channel_id])
    return str(value)


def _resolve_modality_mapping(
    value: Mapping[PSFModality | str, Mapping[str, int]] | None,
    modality: PSFModality,
) -> Mapping[str, int] | None:
    if value is None:
        return None
    for key, selected in value.items():
        if PSFModality.parse(key) is modality:
            return selected
    return None


def _resolve_modality_value(
    value: Mapping[PSFModality | str, str | Mapping[str, str]] | None,
    modality: PSFModality,
) -> str | Mapping[str, str] | None:
    if value is None:
        return None
    for key, selected in value.items():
        if PSFModality.parse(key) is modality:
            return selected
    return None


def _resolve_entrypoint(
    value: str | Mapping[PSFModality | str, str], modality: PSFModality,
) -> str:
    if isinstance(value, Mapping):
        for key, entrypoint in value.items():
            if PSFModality.parse(key) is modality:
                return str(entrypoint)
        raise ValueError(f"no entrypoint configured for modality {modality.value!r}")
    return str(value)


def _validate_run_name(value: str, label: str) -> None:
    text = str(value).strip()
    path = Path(text)
    if not text or path.is_absolute() or ".." in path.parts or len(path.parts) != 1:
        raise ValueError(f"{label} must be a single relative path component")


__all__ = [
    "ChannelExecutionResult",
    "ChannelRunSpec",
    "ChannelRunStatus",
    "MultichannelExecutionResult",
    "MultichannelTrainingPlan",
    "SlurmChannelJob",
    "build_multichannel_plan",
    "build_multimodal_training_plans",
]
