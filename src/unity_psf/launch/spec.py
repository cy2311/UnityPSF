from __future__ import annotations

import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence


@dataclass(frozen=True)
class LaunchSpec:
    job_name: str
    entrypoint: str
    args: Sequence[str] = ()
    env: Mapping[str, str] = field(default_factory=dict)
    workdir: Path | str = Path(".")
    output_dir: Path | str = Path("output")
    logs_dir: Path | str = Path("logs")
    resources: Mapping[str, str | int] = field(default_factory=dict)
    python_executable: str = "python"


def build_local_launch_command(spec: LaunchSpec) -> list[str]:
    return [spec.python_executable, "-m", spec.entrypoint, *[str(arg) for arg in spec.args]]


def write_slurm_script(spec: LaunchSpec, path: Path | str) -> Path:
    script_path = Path(path)
    script_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "#!/usr/bin/env bash",
        f"#SBATCH --job-name={spec.job_name}",
    ]
    lines.extend(_resource_lines(spec))
    lines.extend(
        [
            f"#SBATCH --output={spec.logs_dir}/%x-%j.out",
            f"#SBATCH --error={spec.logs_dir}/%x-%j.err",
            "",
            "set -euo pipefail",
            f"mkdir -p {shlex.quote(str(spec.output_dir))} {shlex.quote(str(spec.logs_dir))}",
        ]
    )
    for key, value in sorted(spec.env.items()):
        lines.append(f"export {key}={shlex.quote(str(value))}")
    lines.extend(
        [
            f"cd {shlex.quote(str(spec.workdir))}",
            shlex.join(build_local_launch_command(spec)),
            "",
        ]
    )
    script_path.write_text("\n".join(lines), encoding="utf-8")
    return script_path


def _resource_lines(spec: LaunchSpec) -> list[str]:
    resources = spec.resources
    lines: list[str] = []
    if "partition" in resources:
        lines.append(f"#SBATCH --partition={resources['partition']}")
    if "gpus" in resources:
        lines.append(f"#SBATCH --gres=gpu:{resources['gpus']}")
    if "cpus_per_task" in resources:
        lines.append(f"#SBATCH --cpus-per-task={resources['cpus_per_task']}")
    if "mem" in resources:
        lines.append(f"#SBATCH --mem={resources['mem']}")
    if "time" in resources:
        lines.append(f"#SBATCH --time={resources['time']}")
    return lines
