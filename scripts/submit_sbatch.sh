#!/usr/bin/env bash
# Submit an active UnityPSF SLURM script with the project root as cwd.
# Slurm resolves #SBATCH output paths before the batch script starts, so the
# caller must establish the cwd before invoking sbatch.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

if [[ $# -lt 1 ]]; then
  printf 'Usage: %s <script.sbatch> [sbatch options]\n' "${0##*/}" >&2
  exit 2
fi

JOB_SCRIPT="$1"
shift
SBATCH_ARGS=("$@")
if [[ "$JOB_SCRIPT" != /* ]]; then
  if [[ -f "$PWD/$JOB_SCRIPT" ]]; then
    JOB_SCRIPT="$(cd "$(dirname "$PWD/$JOB_SCRIPT")" && pwd)/$(basename "$JOB_SCRIPT")"
  else
    JOB_SCRIPT="$PROJECT_ROOT/$JOB_SCRIPT"
  fi
fi

if [[ ! -f "$JOB_SCRIPT" ]]; then
  printf 'SLURM script not found: %s\n' "$JOB_SCRIPT" >&2
  exit 2
fi

cd "$PROJECT_ROOT"
mkdir -p logs/slurm
exec sbatch "${SBATCH_ARGS[@]}" "$JOB_SCRIPT"
