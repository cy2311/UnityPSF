#!/usr/bin/env python3
"""Validate and submit the Neptune v0.3 formal training default."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from neptune_v03.config import materialize_config
from neptune_v03.localization.runtime_config import build_localization_runtime_config

BASE_CONFIG = ROOT / "configs" / "microtube_base.yaml"
STANDARD_OVERRIDE = ROOT / "configs" / "overrides" / "standard_roi_gamma.yaml"
BATCH_BUDGET_OVERRIDE = ROOT / "configs" / "overrides" / "standard_roi_gamma_batch_budget.yaml"
SBATCH_PATH = ROOT / "scripts" / "train" / "standard_roi_gamma_batch_budget.sbatch"

EXPECTED: dict[str, Any] = {
    "epochs": 999,
    "max_batches": 10000,
    "batch_size": 24,
    "online_generation.steps_per_epoch": 417,
    "online_generation.height": 128,
    "online_generation.width": 128,
    "online_generation.batch_strategy": "cached_window",
    "online_generation.sequence_count": 64,
    "online_generation.conditioning_mode": "film",
    "online_generation.expert_mode": "soft_moe",
    "online_generation.append_domain_onehot": True,
    "online_generation.domain_balance_mode": "alternate_step",
    "online_generation.emitter_density_um2": 1.0,
    "eval.enabled": True,
    "eval.source": "online_generation",
    "eval.batch_count": 4,
    "eval.batch_size": 4,
    "simulation.psf.psf_type": "vector",
    "peak_zmap_bootstrap.enabled": True,
    "peak_zmap_bootstrap.frame_start": 0,
    "peak_zmap_bootstrap.frame_stop": 100,
    "joint_training.real_loc_enabled": False,
    "joint_training.real_batch_interval": 0,
    "joint_training.real_loc_start_step": 0,
    "nat_wake.patch_size_px": 25,
    "nat_wake.loss_mode": "roi_projection_poisson_nll",
    "nat_wake.feedback_interval_steps": 417,
    "roi_bank_gamma.enabled": True,
    "roi_bank_gamma.fixed_roi_library": True,
    "roi_bank_gamma.roi_size_px": 128,
    "roi_bank_gamma.roi_bank_frame_range": [0, 100],
    "roi_bank_gamma.update_interval_epochs": 1,
    "roi_bank_gamma.start_epoch": 1,
    "roi_bank_gamma.stop_epoch": 999,
    "roi_bank_gamma.gamma_steps": 100,
    "roi_bank_gamma.gamma_lr": 0.025,
    "roi_bank_gamma.num_posterior_samples": 25,
    "roi_bank_gamma.roi_bank_objective": "importance_wake",
    "roi_bank_gamma.target_projected_emitters": 5000,
    "roi_bank_gamma.auto_heldout_min_rois": 20,
    "roi_bank_gamma.auto_heldout_max_rois": 20,
    "roi_bank_gamma.start_batch": 2000,
    "roi_bank_gamma.stop_batch": 10000,
    "roi_bank_gamma.update_interval_batches": 500,
}

RUNTIME_EXPECTED: dict[str, Any] = {
    "loss.name": "active_smlm_gmm_loss",
    "loss.params.gmm_target_chunk": 4,
    "loss.params.gmm_component_chunk": 64,
    "loss.params.gmm_backend": "mixture_same_family",
    "loss.params.photon_scale": 31000.0,
    "model.params.depth_shared": 2,
    "model.params.depth_union": 2,
    "model.params.nfeatures_init": 48,
    "model.params.nfeatures_inter": None,
    "model.params.upsample_mode": "nearest",
    "model.params.norm_start_level": -1,
    "model.params.depthwise": True,
    "batch_provider.params.batch_strategy": "cached_window",
    "batch_provider.params.sequence_count": 64,
    "batch_provider.params.simulation_backend": "lut",
    "resolved_contract.batch_provider.batch_strategy": "cached_window",
    "resolved_contract.batch_provider.sequence_count": 64,
    "resolved_contract.training_runtime.scheduler.params.step_size": 1000,
    "resolved_contract.training_runtime.scheduler.step_unit": "optimizer_step",
}


def _nested_get(payload: dict[str, Any], dotted: str) -> Any:
    if dotted.startswith("simulation.") or dotted.startswith("optical.") or dotted.startswith("camera."):
        current: Any = payload
    else:
        current = payload["train"]
    for part in dotted.split("."):
        current = current[part]
    return current


def _root_nested_get(payload: dict[str, Any], dotted: str) -> Any:
    current: Any = payload
    for part in dotted.split("."):
        current = current[part]
    return current


def resolved_standard_config() -> dict[str, Any]:
    return materialize_config(BASE_CONFIG, STANDARD_OVERRIDE, BATCH_BUDGET_OVERRIDE)


def validate_default() -> list[str]:
    payload = resolved_standard_config()
    failures: list[str] = []
    for key, expected in EXPECTED.items():
        actual = _nested_get(payload, key) if "." in key else payload["train"][key]
        if actual != expected:
            failures.append(f"{key}: expected {expected!r}, got {actual!r}")
    runtime = build_localization_runtime_config(payload, config_base_dir=ROOT / "configs", seed=0)
    for key, expected in RUNTIME_EXPECTED.items():
        actual = _root_nested_get(runtime, key)
        if actual != expected:
            failures.append(f"runtime.{key}: expected {expected!r}, got {actual!r}")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="validate the standard config and exit")
    parser.add_argument("--submit", action="store_true", help="submit the standard SLURM job after validation")
    args = parser.parse_args()

    failures = validate_default()
    if failures:
        print("Neptune v0.3 standard config check failed:")
        for failure in failures:
            print(f"  - {failure}")
        return 1

    print("Neptune v0.3 standard config check passed.")
    print(f"Base config: {BASE_CONFIG.relative_to(REPO_ROOT)}")
    print(f"Override:    {STANDARD_OVERRIDE.relative_to(REPO_ROOT)}")
    print(f"Override:    {BATCH_BUDGET_OVERRIDE.relative_to(REPO_ROOT)}")
    print(f"SLURM:       {SBATCH_PATH.relative_to(REPO_ROOT)}")

    if args.submit:
        result = subprocess.run(["sbatch", str(SBATCH_PATH.relative_to(REPO_ROOT))], cwd=REPO_ROOT, check=False)
        return int(result.returncode)
    if not args.check:
        print("Use --submit to launch the standard training job.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
