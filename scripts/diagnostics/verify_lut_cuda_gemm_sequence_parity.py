from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import torch
import yaml

from neptune_v03.localization.training_adapter import LocalizationTrainBatch
from neptune_v03.training.localizer_eval import build_localizer_eval_provider


def _eval_batches(config: dict[str, Any], *, backend: str):
    torch.manual_seed(18421)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(18421)
    selected = copy.deepcopy(config)
    train = selected["train"]
    online = train["online_generation"]
    online["simulation_backend"] = backend
    online["batch_strategy"] = "cached_window"
    online["steps_per_epoch"] = 2
    online["sequence_count"] = 2
    online["height"] = 32
    online["width"] = 32
    online["field_origin_sampling_mode"] = "grid"
    online["field_origin_stride_px"] = 16
    online["vector_psf_size"] = 25
    online["vector_batch_size"] = 128
    online["lut_simulation"].update(
        {
            "device": "cuda",
            "field_mode": "roi_origin",
            "field_stride": 32,
            "z_steps": 5,
            "psf_size": 25,
            "storage_dtype": "fp32",
        }
    )
    train["batch_size"] = 2
    train["eval"].update(
        {
            "enabled": True,
            "source": "online_generation",
            "batch_count": 2,
            "batch_size": 2,
            "seed": 18421,
        }
    )
    provider = build_localizer_eval_provider(
        train,
        config_base_dir=Path.cwd(),
        root_config=selected,
    )
    batches = provider()
    return batches


def _image_and_condition(model_input: Any) -> tuple[torch.Tensor, torch.Tensor | None]:
    if isinstance(model_input, tuple):
        return model_input[0], model_input[1]
    return model_input, None


def compare(config_path: Path) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this parity verification")
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    lut_batches = _eval_batches(config, backend="lut")
    cuda_batches = _eval_batches(config, backend="cuda_gemm")
    if len(lut_batches) != len(cuda_batches):
        raise AssertionError("LUT and CUDA GEMM eval batch counts differ")

    batch_reports = []
    for batch_index, (lut_batch, cuda_batch) in enumerate(zip(lut_batches, cuda_batches, strict=True)):
        lut_loc = lut_batch.inputs
        cuda_loc = cuda_batch.inputs
        if not isinstance(lut_loc, LocalizationTrainBatch) or not isinstance(cuda_loc, LocalizationTrainBatch):
            raise TypeError("eval provider did not return LocalizationTrainBatch inputs")
        lut_image, lut_condition = _image_and_condition(lut_loc.model_input)
        cuda_image, cuda_condition = _image_and_condition(cuda_loc.model_input)
        targets_equal = torch.equal(lut_loc.pxyz_tar.cpu(), cuda_loc.pxyz_tar.cpu())
        masks_equal = torch.equal(lut_loc.mask_tar.cpu(), cuda_loc.mask_tar.cpu())
        detections_equal = torch.equal(lut_loc.detect_tar.cpu(), cuda_loc.detect_tar.cpu())
        backgrounds_equal = torch.equal(lut_loc.bkg_tar.cpu(), cuda_loc.bkg_tar.cpu())
        conditions_equal = (
            lut_condition is None and cuda_condition is None
        ) or (
            lut_condition is not None
            and cuda_condition is not None
            and torch.equal(lut_condition.cpu(), cuda_condition.cpu())
        )
        if not (targets_equal and masks_equal and detections_equal and backgrounds_equal and conditions_equal):
            raise AssertionError(f"seeded eval scene/targets differ in batch {batch_index}")
        if tuple(lut_image.shape) != tuple(cuda_image.shape):
            raise AssertionError(f"eval image shapes differ in batch {batch_index}")
        difference = lut_image.float() - cuda_image.float()
        lut_flat = lut_image.float().reshape(-1)
        cuda_flat = cuda_image.float().reshape(-1)
        lut_centered = lut_flat - lut_flat.mean()
        cuda_centered = cuda_flat - cuda_flat.mean()
        denominator = torch.linalg.vector_norm(lut_centered) * torch.linalg.vector_norm(cuda_centered)
        ncc = 1.0 if float(denominator) == 0.0 else float(torch.dot(lut_centered, cuda_centered) / denominator)
        batch_reports.append(
            {
                "batch_index": batch_index,
                "image_shape": list(lut_image.shape),
                "emitter_counts_per_sample": lut_loc.mask_tar.sum(dim=1).cpu().tolist(),
                "targets_equal": targets_equal,
                "masks_equal": masks_equal,
                "detections_equal": detections_equal,
                "background_targets_equal": backgrounds_equal,
                "conditioning_equal": conditions_equal,
                "rendered_input_exact_equal": torch.equal(lut_image.cpu(), cuda_image.cpu()),
                "rendered_input_mae_adu": float(difference.abs().mean()),
                "rendered_input_max_abs_delta_adu": float(difference.abs().max()),
                "rendered_input_ncc": ncc,
            }
        )
    return {
        "schema_version": "lut_cuda_gemm_cached_sequence_parity.v1",
        "cuda_device": torch.cuda.get_device_name(torch.cuda.current_device()),
        "seed": 18421,
        "batch_count": len(batch_reports),
        "scene_and_target_parity": all(
            report["targets_equal"]
            and report["masks_equal"]
            and report["detections_equal"]
            and report["background_targets_equal"]
            and report["conditioning_equal"]
            for report in batch_reports
        ),
        "rendered_pixel_arrays_expected_to_differ_by_renderer": True,
        "batches": batch_reports,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = compare(args.config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
