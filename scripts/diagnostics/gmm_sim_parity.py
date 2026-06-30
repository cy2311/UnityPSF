from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
V03_ROOT = REPO_ROOT / "neptune_v0.3"
IWAE_ROOT = REPO_ROOT / "neptune_iwae"


def _install_import_paths() -> None:
    paths = [
        str(V03_ROOT / "src"),
        str(IWAE_ROOT),
    ]
    for path in reversed(paths):
        if path not in sys.path:
            sys.path.insert(0, path)


def _tensor_stats(value: torch.Tensor | np.ndarray) -> dict[str, Any]:
    arr = value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)
    arr = np.asarray(arr, dtype=np.float64)
    if arr.size == 0:
        return {"shape": list(arr.shape), "count": 0}
    return {
        "shape": list(arr.shape),
        "count": int(arr.size),
        "min": float(np.nanmin(arr)),
        "max": float(np.nanmax(arr)),
        "mean": float(np.nanmean(arr)),
        "std": float(np.nanstd(arr)),
        "p01": float(np.nanpercentile(arr, 1)),
        "p50": float(np.nanpercentile(arr, 50)),
        "p99": float(np.nanpercentile(arr, 99)),
    }


def _active_values(values: torch.Tensor | np.ndarray, mask: torch.Tensor | np.ndarray) -> np.ndarray:
    values_np = values.detach().cpu().numpy() if torch.is_tensor(values) else np.asarray(values)
    mask_np = mask.detach().cpu().numpy().astype(bool) if torch.is_tensor(mask) else np.asarray(mask).astype(bool)
    return np.asarray(values_np)[mask_np]


def run_gmm_loss_fixture() -> dict[str, Any]:
    from neptune_v03.localization.losses import ActiveSMLMGMMLoss
    from smlm_v2a.training.losses import GMMLoss

    torch.manual_seed(123)
    dtype = torch.float64
    batch, height, width, n_targets = 2, 5, 6, 4
    y_out = torch.zeros((batch, 10, height, width), dtype=dtype)
    y_out[:, 0] = torch.rand((batch, height, width), dtype=dtype) * 0.28 + 0.02
    y_out[:, 1] = torch.rand((batch, height, width), dtype=dtype) * 0.8 + 0.6
    y_out[:, 2] = torch.rand((batch, height, width), dtype=dtype) * 0.6 - 0.3
    y_out[:, 3] = torch.rand((batch, height, width), dtype=dtype) * 0.6 - 0.3
    y_out[:, 4] = torch.rand((batch, height, width), dtype=dtype) * 1.2 - 0.6
    y_out[:, 5:9] = torch.rand((batch, 4, height, width), dtype=dtype) * 0.35 + 0.15
    y_out[:, 9] = torch.rand((batch, height, width), dtype=dtype) * 0.5

    mask = torch.tensor([[True, True, True, False], [True, False, True, True]], dtype=torch.bool)
    x = torch.tensor([[0.5, 2.5, 4.5, 0.0], [1.5, 3.5, 5.2, 0.0]], dtype=dtype)
    y = torch.tensor([[0.5, 1.5, 3.5, 0.0], [0.8, 2.5, 4.2, 0.0]], dtype=dtype)
    z_um = torch.tensor([[-0.3, 0.0, 0.24, 0.0], [0.42, -0.12, 0.18, 0.0]], dtype=dtype)
    photons = torch.tensor([[18000.0, 21000.0, 15000.0, 0.0], [25000.0, 12000.0, 19500.0, 0.0]], dtype=dtype)
    v03_target = torch.stack((x, y, z_um, photons), dim=-1)
    legacy_target = torch.stack((photons / 31000.0, x, y, z_um / 0.6), dim=-1)
    bkg = torch.rand((batch, height, width), dtype=dtype) * 0.5
    detect = torch.zeros((batch, height, width), dtype=dtype)

    new_criterion = ActiveSMLMGMMLoss(
        xyoffset=(0.0, 0.0),
        ch_weight=(1.0, 1.0),
        photon_scale=31000.0,
        z_scale=0.6,
        gmm_backend="manual_chunked",
        gmm_target_chunk=2,
        gmm_component_chunk=7,
    )
    old_criterion = GMMLoss(
        xyoffset=(0.0, 0.0),
        ch_weight=(1.0, 1.0),
        forward_safety=True,
        gmm_backend="manual_chunked",
        gmm_target_chunk=2,
        gmm_component_chunk=7,
    )
    new_loss = new_criterion.forward(y_out, detect, v03_target, mask, bkg)
    old_loss = old_criterion.forward(y_out, detect, legacy_target, mask, bkg)

    new_mix_criterion = ActiveSMLMGMMLoss(
        xyoffset=(0.0, 0.0),
        ch_weight=(1.0, 1.0),
        photon_scale=31000.0,
        z_scale=0.6,
        gmm_backend="mixture_same_family",
    )
    old_mix_criterion = GMMLoss(
        xyoffset=(0.0, 0.0),
        ch_weight=(1.0, 1.0),
        forward_safety=True,
        gmm_backend="mixture_same_family",
    )
    new_mix = new_mix_criterion.forward(y_out, detect, v03_target, mask, bkg)
    old_mix = old_mix_criterion.forward(y_out, detect, legacy_target, mask, bkg)

    diff = (new_loss - old_loss).detach().cpu()
    mix_diff = (new_mix - old_mix).detach().cpu()

    def old_components(criterion: GMMLoss) -> dict[str, Any]:
        p = y_out[:, 0]
        pxyz_mu = y_out[:, 1:5]
        pxyz_sig = y_out[:, 5:9]
        loss_gmm = criterion._compute_loss_gmm(p, detect, pxyz_mu, pxyz_sig, legacy_target, mask)
        loss_bkg = criterion._compute_loss_bkg(y_out[:, 9], bkg)
        loss = 2.0 * torch.stack((loss_gmm, loss_bkg), dim=1)
        return {
            "loss_bkg": loss_bkg.detach().cpu().tolist(),
            "loss_gmm": loss_gmm.detach().cpu().tolist(),
            "loss_total": loss.sum(dim=1).detach().cpu().tolist(),
            "loss_return": loss.detach().cpu().tolist(),
            "means": {
                "loss_gmm": float(loss_gmm.detach().mean().cpu().item()),
                "loss_bkg": float(loss_bkg.detach().mean().cpu().item()),
                "loss_total": float(loss.sum(dim=1).detach().mean().cpu().item()),
            },
        }

    def new_components(criterion: ActiveSMLMGMMLoss) -> dict[str, Any]:
        p = y_out[:, 0]
        pxyz_mu = y_out[:, 1:5]
        pxyz_sig = y_out[:, 5:9]
        target_gmm = criterion.target_adapter.v03_to_gmm_order(v03_target).to(dtype=y_out.dtype, device=y_out.device)
        loss_count, loss_loc = criterion._compute_gmm_terms(p, pxyz_mu, pxyz_sig, target_gmm, mask)
        loss_gmm = loss_count + loss_loc
        loss_bkg = torch.nn.functional.mse_loss(y_out[:, 9], bkg, reduction="none").sum(dim=(-1, -2))
        loss = 2.0 * torch.stack((loss_gmm, loss_bkg), dim=1)
        return {
            "loss_bkg": loss_bkg.detach().cpu().tolist(),
            "loss_gmm": loss_gmm.detach().cpu().tolist(),
            "loss_total": loss.sum(dim=1).detach().cpu().tolist(),
            "loss_return": loss.detach().cpu().tolist(),
            "internal": {
                "count_contribution": loss_count.detach().cpu().tolist(),
                "mixture_localization_contribution": loss_loc.detach().cpu().tolist(),
            },
            "means": {
                "loss_gmm": float(loss_gmm.detach().mean().cpu().item()),
                "loss_bkg": float(loss_bkg.detach().mean().cpu().item()),
                "loss_total": float(loss.sum(dim=1).detach().mean().cpu().item()),
                "internal_count_contribution": float(loss_count.detach().mean().cpu().item()),
                "internal_mixture_localization_contribution": float(loss_loc.detach().mean().cpu().item()),
            },
        }

    manual_old_components = old_components(old_criterion)
    manual_new_components = new_components(new_criterion)
    mixture_old_components = old_components(old_mix_criterion)
    mixture_new_components = new_components(new_mix_criterion)

    def component_deltas(old: dict[str, Any], new: dict[str, Any]) -> dict[str, Any]:
        deltas = {}
        for key in ("loss_bkg", "loss_gmm", "loss_total", "loss_return"):
            old_t = torch.as_tensor(old[key], dtype=torch.float64)
            new_t = torch.as_tensor(new[key], dtype=torch.float64)
            delta = new_t - old_t
            deltas[key] = {
                "max_abs_diff": float(delta.abs().max().item()),
                "mean_abs_diff": float(delta.abs().mean().item()),
            }
        return deltas

    return {
        "status": "ok",
        "fixture": {
            "batch": batch,
            "height": height,
            "width": width,
            "n_targets": n_targets,
            "photon_scale": 31000.0,
            "z_scale": 0.6,
            "target_order_old": ["photons_norm", "x_px", "y_px", "z_norm"],
            "target_order_v03": ["x_px", "y_px", "z_um", "photons"],
        },
        "manual_chunked": {
            "old_loss": old_loss.detach().cpu().tolist(),
            "new_loss": new_loss.detach().cpu().tolist(),
            "old_components": manual_old_components,
            "new_components": manual_new_components,
            "component_deltas": component_deltas(manual_old_components, manual_new_components),
            "max_abs_diff": float(diff.abs().max().item()),
            "mean_abs_diff": float(diff.abs().mean().item()),
            "per_channel_max_abs_diff": diff.abs().max(dim=0).values.tolist(),
        },
        "mixture_same_family": {
            "old_loss": old_mix.detach().cpu().tolist(),
            "new_loss": new_mix.detach().cpu().tolist(),
            "old_components": mixture_old_components,
            "new_components": mixture_new_components,
            "component_deltas": component_deltas(mixture_old_components, mixture_new_components),
            "max_abs_diff": float(mix_diff.abs().max().item()),
            "mean_abs_diff": float(mix_diff.abs().mean().item()),
            "per_channel_max_abs_diff": mix_diff.abs().max(dim=0).values.tolist(),
        },
    }


def _small_lut_config(zmap_path: Path, *, seed: int, device: str) -> dict[str, Any]:
    return {
        "optical": {
            "NA": 1.4,
            "pixel_size_nm_x": 101.11,
            "pixel_size_nm_y": 98.83,
            "wavelength_nm": 660.0,
        },
        "simulation": {
            "frames_per_sample": 5,
            "num_samples": 1,
            "background_uniform": [110.0, 110.0],
            "simulation": {"random_seed": int(seed)},
            "emitter": {
                "emitter_extent": [[-0.5, 63.5], [-0.5, 63.5], [-0.6, 0.6]],
                "field_extent": [[540, 604], [540, 604]],
                "z_range": [-0.6, 0.6],
                "density_um2": 1.0,
                "intensity_mu_sig": [20000.0, 1000.0],
                "intensity_clip": [0.0, 31000.0],
                "lifetime_avg": 1.0,
            },
            "profiles": {
                "train": {
                    "frames_per_sample": 5,
                    "num_samples": 1,
                    "emitter": {
                        "emitter_extent": [[-0.5, 63.5], [-0.5, 63.5], [-0.6, 0.6]],
                        "field_extent": [[540, 604], [540, 604]],
                        "density_um2": 1.0,
                    },
                }
            },
        },
        "train": {
            "batch_size": 3,
            "scaling": {"photon_max": 31000.0, "z_max": 0.6, "bg_max": 240.0},
            "online_generation": {
                "enabled": True,
                "batch_strategy": "sequence_window",
                "sequence_window_chunks": 1,
                "simulation_backend": "lut",
                "psf_type": "vector",
                "height": 64,
                "width": 64,
                "channels": 3,
                "emitter_density_um2": 1.0,
                "lifetime_avg": 1.0,
                "warmup_frames": 6.0,
                "background_range": [110.0, 110.0],
                "background_scale": 240.0,
                "photon_mean": 20000.0,
                "photon_sigma": 1000.0,
                "photon_range": [0.0, 31000.0],
                "z_range": [-0.6, 0.6],
                "domain_count": 1,
                "dual_domain_coeff_maps": [{"name": "left", "coeff_maps_npz": str(zmap_path)}],
                "lut_simulation": {
                    "zmap_npz": str(zmap_path),
                    "field_stride": 16,
                    "z_step_nm": 20.0,
                    "dtype": "fp16",
                    "subpixel_bins": 1,
                    "z_scale_nm": 600.0,
                    "place_delta": 0,
                    "subpixel_center_offset": 0.5,
                    "shift_sign": 1,
                    "device": str(device),
                    "psf_size": 51,
                },
            },
        },
    }


def _old_dataset_stats(cfg: dict[str, Any]) -> dict[str, Any]:
    from neptune_core.lut_simulation import LutSimulationBackend

    backend = LutSimulationBackend.from_config(cfg, base_dir=REPO_ROOT)
    dataset = backend.generate_dataset(copy.deepcopy(cfg), profile="train")
    sample = dataset.samples[0]
    frames = np.asarray(sample.frames, dtype=np.float32)
    param = np.asarray(sample.param_tar, dtype=np.float32)
    mask = np.asarray(sample.mask_tar, dtype=bool)
    bg = np.asarray(sample.bg_tar, dtype=np.float32)
    center_slice = slice(1, 4)
    center_mask = mask[center_slice]
    center_param = param[center_slice]
    return {
        "status": "ok",
        "metadata": dataset.metadata,
        "frames_adu": _tensor_stats(frames),
        "center_window_adu": _tensor_stats(frames[1:4]),
        "emitter_counts_per_frame": mask.sum(axis=1).astype(int).tolist(),
        "active_emitter_count_total": int(mask.sum()),
        "target_photons": _tensor_stats(_active_values(param[..., 0], mask)),
        "target_x_px": _tensor_stats(_active_values(param[..., 1], mask)),
        "target_y_px": _tensor_stats(_active_values(param[..., 2], mask)),
        "target_z_um": _tensor_stats(_active_values(param[..., 3], mask)),
        "center_target_photons": _tensor_stats(_active_values(center_param[..., 0], center_mask)),
        "center_target_x_px": _tensor_stats(_active_values(center_param[..., 1], center_mask)),
        "center_target_y_px": _tensor_stats(_active_values(center_param[..., 2], center_mask)),
        "center_target_z_um": _tensor_stats(_active_values(center_param[..., 3], center_mask)),
        "background_photons": _tensor_stats(bg),
        "frame_minus_bg": _tensor_stats(frames - bg),
    }


def _v03_batch_stats(cfg: dict[str, Any], *, seed: int) -> dict[str, Any]:
    from dataclasses import fields

    from neptune_v03.localization.online import OnlineBatchProviderConfig, build_online_batch_provider
    from neptune_v03.localization.runtime_config import build_localization_runtime_config

    runtime = build_localization_runtime_config(cfg, config_base_dir=REPO_ROOT, seed=seed)
    params = dict(runtime["batch_provider"]["params"])
    params.update({"batch_size": 3, "steps_per_epoch": 1, "seed": seed})
    field_names = {field.name for field in fields(OnlineBatchProviderConfig)}
    provider = build_online_batch_provider(OnlineBatchProviderConfig(**{k: v for k, v in params.items() if k in field_names}))
    batch = provider(epoch=1)[0].inputs
    model_input = batch.model_input[0] if isinstance(batch.model_input, tuple) else batch.model_input
    mask = batch.mask_tar.detach().cpu()
    pxyz = batch.pxyz_tar.detach().cpu()
    return {
        "status": "ok",
        "metadata": batch.metadata,
        "model_input_adu": _tensor_stats(model_input),
        "center_channel_adu": _tensor_stats(model_input[:, 1]),
        "emitter_counts_per_center_sample": mask.sum(dim=1).to(dtype=torch.int64).tolist(),
        "active_emitter_count_total": int(mask.sum().item()),
        "target_photons": _tensor_stats(pxyz[..., 3][mask]),
        "target_x_px": _tensor_stats(pxyz[..., 0][mask]),
        "target_y_px": _tensor_stats(pxyz[..., 1][mask]),
        "target_z_um": _tensor_stats(pxyz[..., 2][mask]),
        "background_target_scaled": _tensor_stats(batch.bkg_tar.detach().cpu()),
        "input_minus_camera_baseline": _tensor_stats(model_input - 398.6),
    }


def run_simulator_stats_fixture(*, zmap_path: Path, seed: int, device: str) -> dict[str, Any]:
    cfg = _small_lut_config(zmap_path, seed=seed, device=device)
    result: dict[str, Any] = {
        "fixture": {
            "zmap_path": str(zmap_path),
            "seed": int(seed),
            "device": str(device),
            "roi_size": 64,
            "frames_per_sample_old": 5,
            "center_samples_v03": 3,
            "density_um2": 1.0,
        }
    }
    try:
        result["old_lut"] = _old_dataset_stats(copy.deepcopy(cfg))
    except Exception as exc:  # pragma: no cover - diagnostics should report partial failures.
        result["old_lut"] = {"status": "error", "error": type(exc).__name__, "message": str(exc)}
    try:
        result["v03_lut_like"] = _v03_batch_stats(copy.deepcopy(cfg), seed=seed)
    except Exception as exc:  # pragma: no cover - diagnostics should report partial failures.
        result["v03_lut_like"] = {"status": "error", "error": type(exc).__name__, "message": str(exc)}
    if result.get("old_lut", {}).get("status") == "ok" and result.get("v03_lut_like", {}).get("status") == "ok":
        old_counts = np.asarray(result["old_lut"]["emitter_counts_per_frame"][1:4], dtype=np.float64)
        new_counts = np.asarray(result["v03_lut_like"]["emitter_counts_per_center_sample"], dtype=np.float64)
        result["comparison"] = {
            "center_count_old_mean": float(old_counts.mean()),
            "center_count_v03_mean": float(new_counts.mean()),
            "center_count_mean_delta": float(new_counts.mean() - old_counts.mean()),
            "old_input_adu_mean": float(result["old_lut"]["center_window_adu"]["mean"]),
            "v03_input_adu_mean": float(result["v03_lut_like"]["model_input_adu"]["mean"]),
            "input_adu_mean_delta": float(result["v03_lut_like"]["model_input_adu"]["mean"] - result["old_lut"]["center_window_adu"]["mean"]),
            "old_photon_mean": float(result["old_lut"]["center_target_photons"]["mean"]),
            "v03_photon_mean": float(result["v03_lut_like"]["target_photons"]["mean"]),
            "target_photon_mean_delta": float(result["v03_lut_like"]["target_photons"]["mean"] - result["old_lut"]["center_target_photons"]["mean"]),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose v0.3 vs neptune_iwae GMM loss and simulator parity.")
    parser.add_argument(
        "--output",
        default=str(V03_ROOT / ".local/tmp/diagnostics/gmm_sim_parity.json"),
        help="JSON report path.",
    )
    parser.add_argument(
        "--zmap",
        default=str(IWAE_ROOT / "test_data/microtube/zmap/left/alternating_full_roi_zernike_maps_nm.npz"),
        help="Coeff-map NPZ for simulator parity.",
    )
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    _install_import_paths()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "gmm_loss": run_gmm_loss_fixture(),
        "simulator_stats": run_simulator_stats_fixture(
            zmap_path=Path(args.zmap),
            seed=int(args.seed),
            device=str(args.device),
        ),
    }
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(str(output_path))


if __name__ == "__main__":
    main()
