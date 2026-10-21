from __future__ import annotations

import copy
import json
import math
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import load_pipeline_config
from phase_retrieval.api import run_phase_retrieval
from simulation.api import generate_dataset

from ..data.data_factory import FrameRescalar
from ..data.smlm_dataset import InferenceDataset
from ..generic.localizations import FrameLocSet
from ..training import train_kernels, train_setup


@dataclass
class FullFieldSample:
    row: int
    col: int
    x0: int
    y0: int
    frames: torch.Tensor
    loc_tar: FrameLocSet
    param_tar: np.ndarray
    mask_tar: np.ndarray


@dataclass
class FullFieldEvalData:
    samples: List[FullFieldSample]
    rows: int
    cols: int
    frame_window: int
    roi_size: int


def _roi_grid(
    *,
    full_h: int,
    full_w: int,
    roi: int,
    drop_top: bool = True,
    drop_bottom: bool = True,
    drop_right: bool = True,
) -> Tuple[List[Tuple[int, int, int, int]], int, int]:
    x_start = full_w // 2
    x_end = full_w
    y_start = 0
    y_end = full_h

    if drop_top:
        y_start += roi
    if drop_bottom:
        y_end -= roi
    if drop_right:
        x_end -= roi

    cols = (x_end - x_start) // roi
    rows = (y_end - y_start) // roi

    rois = []
    for r in range(rows):
        y0 = y_start + r * roi
        for c in range(cols):
            x0 = x_start + c * roi
            rois.append((r, c, x0, y0))
    return rois, rows, cols


def _digest_dict(data: dict) -> str:
    payload = json.dumps(data, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _sim_params_from_param(param) -> dict:
    sim = param.Simulation
    return {
        "photon_mu_sig": list(sim.photon_mu_sig),
        "bkg_uniform": list(sim.bkg_uniform),
        "lifetime_avg": float(sim.lifetime_avg),
        "density": None if sim.density is None else float(sim.density),
        "em_avg": None if sim.em_avg is None else float(sim.em_avg),
        "emitter_extent": [list(sim.emitter_extent[0]), list(sim.emitter_extent[1]), list(sim.emitter_extent[2])],
        "n_max": int(param.DataFilter.max_number_targets),
        "cam_pxszs": [float(v) for v in param.Camera.cam_pxszs],
        "channels_in": int(param.HyperParameter.channels_in),
    }


def _build_cache_meta(
    *,
    cfg: dict,
    tiff_path: Path,
    param,
    rows: int,
    cols: int,
    roi_size: int,
    frames_per_sample: int,
    full_h: int,
    full_w: int,
) -> dict:
    meta = {
        "tiff_path": str(tiff_path),
        "roi_size": int(roi_size),
        "rows": int(rows),
        "cols": int(cols),
        "frames_per_sample": int(frames_per_sample),
        "full_h": int(full_h),
        "full_w": int(full_w),
        "phase_config_digest": _digest_dict(cfg),
        "sim_params": _sim_params_from_param(param),
    }
    meta["cache_digest"] = _digest_dict(meta)
    return meta


def _cache_paths(cache_dir: Path, rows: int, cols: int) -> List[Path]:
    paths = []
    for r in range(rows):
        for c in range(cols):
            paths.append(cache_dir / f"roi_r{r:02d}_c{c:02d}.npz")
    return paths


def _load_full_field_cache(
    cache_dir: Path,
    *,
    cfg: dict,
    tiff_path: Path,
    param,
    calibration,
    frame_window: int,
    roi_size: int,
    frames_per_sample: int,
) -> FullFieldEvalData | None:
    meta_path = cache_dir / "metadata.json"
    if not meta_path.exists():
        return None
    try:
        meta = json.loads(meta_path.read_text())
    except json.JSONDecodeError:
        return None

    if meta.get("roi_size") != int(roi_size):
        return None
    if meta.get("frames_per_sample") != int(frames_per_sample):
        return None
    if meta.get("tiff_path") != str(tiff_path):
        return None
    if meta.get("phase_config_digest") != _digest_dict(cfg):
        return None
    if meta.get("sim_params") != _sim_params_from_param(param):
        return None

    rows = int(meta.get("rows", 0))
    cols = int(meta.get("cols", 0))
    if rows <= 0 or cols <= 0:
        return None

    expected_paths = _cache_paths(cache_dir, rows, cols)
    if not all(p.exists() for p in expected_paths):
        return None

    loc_meta = train_setup.setup_loc_meta(param, calibration)
    frame_offset = -(frame_window - 1) // 2

    samples: List[FullFieldSample] = []
    for r in range(rows):
        for c in range(cols):
            path = cache_dir / f"roi_r{r:02d}_c{c:02d}.npz"
            data = np.load(path, allow_pickle=False)
            frames = torch.from_numpy(data["frames"].astype(np.float32, copy=False))
            param_tar = data["param_tar"].astype(np.float32, copy=False)
            mask_tar = data["mask_tar"].astype(bool, copy=False)
            x0 = int(data["x0"])
            y0 = int(data["y0"])
            loc_tar = _locset_from_param(param_tar, mask_tar, loc_meta)
            loc_tar = loc_tar.realign_frame(frame_offset=frame_offset)
            samples.append(
                FullFieldSample(
                    row=r,
                    col=c,
                    x0=x0,
                    y0=y0,
                    frames=frames,
                    loc_tar=loc_tar,
                    param_tar=param_tar,
                    mask_tar=mask_tar,
                )
            )

    return FullFieldEvalData(samples=samples, rows=rows, cols=cols, frame_window=frame_window, roi_size=roi_size)


def _save_full_field_cache(
    cache_dir: Path,
    meta: dict,
    samples: Iterable[FullFieldSample],
) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    meta_path = cache_dir / "metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2))

    for sample in samples:
        out_path = cache_dir / f"roi_r{sample.row:02d}_c{sample.col:02d}.npz"
        np.savez_compressed(
            out_path,
            frames=sample.frames.cpu().numpy().astype(np.float32, copy=False),
            param_tar=sample.param_tar.astype(np.float32, copy=False),
            mask_tar=sample.mask_tar.astype(bool, copy=False),
            x0=int(sample.x0),
            y0=int(sample.y0),
        )


def _estimate_num_emitters(param, frames_per_sample: int) -> int:
    density = param.Simulation.density
    em_avg = param.Simulation.em_avg
    if density is None and em_avg is None:
        raise ValueError("Both density and em_avg are None; one must be provided.")
    if density is not None and em_avg is not None:
        raise ValueError("Both density and em_avg are set; only one should be provided.")

    x_extent = param.Simulation.emitter_extent[0]
    y_extent = param.Simulation.emitter_extent[1]
    area_px = (x_extent[1] - x_extent[0]) * (y_extent[1] - y_extent[0])
    px_x_nm, px_y_nm = param.Camera.cam_pxszs[:2]
    area_um = area_px * px_x_nm * px_y_nm * 1e-6
    em_avg = em_avg if em_avg is not None else density * area_um

    lifetime = float(param.Simulation.lifetime_avg)
    total = em_avg * (frames_per_sample + 6.0 * lifetime) / (lifetime + 1.0)
    return max(1, int(round(total)))


def _apply_sim_overrides(cfg: dict, param, calibration, *, frames_per_sample: int, n_max: int, num_emitters: int) -> None:
    cam = calibration.Camera
    cfg.setdefault("camera", {})
    cfg["camera"].update(
        {
            "qe": float(cam.QE),
            "spurious_charge": float(cam.spur_noise),
            "em_gain": float(cam.EMGain),
            "baseline": float(cam.offset),
            "read_sigma": float(math.sqrt(cam.var)),
            "e_per_adu": float(cam.A2D),
            "cam_pxszx": float(cam.cam_pxszs[0]),
            "cam_pxszy": float(cam.cam_pxszs[1]),
        }
    )

    cfg.setdefault("optical", {})
    cfg["optical"].update(
        {
            "pixel_size_nm_x": float(cam.cam_pxszs[0]),
            "pixel_size_nm_y": float(cam.cam_pxszs[1]),
        }
    )

    sim_cfg = cfg.setdefault("simulation", {})
    sim_cfg["background_uniform"] = list(param.Simulation.bkg_uniform)
    sim_cfg.setdefault("emitter", {})
    sim_cfg["emitter"].update(
        {
            "emitter_extent": [list(param.Simulation.emitter_extent[0]), list(param.Simulation.emitter_extent[1])],
            "z_range": [-0.6, 0.6],
            "intensity_mu_sig": list(param.Simulation.photon_mu_sig),
            "lifetime_avg": float(param.Simulation.lifetime_avg),
        }
    )
    sim_cfg.setdefault("profiles", {})
    sim_cfg["profiles"]["eval"] = {
        "frames_per_sample": int(frames_per_sample),
        "num_samples": 1,
        "num_emitters": int(num_emitters),
        "n_max": int(n_max),
    }
    sim_cfg.setdefault("simulation", {})


def _convert_z_um_to_index(param_tar: np.ndarray, *, z_min_um: float, z_step_nm: float, z_max_index: float) -> None:
    z_step_um = z_step_nm * 1e-3
    z_idx = (param_tar[..., 3] - z_min_um) / z_step_um
    np.clip(z_idx, 0.0, z_max_index - 1e-3, out=z_idx)
    param_tar[..., 3] = z_idx


def _locset_from_param(
    param_tar: np.ndarray,
    mask_tar: np.ndarray,
    loc_meta: FrameLocSet,
) -> FrameLocSet:
    mask_idx = np.where(mask_tar)
    if mask_idx[0].size == 0:
        return FrameLocSet(**loc_meta.meta)

    indf = torch.from_numpy(mask_idx[0].astype(np.int32))
    phot = torch.from_numpy(param_tar[mask_idx][:, 0].astype(np.float32))
    xyz = torch.from_numpy(param_tar[mask_idx][:, 1:4].astype(np.float32))
    ids = torch.arange(len(indf), dtype=loc_meta.i_type)

    loc_tar = FrameLocSet(**loc_meta.meta)
    loc_tar.init_load(id=ids, indf=indf, xyz=xyz, phot=phot)
    return loc_tar


def prepare_full_field_eval(
    *,
    param,
    calibration,
    phase_config_path: Path,
    tiff_path: Path,
    frames_per_sample: int,
    output_dir: Path,
    roi_size: int = 40,
    cache_dir: Path | None = None,
) -> FullFieldEvalData:
    cfg = load_pipeline_config(phase_config_path)
    cfg = copy.deepcopy(cfg)
    cfg.setdefault("phase_retrieval", {}).setdefault("input", {})["tiff_path"] = str(tiff_path)

    n_max = int(param.DataFilter.max_number_targets)
    num_emitters = _estimate_num_emitters(param, frames_per_sample)
    _apply_sim_overrides(cfg, param, calibration, frames_per_sample=frames_per_sample, n_max=n_max, num_emitters=num_emitters)

    frame_window = int(param.HyperParameter.channels_in)
    if cache_dir is not None:
        cached = _load_full_field_cache(
            cache_dir,
            cfg=cfg,
            tiff_path=tiff_path,
            param=param,
            calibration=calibration,
            frame_window=frame_window,
            roi_size=roi_size,
            frames_per_sample=frames_per_sample,
        )
        if cached is not None:
            return cached

    phase_result = run_phase_retrieval(cfg, mode="single")
    payload = phase_result.patches_payload
    if payload is None or payload.phase_maps is None:
        raise ValueError("Phase retrieval returned empty phase maps.")

    phase_maps = np.asarray(payload.phase_maps)
    full_h = int(phase_maps.shape[1])
    full_w = int(phase_maps.shape[2])

    rois, rows, cols = _roi_grid(full_h=full_h, full_w=full_w, roi=roi_size)

    loc_meta = train_setup.setup_loc_meta(param, calibration)
    frame_offset = -(frame_window - 1) // 2

    z_min_um = float(cfg.get("simulation", {}).get("emitter", {}).get("z_range", [-0.6, 0.6])[0])
    z_step_nm = float(param.Camera.cam_pxszs[2])
    z_max_index = float(param.Simulation.emitter_extent[2][1])

    samples: List[FullFieldSample] = []
    output_dir.mkdir(parents=True, exist_ok=True)

    for r, c, x0, y0 in rois:
        cfg_roi = copy.deepcopy(cfg)
        cfg_roi.setdefault("simulation", {}).setdefault("emitter", {})["field_extent"] = [[x0, x0 + roi_size], [y0, y0 + roi_size]]
        cfg_roi["simulation"].setdefault("simulation", {})["random_seed"] = int(r * 1000 + c)

        dataset = generate_dataset(cfg_roi, profile="eval", phase_payload=payload)
        if not dataset.samples:
            raise ValueError("Simulation dataset returned no samples for full-field eval.")

        sample = dataset.samples[0]
        _convert_z_um_to_index(
            sample.param_tar,
            z_min_um=z_min_um,
            z_step_nm=z_step_nm,
            z_max_index=z_max_index,
        )
        loc_tar = _locset_from_param(sample.param_tar, sample.mask_tar, loc_meta)
        loc_tar = loc_tar.realign_frame(frame_offset=frame_offset)

        frames = torch.from_numpy(sample.frames).float()
        samples.append(
            FullFieldSample(
                row=r,
                col=c,
                x0=x0,
                y0=y0,
                frames=frames,
                loc_tar=loc_tar,
                param_tar=sample.param_tar,
                mask_tar=sample.mask_tar,
            )
        )

    meta = _build_cache_meta(
        cfg=cfg,
        tiff_path=tiff_path,
        param=param,
        rows=rows,
        cols=cols,
        roi_size=roi_size,
        frames_per_sample=frames_per_sample,
        full_h=full_h,
        full_w=full_w,
    )
    if cache_dir is not None:
        _save_full_field_cache(cache_dir, meta, samples)

    return FullFieldEvalData(samples=samples, rows=rows, cols=cols, frame_window=frame_window, roi_size=roi_size)


def _infer_locset(
    *,
    model,
    frames: torch.Tensor,
    frame_proc: FrameRescalar,
    post_processor,
    device: torch.device,
    batch_size: int,
    frame_window: int,
) -> FrameLocSet:
    dataset = InferenceDataset(frames, frame_proc=frame_proc, frame_window=frame_window)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)

    locs: List[FrameLocSet] = []
    indf_offset = 0
    id_offset = 0

    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            y_out = model(batch)
            loc_out = post_processor.forward(y_out, indf_offset=indf_offset, id_offset=id_offset)
            if len(loc_out) > 0:
                locs.append(loc_out)
                id_offset += len(loc_out)
            indf_offset += y_out.size(0)

    if not locs:
        return FrameLocSet(**post_processor.loc_meta.meta)
    return FrameLocSet.loc_cat(locs)


def _heatmap_figsize(rows: int, cols: int, panels: int, cell_size: float = 0.3) -> Tuple[float, float]:
    height = max(2.5, rows * cell_size)
    width = max(2.5, cols * cell_size * panels)
    return width, height


def _expand_limits(limit, panels: int) -> List[Optional[float]]:
    if isinstance(limit, (list, tuple)):
        if len(limit) != panels:
            raise ValueError("vmin/vmax list length must match number of panels.")
        return list(limit)
    return [limit] * panels


def _plot_heatmaps(
    values: List[np.ndarray],
    titles: List[str],
    out_path: Path,
    *,
    vmin=None,
    vmax=None,
) -> None:
    if len(values) != len(titles):
        raise ValueError("values and titles must have the same length.")
    shapes = {val.shape for val in values}
    if len(shapes) != 1:
        raise ValueError("All heatmaps must share the same shape.")

    panels = len(values)
    rows, cols = values[0].shape
    vmin_list = _expand_limits(vmin, panels)
    vmax_list = _expand_limits(vmax, panels)

    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad(color="#dddddd")

    fig, axes = plt.subplots(
        1,
        panels,
        figsize=_heatmap_figsize(rows, cols, panels),
        constrained_layout=True,
    )
    if panels == 1:
        axes = [axes]

    for idx, (ax, data, title) in enumerate(zip(axes, values, titles)):
        data = np.ma.masked_invalid(data)
        im = ax.imshow(
            data,
            origin="upper",
            aspect="equal",
            vmin=vmin_list[idx],
            vmax=vmax_list[idx],
            cmap=cmap,
        )
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_summary(rmse_lat: np.ndarray, rmse_ax: np.ndarray, out_path: Path) -> None:
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].hist(rmse_lat, bins=30, color="#1f77b4", alpha=0.8)
    ax[0].set_title("RMSE Lat Distribution")
    ax[1].hist(rmse_ax, bins=30, color="#ff7f0e", alpha=0.8)
    ax[1].set_title("RMSE Ax Distribution")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def evaluate_full_field(
    *,
    model,
    eval_data: FullFieldEvalData,
    param,
    calibration,
    device: torch.device,
    batch_size: int,
    output_dir: Path,
    epoch: int,
) -> Tuple[float, float]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    loc_meta = train_setup.setup_loc_meta(param, calibration)
    post_processor, matcher = train_setup.setup_postprocess(loc_meta, param)

    frame_proc = FrameRescalar(scale=param.Scaling.input_scale, offset=param.Scaling.input_offset)

    rmse_lat_map = np.full((eval_data.rows, eval_data.cols), np.nan, dtype=np.float32)
    rmse_ax_map = np.full((eval_data.rows, eval_data.cols), np.nan, dtype=np.float32)
    prec_map = np.full((eval_data.rows, eval_data.cols), np.nan, dtype=np.float32)
    rec_map = np.full((eval_data.rows, eval_data.cols), np.nan, dtype=np.float32)
    f1_map = np.full((eval_data.rows, eval_data.cols), np.nan, dtype=np.float32)
    jac_map = np.full((eval_data.rows, eval_data.cols), np.nan, dtype=np.float32)

    for sample in tqdm(eval_data.samples, desc="Full-field eval", leave=False):
        loc_out = _infer_locset(
            model=model,
            frames=sample.frames,
            frame_proc=frame_proc,
            post_processor=post_processor,
            device=device,
            batch_size=batch_size,
            frame_window=eval_data.frame_window,
        )
        _, eval_set = train_kernels.eval_forward(
            cam_pxszs=param.Camera.cam_pxszs,
            matcher=matcher,
            loc_out=loc_out,
            loc_tar=sample.loc_tar,
        )
        rmse_lat = eval_set.rmse_lat
        rmse_ax = eval_set.rmse_ax
        rmse_lat_map[sample.row, sample.col] = rmse_lat if np.isfinite(rmse_lat) else np.nan
        rmse_ax_map[sample.row, sample.col] = rmse_ax if np.isfinite(rmse_ax) else np.nan
        prec_map[sample.row, sample.col] = eval_set.prec
        rec_map[sample.row, sample.col] = eval_set.rec
        f1_map[sample.row, sample.col] = eval_set.f1
        jac_map[sample.row, sample.col] = eval_set.jac

    rmse_lat_vals = rmse_lat_map[np.isfinite(rmse_lat_map)]
    rmse_ax_vals = rmse_ax_map[np.isfinite(rmse_ax_map)]
    mean_lat = float(np.mean(rmse_lat_vals)) if rmse_lat_vals.size else float("inf")
    mean_ax = float(np.mean(rmse_ax_vals)) if rmse_ax_vals.size else float("inf")

    epoch_tag = f"epoch_{epoch:04d}"
    _plot_heatmaps(
        [rmse_lat_map, rmse_ax_map],
        [f"RMSE Lat ({epoch_tag})", f"RMSE Ax ({epoch_tag})"],
        output_dir / f"{epoch_tag}_rmse.png",
    )
    _plot_heatmaps(
        [prec_map, rec_map],
        [f"Precision ({epoch_tag})", f"Recall ({epoch_tag})"],
        output_dir / f"{epoch_tag}_prec_recall.png",
        vmin=0.0,
        vmax=1.0,
    )
    _plot_heatmaps(
        [f1_map, jac_map],
        [f"F1 Score ({epoch_tag})", f"Jaccard ({epoch_tag})"],
        output_dir / f"{epoch_tag}_f1_jaccard.png",
        vmin=0.0,
        vmax=1.0,
    )
    if rmse_lat_vals.size and rmse_ax_vals.size:
        _plot_summary(rmse_lat_vals, rmse_ax_vals, output_dir / f"{epoch_tag}_rmse_summary.png")

    return mean_lat, mean_ax
