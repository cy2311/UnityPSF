from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
from scipy.optimize import minimize
import tifffile

from neptune_v03.infer_recon.standard import camera_pixels_from_runtime, read_json, write_json


@dataclass(frozen=True)
class FilterConfig:
    prob_min: float | None = None
    frame_min: int | None = None
    frame_max: int | None = None
    locprec_xy_nm_min: float | None = None
    locprec_xy_nm_max: float | None = None
    photon_min: float | None = None
    photon_max: float | None = None
    x_sig_px_max: float | None = None
    y_sig_px_max: float | None = None
    llrel_min: float | None = None
    psf_xy_nm_max: float | None = None
    require_fit_status: bool = False


QUALITY_METRIC_KEYS = (
    "fit_status",
    "x_fit_px",
    "y_fit_px",
    "background",
    "logLikelihood",
    "log_likelihood",
    "negative_log_likelihood",
    "LLrel",
    "llrel",
    "PSFxpix",
    "PSFypix",
    "PSFxnm",
    "PSFynm",
    "psf_x_nm",
    "psf_y_nm",
    "psf_xy_nm",
)


def _optional_float(value: object) -> float | None:
    if value in {None, ""}:
        return None
    out = float(value)
    return out if math.isfinite(out) else None


def _optional_int(value: object) -> int | None:
    if value in {None, ""}:
        return None
    return int(value)


def _first_optional_float(row: dict[str, object], keys: Iterable[str]) -> float | None:
    for key in keys:
        value = _optional_float(row.get(key))
        if value is not None:
            return value
    return None


def compute_locprec_xy_nm(
    *,
    x_sig_px: float | None,
    y_sig_px: float | None,
    camera_pixel_nm_x: float,
    camera_pixel_nm_y: float,
) -> float | None:
    if x_sig_px is None or y_sig_px is None:
        return None
    x_nm = abs(float(x_sig_px)) * float(camera_pixel_nm_x)
    y_nm = abs(float(y_sig_px)) * float(camera_pixel_nm_y)
    return math.sqrt((x_nm * x_nm + y_nm * y_nm) / 2.0)


def _psf_xy_nm_from_row(row: dict[str, object]) -> float | None:
    direct = _first_optional_float(row, ("psf_xy_nm", "PSFxy_nm", "PSFxynm"))
    if direct is not None:
        return direct
    psf_x = _first_optional_float(row, ("psf_x_nm", "PSFxnm"))
    psf_y = _first_optional_float(row, ("psf_y_nm", "PSFynm"))
    if psf_x is None:
        return None
    if psf_y is None:
        psf_y = psf_x
    return math.sqrt((psf_x * psf_x + psf_y * psf_y) / 2.0)


def _photon_from_row(row: dict[str, object]) -> float | None:
    return _first_optional_float(row, ("photon", "photons", "N", "phot"))


def _fit_status_from_row(row: dict[str, object]) -> float | None:
    return _first_optional_float(row, ("fit_status", "postfit_status", "converged", "locprecnm_converged"))


def summarize_quality_fields(rows: Iterable[dict[str, object]]) -> dict[str, int]:
    work = list(rows)
    locprec_rows = 0
    llrel_rows = 0
    psf_rows = 0
    for row in work:
        if _optional_float(row.get("x_sig")) is not None and _optional_float(row.get("y_sig")) is not None:
            locprec_rows += 1
        if _first_optional_float(row, ("llrel", "LLrel")) is not None:
            llrel_rows += 1
        if _psf_xy_nm_from_row(row) is not None:
            psf_rows += 1
    return {
        "quality_total_rows": len(work),
        "quality_locprec_xy_nm_rows": locprec_rows,
        "quality_llrel_rows": llrel_rows,
        "quality_psf_xy_nm_rows": psf_rows,
    }


def _poisson_nll(model: np.ndarray, observed: np.ndarray) -> float:
    model = np.clip(model.astype(np.float64, copy=False), 1e-6, None)
    observed = np.clip(observed.astype(np.float64, copy=False), 0.0, None)
    return float(np.sum(model - observed * np.log(model)))


def _poisson_deviance(model: np.ndarray, observed: np.ndarray) -> float:
    model = np.clip(model.astype(np.float64, copy=False), 1e-6, None)
    observed = np.clip(observed.astype(np.float64, copy=False), 0.0, None)
    positive = observed > 0
    out = np.zeros_like(model, dtype=np.float64)
    out[positive] = (model[positive] - observed[positive]) - observed[positive] * np.log(
        model[positive] / observed[positive]
    )
    out[~positive] = model[~positive]
    return float(2.0 * np.sum(out))


def compute_smap_llrel(
    *,
    log_likelihood: float,
    roi_size_px: int,
    em_factor: float = 1.0,
    num_channels: int = 1,
) -> float:
    roi_size = max(int(roi_size_px), 1)
    channels = max(int(num_channels), 1)
    return float(log_likelihood) * float(em_factor) / float(roi_size * roi_size * channels)


def _fit_gaussian_roi(roi: np.ndarray) -> dict[str, float]:
    roi_f = np.clip(np.asarray(roi, dtype=np.float64), 0.0, None)
    height, width = roi_f.shape
    if height == 0 or width == 0:
        return {"fit_status": 0.0}

    border_mask = np.ones_like(roi_f, dtype=bool)
    if height > 2 and width > 2:
        border_mask[1:-1, 1:-1] = False
    border = roi_f[border_mask]
    background0 = float(np.median(border if border.size else roi_f))
    signal = np.clip(roi_f - background0, 0.0, None)
    signal_sum = float(signal.sum())
    if signal_sum <= 1e-6:
        return {
            "fit_status": 0.0,
            "background": background0,
            "photons": 0.0,
            "sigma_x_px": float("nan"),
            "sigma_y_px": float("nan"),
            "poisson_deviance": float("inf"),
            "negative_log_likelihood": float("inf"),
            "log_likelihood": float("-inf"),
        }

    yy, xx = np.mgrid[0:height, 0:width].astype(np.float64)
    cx0 = float((signal * xx).sum() / max(signal_sum, 1e-6))
    cy0 = float((signal * yy).sum() / max(signal_sum, 1e-6))
    var_x0 = float((signal * (xx - cx0) ** 2).sum() / max(signal_sum, 1e-6))
    var_y0 = float((signal * (yy - cy0) ** 2).sum() / max(signal_sum, 1e-6))
    sigma_x0 = math.sqrt(max(var_x0, 0.7**2))
    sigma_y0 = math.sqrt(max(var_y0, 0.7**2))
    photons0 = max(signal_sum, 1.0)

    max_sigma = max(float(max(height, width)), 1.0)
    bounds = [
        (0.0, float(width - 1)),
        (0.0, float(height - 1)),
        (1e-3, max(float(roi_f.sum()) * 4.0, 1.0)),
        (1e-6, max(float(roi_f.max()) * 2.0, background0 * 4.0 + 1.0)),
        (0.45, max_sigma),
        (0.45, max_sigma),
    ]
    x0 = np.asarray(
        [
            min(max(cx0, bounds[0][0]), bounds[0][1]),
            min(max(cy0, bounds[1][0]), bounds[1][1]),
            min(max(photons0, bounds[2][0]), bounds[2][1]),
            min(max(background0, bounds[3][0]), bounds[3][1]),
            min(max(sigma_x0, bounds[4][0]), bounds[4][1]),
            min(max(sigma_y0, bounds[5][0]), bounds[5][1]),
        ],
        dtype=np.float64,
    )

    def model_from_params(params: np.ndarray) -> np.ndarray:
        cx, cy, photons, background, sigma_x, sigma_y = params
        gauss = np.exp(-0.5 * (((xx - cx) / sigma_x) ** 2 + ((yy - cy) / sigma_y) ** 2))
        gauss_sum = max(float(gauss.sum()), 1e-12)
        return background + photons * gauss / gauss_sum

    def objective(params: np.ndarray) -> float:
        return _poisson_nll(model_from_params(params), roi_f)

    result = minimize(objective, x0, method="L-BFGS-B", bounds=bounds, options={"maxiter": 80, "ftol": 1e-8})
    params = np.asarray(result.x if result.x is not None else x0, dtype=np.float64)
    model = model_from_params(params)
    deviance = _poisson_deviance(model, roi_f)
    log_likelihood = -0.5 * deviance
    return {
        "fit_status": 1.0 if bool(result.success) else 0.0,
        "x_fit_local_px": float(params[0]),
        "y_fit_local_px": float(params[1]),
        "photons": float(params[2]),
        "background": float(params[3]),
        "sigma_x_px": float(params[4]),
        "sigma_y_px": float(params[5]),
        "poisson_deviance": float(deviance),
        "negative_log_likelihood": float(-log_likelihood),
        "log_likelihood": float(log_likelihood),
    }


def _moment_gaussian_roi(roi: np.ndarray) -> dict[str, float]:
    roi_f = np.clip(np.asarray(roi, dtype=np.float64), 0.0, None)
    height, width = roi_f.shape
    if height == 0 or width == 0:
        return {"fit_status": 0.0}

    border_mask = np.ones_like(roi_f, dtype=bool)
    if height > 2 and width > 2:
        border_mask[1:-1, 1:-1] = False
    border = roi_f[border_mask]
    background = float(np.median(border if border.size else roi_f))
    signal = np.clip(roi_f - background, 0.0, None)
    signal_sum = float(signal.sum())
    if signal_sum <= 1e-6:
        return {
            "fit_status": 0.0,
            "background": background,
            "photons": 0.0,
            "sigma_x_px": float("nan"),
            "sigma_y_px": float("nan"),
            "poisson_deviance": float("inf"),
            "negative_log_likelihood": float("inf"),
            "log_likelihood": float("-inf"),
        }

    yy, xx = np.mgrid[0:height, 0:width].astype(np.float64)
    cx = float((signal * xx).sum() / signal_sum)
    cy = float((signal * yy).sum() / signal_sum)
    sigma_x = math.sqrt(max(float((signal * (xx - cx) ** 2).sum() / signal_sum), 0.45**2))
    sigma_y = math.sqrt(max(float((signal * (yy - cy) ** 2).sum() / signal_sum), 0.45**2))
    gauss = np.exp(-0.5 * (((xx - cx) / sigma_x) ** 2 + ((yy - cy) / sigma_y) ** 2))
    model = background + signal_sum * gauss / max(float(gauss.sum()), 1e-12)
    deviance = _poisson_deviance(model, roi_f)
    log_likelihood = -0.5 * deviance
    return {
        "fit_status": 1.0,
        "x_fit_local_px": float(cx),
        "y_fit_local_px": float(cy),
        "photons": float(signal_sum),
        "background": float(background),
        "sigma_x_px": float(sigma_x),
        "sigma_y_px": float(sigma_y),
        "poisson_deviance": float(deviance),
        "negative_log_likelihood": float(-log_likelihood),
        "log_likelihood": float(log_likelihood),
    }


def estimate_gaussian_roi_metrics(
    frame: np.ndarray,
    *,
    x_px: float,
    y_px: float,
    camera_pixel_nm_x: float,
    camera_pixel_nm_y: float,
    roi_radius_px: int = 3,
    em_factor: float = 1.0,
    num_channels: int = 1,
) -> dict[str, float]:
    if frame.ndim != 2:
        raise ValueError(f"Expected 2D frame, got shape={frame.shape}")
    radius = max(int(roi_radius_px), 1)
    roi_size_px = 2 * radius + 1
    center_x = int(round(float(x_px)))
    center_y = int(round(float(y_px)))
    x0 = max(center_x - radius, 0)
    y0 = max(center_y - radius, 0)
    x1 = min(center_x + radius + 1, int(frame.shape[1]))
    y1 = min(center_y + radius + 1, int(frame.shape[0]))
    roi = np.asarray(frame[y0:y1, x0:x1], dtype=np.float32)
    if roi.size == 0:
        return {
            "fit_status": 0.0,
            "llrel": float("-inf"),
            "log_likelihood": float("-inf"),
            "negative_log_likelihood": float("inf"),
            "poisson_deviance": float("inf"),
            "psf_x_nm": float("nan"),
            "psf_y_nm": float("nan"),
            "psf_xy_nm": float("nan"),
        }

    fit = _fit_gaussian_roi(roi)
    sigma_x_px = fit.get("sigma_x_px", float("nan"))
    sigma_y_px = fit.get("sigma_y_px", float("nan"))
    sigma_x_nm = float(sigma_x_px) * float(camera_pixel_nm_x)
    sigma_y_nm = float(sigma_y_px) * float(camera_pixel_nm_y)
    psf_xy_nm = math.sqrt((sigma_x_nm * sigma_x_nm + sigma_y_nm * sigma_y_nm) / 2.0) if math.isfinite(sigma_x_nm + sigma_y_nm) else float("nan")
    log_likelihood = float(fit.get("log_likelihood", float("-inf")))
    llrel = compute_smap_llrel(
        log_likelihood=log_likelihood,
        roi_size_px=roi_size_px,
        em_factor=float(em_factor),
        num_channels=int(num_channels),
    )
    return {
        **fit,
        "x_fit_px": float(x0) + float(fit.get("x_fit_local_px", float("nan"))),
        "y_fit_px": float(y0) + float(fit.get("y_fit_local_px", float("nan"))),
        "logLikelihood": float(log_likelihood),
        "LLrel": float(llrel),
        "llrel": float(llrel),
        "PSFxpix": float(sigma_x_px),
        "PSFypix": float(sigma_y_px),
        "PSFxnm": float(sigma_x_nm),
        "PSFynm": float(sigma_y_nm),
        "psf_x_nm": float(sigma_x_nm),
        "psf_y_nm": float(sigma_y_nm),
        "psf_xy_nm": float(psf_xy_nm),
    }


def estimate_gaussian_roi_metrics_moment(
    frame: np.ndarray,
    *,
    x_px: float,
    y_px: float,
    camera_pixel_nm_x: float,
    camera_pixel_nm_y: float,
    roi_radius_px: int = 3,
    em_factor: float = 1.0,
    num_channels: int = 1,
) -> dict[str, float]:
    if frame.ndim != 2:
        raise ValueError(f"Expected 2D frame, got shape={frame.shape}")
    radius = max(int(roi_radius_px), 1)
    roi_size_px = 2 * radius + 1
    center_x = int(round(float(x_px)))
    center_y = int(round(float(y_px)))
    x0 = max(center_x - radius, 0)
    y0 = max(center_y - radius, 0)
    x1 = min(center_x + radius + 1, int(frame.shape[1]))
    y1 = min(center_y + radius + 1, int(frame.shape[0]))
    roi = np.asarray(frame[y0:y1, x0:x1], dtype=np.float32)
    if roi.size == 0:
        return {
            "fit_status": 0.0,
            "llrel": float("-inf"),
            "log_likelihood": float("-inf"),
            "negative_log_likelihood": float("inf"),
            "poisson_deviance": float("inf"),
            "psf_x_nm": float("nan"),
            "psf_y_nm": float("nan"),
            "psf_xy_nm": float("nan"),
        }

    fit = _moment_gaussian_roi(roi)
    sigma_x_px = fit.get("sigma_x_px", float("nan"))
    sigma_y_px = fit.get("sigma_y_px", float("nan"))
    sigma_x_nm = float(sigma_x_px) * float(camera_pixel_nm_x)
    sigma_y_nm = float(sigma_y_px) * float(camera_pixel_nm_y)
    psf_xy_nm = math.sqrt((sigma_x_nm * sigma_x_nm + sigma_y_nm * sigma_y_nm) / 2.0) if math.isfinite(sigma_x_nm + sigma_y_nm) else float("nan")
    log_likelihood = float(fit.get("log_likelihood", float("-inf")))
    llrel = compute_smap_llrel(
        log_likelihood=log_likelihood,
        roi_size_px=roi_size_px,
        em_factor=float(em_factor),
        num_channels=int(num_channels),
    )
    return {
        **fit,
        "x_fit_px": float(x0) + float(fit.get("x_fit_local_px", float("nan"))),
        "y_fit_px": float(y0) + float(fit.get("y_fit_local_px", float("nan"))),
        "logLikelihood": float(log_likelihood),
        "LLrel": float(llrel),
        "llrel": float(llrel),
        "PSFxpix": float(sigma_x_px),
        "PSFypix": float(sigma_y_px),
        "PSFxnm": float(sigma_x_nm),
        "PSFynm": float(sigma_y_nm),
        "psf_x_nm": float(sigma_x_nm),
        "psf_y_nm": float(sigma_y_nm),
        "psf_xy_nm": float(psf_xy_nm),
    }


def enrich_rows_with_quality_metrics(
    rows: Iterable[dict[str, object]],
    *,
    frame_loader: Callable[[int], np.ndarray],
    camera_pixel_nm_x: float,
    camera_pixel_nm_y: float,
    roi_radius_px: int = 3,
    em_factor: float = 1.0,
    num_channels: int = 1,
) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    cached_frame_index: int | None = None
    cached_frame: np.ndarray | None = None
    for row in rows:
        row_out = dict(row)
        frame_index = _optional_int(row_out.get("frame"))
        x_px = _optional_float(row_out.get("x_px"))
        y_px = _optional_float(row_out.get("y_px"))
        if frame_index is None or x_px is None or y_px is None:
            out.append(row_out)
            continue
        if cached_frame_index != frame_index or cached_frame is None:
            cached_frame = np.asarray(frame_loader(int(frame_index)), dtype=np.float32)
            cached_frame_index = int(frame_index)
        row_out.update(
            estimate_gaussian_roi_metrics(
                cached_frame,
                x_px=float(x_px),
                y_px=float(y_px),
                camera_pixel_nm_x=float(camera_pixel_nm_x),
                camera_pixel_nm_y=float(camera_pixel_nm_y),
                roi_radius_px=int(roi_radius_px),
                em_factor=float(em_factor),
                num_channels=int(num_channels),
            )
        )
        out.append(row_out)
    return out


def _with_locprec(
    rows: Iterable[dict[str, object]],
    *,
    camera_pixel_nm_x: float,
    camera_pixel_nm_y: float,
) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for row in rows:
        row_out = dict(row)
        locprec = compute_locprec_xy_nm(
            x_sig_px=_optional_float(row_out.get("x_sig")),
            y_sig_px=_optional_float(row_out.get("y_sig")),
            camera_pixel_nm_x=float(camera_pixel_nm_x),
            camera_pixel_nm_y=float(camera_pixel_nm_y),
        )
        row_out["locprec_xy_nm"] = locprec
        out.append(row_out)
    return out


def _filter_stage(
    rows: list[dict[str, object]],
    *,
    predicate,
) -> list[dict[str, object]]:
    return [row for row in rows if predicate(row)]


def filter_rows(
    rows: Iterable[dict[str, object]],
    config: FilterConfig,
    *,
    camera_pixel_nm_x: float,
    camera_pixel_nm_y: float,
) -> tuple[list[dict[str, object]], dict[str, int]]:
    work = _with_locprec(
        rows,
        camera_pixel_nm_x=float(camera_pixel_nm_x),
        camera_pixel_nm_y=float(camera_pixel_nm_y),
    )
    summary = {"total_in": len(work), **summarize_quality_fields(work)}

    if config.prob_min is not None:
        work = _filter_stage(work, predicate=lambda row: float(row["prob"]) >= float(config.prob_min))
    summary["after_prob"] = len(work)

    if config.frame_min is not None or config.frame_max is not None:
        def _frame_ok(row: dict[str, object]) -> bool:
            frame = int(row["frame"])
            if config.frame_min is not None and frame < int(config.frame_min):
                return False
            if config.frame_max is not None and frame > int(config.frame_max):
                return False
            return True

        work = _filter_stage(work, predicate=_frame_ok)
    summary["after_frame"] = len(work)
    quality_gate_base = list(work)

    if config.locprec_xy_nm_min is not None or config.locprec_xy_nm_max is not None:
        summary["missing_locprec_xy_nm_for_requested_gate"] = sum(row["locprec_xy_nm"] is None for row in quality_gate_base)
        work = _filter_stage(
            work,
            predicate=lambda row: (
                row["locprec_xy_nm"] is not None
                and (config.locprec_xy_nm_min is None or float(row["locprec_xy_nm"]) > float(config.locprec_xy_nm_min))
                and (config.locprec_xy_nm_max is None or float(row["locprec_xy_nm"]) <= float(config.locprec_xy_nm_max))
            ),
        )
    summary["after_locprec_xy_nm"] = len(work)

    if config.llrel_min is not None:
        summary["missing_llrel_for_requested_gate"] = sum(_first_optional_float(row, ("llrel", "LLrel")) is None for row in quality_gate_base)
        work = _filter_stage(
            work,
            predicate=lambda row: (
                (value := _first_optional_float(row, ("llrel", "LLrel"))) is not None and value >= float(config.llrel_min)
            ),
        )
    summary["after_llrel"] = len(work)

    if config.psf_xy_nm_max is not None:
        summary["missing_psf_xy_nm_for_requested_gate"] = sum(_psf_xy_nm_from_row(row) is None for row in quality_gate_base)
        work = _filter_stage(
            work,
            predicate=lambda row: (
                (value := _psf_xy_nm_from_row(row)) is not None and value <= float(config.psf_xy_nm_max)
            ),
        )
    summary["after_psf_xy_nm"] = len(work)

    if config.photon_min is not None or config.photon_max is not None:
        summary["missing_photon_for_requested_gate"] = sum(_photon_from_row(row) is None for row in quality_gate_base)
        work = _filter_stage(
            work,
            predicate=lambda row: (
                (value := _photon_from_row(row)) is not None
                and (config.photon_min is None or value >= float(config.photon_min))
                and (config.photon_max is None or value <= float(config.photon_max))
            ),
        )
        summary["after_photon"] = len(work)

    if config.x_sig_px_max is not None:
        summary["missing_x_sig_for_requested_gate"] = sum(_optional_float(row.get("x_sig")) is None for row in quality_gate_base)
        work = _filter_stage(
            work,
            predicate=lambda row: (
                (value := _optional_float(row.get("x_sig"))) is not None and abs(value) <= float(config.x_sig_px_max)
            ),
        )
        summary["after_x_sig_px"] = len(work)

    if config.y_sig_px_max is not None:
        summary["missing_y_sig_for_requested_gate"] = sum(_optional_float(row.get("y_sig")) is None for row in quality_gate_base)
        work = _filter_stage(
            work,
            predicate=lambda row: (
                (value := _optional_float(row.get("y_sig"))) is not None and abs(value) <= float(config.y_sig_px_max)
            ),
        )
        summary["after_y_sig_px"] = len(work)

    if config.require_fit_status:
        summary["missing_fit_status_for_requested_gate"] = sum(_fit_status_from_row(row) is None for row in quality_gate_base)
        work = _filter_stage(
            work,
            predicate=lambda row: (value := _fit_status_from_row(row)) is not None and value > 0,
        )
        summary["after_fit_status"] = len(work)

    summary["total_out"] = len(work)
    return work, summary


def _read_rows(path: Path) -> list[dict[str, object]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader]


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter standard infer localization CSV with SMAP-like quality gates.")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--sample-tiff", type=Path, default=None)
    parser.add_argument("--runtime-state", type=Path, default=None)
    parser.add_argument("--camera-pixel-nm-x", type=float, default=None)
    parser.add_argument("--camera-pixel-nm-y", type=float, default=None)
    parser.add_argument("--enrich-quality-metrics", action="store_true")
    parser.add_argument("--roi-radius-px", type=int, default=3)
    parser.add_argument("--em-factor", type=float, default=1.0)
    parser.add_argument("--num-channels", type=int, default=1)
    parser.add_argument("--prob-min", type=float, default=None)
    parser.add_argument("--frame-min", type=int, default=None)
    parser.add_argument("--frame-max", type=int, default=None)
    parser.add_argument("--locprec-xy-nm-min", type=float, default=None)
    parser.add_argument("--locprec-xy-nm-max", type=float, default=None)
    parser.add_argument("--photon-min", type=float, default=None)
    parser.add_argument("--photon-max", type=float, default=None)
    parser.add_argument("--x-sig-px-max", type=float, default=None)
    parser.add_argument("--y-sig-px-max", type=float, default=None)
    parser.add_argument("--llrel-min", type=float, default=None)
    parser.add_argument("--psf-xy-nm-max", type=float, default=None)
    parser.add_argument("--require-fit-status", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    camera_x = args.camera_pixel_nm_x
    camera_y = args.camera_pixel_nm_y
    if camera_x is None or camera_y is None:
        if args.runtime_state is None:
            raise ValueError("Provide either --runtime-state or both --camera-pixel-nm-x/--camera-pixel-nm-y")
        runtime = read_json(args.runtime_state)
        camera_x, camera_y = camera_pixels_from_runtime(runtime)
    rows = _read_rows(args.predictions)
    if args.enrich_quality_metrics:
        if args.sample_tiff is None:
            raise ValueError("--sample-tiff is required when --enrich-quality-metrics is enabled")
        with tifffile.TiffFile(args.sample_tiff) as tif:
            rows = enrich_rows_with_quality_metrics(
                rows,
                frame_loader=lambda frame_index: tif.series[0].asarray(key=int(frame_index)),
                camera_pixel_nm_x=float(camera_x),
                camera_pixel_nm_y=float(camera_y),
                roi_radius_px=int(args.roi_radius_px),
                em_factor=float(args.em_factor),
                num_channels=int(args.num_channels),
            )
    filtered, summary = filter_rows(
        rows,
        FilterConfig(
            prob_min=args.prob_min,
            frame_min=args.frame_min,
            frame_max=args.frame_max,
            locprec_xy_nm_min=args.locprec_xy_nm_min,
            locprec_xy_nm_max=args.locprec_xy_nm_max,
            photon_min=args.photon_min,
            photon_max=args.photon_max,
            x_sig_px_max=args.x_sig_px_max,
            y_sig_px_max=args.y_sig_px_max,
            llrel_min=args.llrel_min,
            psf_xy_nm_max=args.psf_xy_nm_max,
            require_fit_status=bool(args.require_fit_status),
        ),
        camera_pixel_nm_x=float(camera_x),
        camera_pixel_nm_y=float(camera_y),
    )
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    _write_rows(args.output_csv, filtered)
    write_json(
        args.summary_json,
        {
            "predictions": str(args.predictions),
            "output_csv": str(args.output_csv),
            "camera_pixel_nm_x": float(camera_x),
            "camera_pixel_nm_y": float(camera_y),
            "sample_tiff": None if args.sample_tiff is None else str(args.sample_tiff),
            "enrich_quality_metrics": bool(args.enrich_quality_metrics),
            "roi_radius_px": int(args.roi_radius_px),
            "em_factor": float(args.em_factor),
            "num_channels": int(args.num_channels),
            "filters": {
                "prob_min": args.prob_min,
                "frame_min": args.frame_min,
                "frame_max": args.frame_max,
                "locprec_xy_nm_min": args.locprec_xy_nm_min,
                "locprec_xy_nm_max": args.locprec_xy_nm_max,
                "photon_min": args.photon_min,
                "photon_max": args.photon_max,
                "x_sig_px_max": args.x_sig_px_max,
                "y_sig_px_max": args.y_sig_px_max,
                "llrel_min": args.llrel_min,
                "psf_xy_nm_max": args.psf_xy_nm_max,
                "require_fit_status": bool(args.require_fit_status),
            },
            **summary,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
