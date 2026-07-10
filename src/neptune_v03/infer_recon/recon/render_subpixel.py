#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
import tifffile
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter

from neptune_v03.infer_recon.predictions_io import read_render_arrays


DEFAULT_Z_MIN_NM = -600.0
DEFAULT_Z_MAX_NM = 600.0
SMAP_GUI_DEFAULT_IMAX_MIN = -3.5
NEPTUNE_DEFAULT_IMAX_MIN = -2.5228787452803374  # q = 0.997


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standard subpixel Gaussian reconstruction from localization CSV.")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--width-px", type=int, required=True)
    parser.add_argument("--height-px", type=int, required=True)
    parser.add_argument("--camera-pixel-nm-x", type=float, required=True)
    parser.add_argument("--camera-pixel-nm-y", type=float, required=True)
    parser.add_argument("--render-pixel-nm", type=float, default=20.0)
    parser.add_argument("--spot-radius-nm", type=float, default=28.0)
    parser.add_argument("--prob-threshold", type=float, default=0.70)
    parser.add_argument("--z-min", "--z-min-nm", dest="z_min_nm", type=float, default=DEFAULT_Z_MIN_NM)
    parser.add_argument("--z-max", "--z-max-nm", dest="z_max_nm", type=float, default=DEFAULT_Z_MAX_NM)
    parser.add_argument("--render-weight", choices=["count", "photon", "probability"], default="count")
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--brightness", type=float, default=1.0)
    parser.add_argument("--display-mode", choices=["quantile", "fixed_imax"], default="quantile")
    parser.add_argument("--display-imax", type=float, default=None)
    parser.add_argument("--display-imax-min", type=float, default=NEPTUNE_DEFAULT_IMAX_MIN)
    parser.add_argument(
        "--normalization-fov",
        type=str,
        default=None,
        metavar="X0,Y0,WIDTH,HEIGHT",
        help="Optional rendered-pixel region used only to estimate display Imax.",
    )
    parser.add_argument("--chunk-size", type=int, default=200000)
    parser.add_argument("--radius-mode", choices=["fixed", "xy_uncertainty_mean"], default="fixed")
    parser.add_argument("--uncertainty-cap-mode", choices=["fixed", "median10"], default="fixed")
    parser.add_argument("--uncertainty-scale", type=float, default=1.0)
    parser.add_argument("--uncertainty-min-sigma-px", type=float, default=0.75)
    parser.add_argument("--uncertainty-max-sigma-px", type=float, default=6.0)
    parser.add_argument("--uncertainty-bin-size-px", type=float, default=0.5)
    parser.add_argument("--suffix", type=str, default="subpixel_standard")
    return parser.parse_args()


def _iris_color(z: np.ndarray, z_min: float, z_max: float) -> np.ndarray:
    t = np.zeros_like(z, dtype=np.float32) if z_max <= z_min else (z.astype(np.float32) - float(z_min)) / (float(z_max) - float(z_min))
    t = np.clip(t, 0.0, 1.0)
    stops = np.asarray(
        [
            [74, 24, 132],
            [45, 75, 185],
            [32, 144, 190],
            [64, 176, 117],
            [238, 205, 64],
            [220, 82, 75],
        ],
        dtype=np.float32,
    ) / 255.0
    pos = t * (len(stops) - 1)
    idx = np.floor(pos).astype(np.int64)
    idx = np.clip(idx, 0, len(stops) - 2)
    frac = (pos - idx)[:, None]
    return stops[idx] * (1.0 - frac) + stops[idx + 1] * frac


def _draw_colorbar(path: Path, *, z_min: float, z_max: float, height: int = 512, width: int = 72) -> None:
    vals = np.linspace(float(z_max), float(z_min), height, dtype=np.float32)
    colors = np.clip(_iris_color(vals, z_min, z_max) * 255.0, 0, 255).astype(np.uint8)
    bar = np.repeat(colors[:, None, :], width, axis=1)
    image = Image.fromarray(bar, mode="RGB")
    canvas = Image.new("RGB", (width + 130, height + 30), "white")
    canvas.paste(image, (10, 10))
    draw = ImageDraw.Draw(canvas)
    draw.text((width + 20, 10), f"z {z_max:.2f}", fill=(0, 0, 0))
    draw.text((width + 20, height - 5), f"z {z_min:.2f}", fill=(0, 0, 0))
    draw.text((width + 20, height // 2), "nm", fill=(0, 0, 0))
    canvas.save(path)


def _resolve_render_weights(
    *,
    mode: str,
    count: int,
    probability: np.ndarray | None,
    photon: np.ndarray | None,
) -> np.ndarray:
    key = str(mode)
    if key == "count":
        return np.ones(int(count), dtype=np.float32)
    source = probability if key == "probability" else photon if key == "photon" else None
    if key not in {"probability", "photon"}:
        raise ValueError(f"Unsupported render_weight={mode!r}")
    if source is None:
        raise KeyError(f"predictions must contain {key} values when --render-weight={key}")
    if int(source.size) != int(count):
        raise ValueError(f"{key} size must match localization count")
    return np.clip(np.nan_to_num(source.astype(np.float32, copy=False), nan=0.0), 0.0, None)


def _smap_quantile_from_imax_min(imax_min: float) -> float:
    """Convert SMAP's negative Imax GUI value into a quantile."""
    value = float(imax_min)
    if value >= 0:
        raise ValueError("SMAP quantile Imax mode requires display_imax_min < 0")
    return float(1.0 - 10.0**value)


def _parse_normalization_fov(spec: str | None, *, width: int, height: int) -> tuple[int, int, int, int] | None:
    if spec is None:
        return None
    try:
        x0, y0, roi_width, roi_height = (int(part.strip()) for part in str(spec).split(","))
    except ValueError as exc:
        raise ValueError("normalization_fov must be X0,Y0,WIDTH,HEIGHT") from exc
    if x0 < 0 or y0 < 0 or roi_width <= 0 or roi_height <= 0 or x0 + roi_width > width or y0 + roi_height > height:
        raise ValueError(f"normalization_fov {spec!r} is outside rendered image {width}x{height}")
    return x0, y0, roi_width, roi_height


def _display_values(values: np.ndarray, *, imax: float, gamma: float, brightness: float) -> np.ndarray:
    image = np.clip(float(brightness) * values / max(float(imax), 1e-12), 0.0, 1.0)
    if float(gamma) > 0 and float(gamma) != 1.0:
        image = np.power(image, float(gamma))
    return image.astype(np.float32, copy=False)


def _colorize_density_display(
    density: np.ndarray,
    linear_rgb: np.ndarray,
    *,
    imax: float,
    gamma: float,
    brightness: float,
) -> np.ndarray:
    """Use linear density as value and accumulated RGB only as chromaticity."""
    if linear_rgb.ndim != 3 or linear_rgb.shape[:2] != density.shape or linear_rgb.shape[2] != 3:
        raise ValueError("linear_rgb must have shape (height, width, 3) matching density")
    peak = np.max(linear_rgb, axis=2, keepdims=True)
    chromaticity = np.divide(linear_rgb, peak, out=np.zeros_like(linear_rgb), where=peak > 0)
    value = _display_values(density, imax=imax, gamma=gamma, brightness=brightness)
    return (chromaticity * value[:, :, None]).astype(np.float32, copy=False)


def _normalize_display(
    density: np.ndarray,
    *,
    mode: str,
    fixed_imax: float | None,
    imax_min: float,
    gamma: float,
    brightness: float,
    normalize_roi: tuple[int, int, int, int] | None,
) -> tuple[np.ndarray, float, str]:
    """Produce a display-only normalized image without modifying linear density."""
    reference = density
    if normalize_roi is not None:
        x0, y0, width, height = normalize_roi
        reference = density[y0 : y0 + height, x0 : x0 + width]
    finite = reference[np.isfinite(reference)]
    if str(mode) == "fixed_imax":
        if fixed_imax is None or float(fixed_imax) <= 0:
            raise ValueError("display_mode=fixed_imax requires --display-imax > 0")
        imax = float(fixed_imax)
        source = "fixed_imax"
    elif str(mode) == "quantile":
        quantile = _smap_quantile_from_imax_min(float(imax_min))
        imax = float(np.quantile(finite, quantile)) if finite.size else 0.0
        if not np.isfinite(imax) or imax <= 0:
            imax = float(np.max(finite)) if finite.size else 1.0
        if not np.isfinite(imax) or imax <= 0:
            imax = 1.0
        source = "smap_quantile"
    else:
        raise ValueError(f"Unsupported display_mode={mode!r}")
    return _display_values(density, imax=imax, gamma=gamma, brightness=brightness), imax, source


def _add_bilinear(canvas: np.ndarray, x: np.ndarray, y: np.ndarray, colors: np.ndarray) -> None:
    height, width, _ = canvas.shape
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    fx = (x - x0).astype(np.float32, copy=False)
    fy = (y - y0).astype(np.float32, copy=False)
    for dx, wx in ((0, 1.0 - fx), (1, fx)):
        ix = x0 + dx
        for dy, wy in ((0, 1.0 - fy), (1, fy)):
            iy = y0 + dy
            keep = (ix >= 0) & (ix < width) & (iy >= 0) & (iy < height)
            if not np.any(keep):
                continue
            weight = (wx[keep] * wy[keep]).astype(np.float32, copy=False)
            for channel in range(canvas.shape[2]):
                np.add.at(canvas[:, :, channel], (iy[keep], ix[keep]), colors[keep, channel] * weight)


def _fixed_sigma_render_px(*, spot_radius_nm: float, render_pixel_nm: float) -> float:
    return max((float(spot_radius_nm) / float(render_pixel_nm)) / 2.0, 0.7)


def _resolve_sigma_render_px(
    *,
    radius_mode: str,
    count: int,
    spot_radius_nm: float,
    render_pixel_nm: float,
    x_sig_px: np.ndarray | None,
    y_sig_px: np.ndarray | None,
    camera_pixel_nm_x: float,
    camera_pixel_nm_y: float,
    uncertainty_scale: float,
    uncertainty_min_sigma_px: float,
    uncertainty_max_sigma_px: float,
    uncertainty_cap_mode: str = "fixed",
) -> np.ndarray:
    if int(count) == 0:
        return np.asarray([], dtype=np.float32)
    if str(radius_mode) == "fixed":
        return np.full(
            int(count),
            _fixed_sigma_render_px(spot_radius_nm=float(spot_radius_nm), render_pixel_nm=float(render_pixel_nm)),
            dtype=np.float32,
        )
    if str(radius_mode) != "xy_uncertainty_mean":
        raise ValueError(f"Unsupported radius_mode={radius_mode!r}")
    if x_sig_px is None or y_sig_px is None:
        raise KeyError("predictions must contain x_sig and y_sig when --radius-mode=xy_uncertainty_mean")
    if x_sig_px.shape != y_sig_px.shape:
        raise ValueError("x_sig and y_sig arrays must have the same shape")
    if x_sig_px.size != int(count):
        raise ValueError("x_sig/y_sig size must match localization count")
    sig_x_render = np.abs(x_sig_px.astype(np.float32, copy=False)) * (float(camera_pixel_nm_x) / float(render_pixel_nm))
    sig_y_render = np.abs(y_sig_px.astype(np.float32, copy=False)) * (float(camera_pixel_nm_y) / float(render_pixel_nm))
    sigma = np.sqrt((sig_x_render * sig_x_render + sig_y_render * sig_y_render) / 2.0).astype(np.float32, copy=False)
    sigma *= float(uncertainty_scale)
    sigma = np.nan_to_num(
        sigma,
        nan=float(uncertainty_min_sigma_px),
        posinf=float(uncertainty_max_sigma_px),
        neginf=float(uncertainty_min_sigma_px),
    )
    max_sigma = float(uncertainty_max_sigma_px)
    if str(uncertainty_cap_mode) == "median10":
        finite = sigma[np.isfinite(sigma) & (sigma > 0)]
        if finite.size:
            max_sigma = min(float(np.median(finite)) * 10.0, 200.0)
    elif str(uncertainty_cap_mode) != "fixed":
        raise ValueError(f"Unsupported uncertainty_cap_mode={uncertainty_cap_mode!r}")
    sigma = np.clip(sigma, float(uncertainty_min_sigma_px), float(max_sigma))
    return sigma.astype(np.float32, copy=False)


def _render_grouped_gaussians(
    *,
    width: int,
    height: int,
    x_render: np.ndarray,
    y_render: np.ndarray,
    colors: np.ndarray,
    sigmas_px: np.ndarray,
    chunk_size: int,
    bin_size_px: float,
) -> np.ndarray:
    canvas = np.zeros((height, width, colors.shape[1]), dtype=np.float32)
    if x_render.size == 0:
        return canvas
    if np.allclose(sigmas_px, sigmas_px[0]):
        for start in range(0, x_render.size, int(chunk_size)):
            stop = min(x_render.size, start + int(chunk_size))
            _add_bilinear(canvas, x_render[start:stop], y_render[start:stop], colors[start:stop])
        sigma = float(sigmas_px[0])
        for channel in range(canvas.shape[2]):
            canvas[:, :, channel] = gaussian_filter(canvas[:, :, channel], sigma=sigma, mode="constant")
        return canvas

    if float(bin_size_px) <= 0:
        raise ValueError("uncertainty_bin_size_px must be > 0")
    sigma_bins = np.round(sigmas_px.astype(np.float32, copy=False) / float(bin_size_px)) * float(bin_size_px)
    sigma_bins = np.clip(sigma_bins, float(sigmas_px.min()), float(sigmas_px.max()))
    out = np.zeros_like(canvas)
    for sigma in np.unique(sigma_bins):
        keep = np.flatnonzero(np.isclose(sigma_bins, sigma))
        if keep.size == 0:
            continue
        tmp = np.zeros_like(canvas)
        for start in range(0, keep.size, int(chunk_size)):
            ix = keep[start : start + int(chunk_size)]
            _add_bilinear(tmp, x_render[ix], y_render[ix], colors[ix])
        for channel in range(out.shape[2]):
            out[:, :, channel] += gaussian_filter(tmp[:, :, channel], sigma=float(sigma), mode="constant")
    return out


def render(args: argparse.Namespace) -> dict[str, object]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    x_px, y_px, z_nm, prob, photon, x_sig_px, y_sig_px = read_render_arrays(args.predictions, float(args.prob_threshold))
    width = max(1, int(np.ceil(args.width_px * args.camera_pixel_nm_x / args.render_pixel_nm)))
    height = max(1, int(np.ceil(args.height_px * args.camera_pixel_nm_y / args.render_pixel_nm)))
    x_render = x_px * float(args.camera_pixel_nm_x) / float(args.render_pixel_nm)
    y_render = y_px * float(args.camera_pixel_nm_y) / float(args.render_pixel_nm)
    render_weights = _resolve_render_weights(
        mode=str(args.render_weight),
        count=int(x_render.size),
        probability=prob,
        photon=photon,
    )
    rgb_colors = _iris_color(z_nm, float(args.z_min_nm), float(args.z_max_nm)) * render_weights[:, None]
    # Channel zero is the quantitative, z-independent localization density.
    colors = np.concatenate((render_weights[:, None], rgb_colors), axis=1)
    sigmas_px = _resolve_sigma_render_px(
        radius_mode=str(args.radius_mode),
        count=int(x_render.size),
        spot_radius_nm=float(args.spot_radius_nm),
        render_pixel_nm=float(args.render_pixel_nm),
        x_sig_px=x_sig_px,
        y_sig_px=y_sig_px,
        camera_pixel_nm_x=float(args.camera_pixel_nm_x),
        camera_pixel_nm_y=float(args.camera_pixel_nm_y),
        uncertainty_scale=float(args.uncertainty_scale),
        uncertainty_min_sigma_px=float(args.uncertainty_min_sigma_px),
        uncertainty_max_sigma_px=float(args.uncertainty_max_sigma_px),
        uncertainty_cap_mode=str(getattr(args, "uncertainty_cap_mode", "fixed")),
    )
    rendered = _render_grouped_gaussians(
        width=width,
        height=height,
        x_render=x_render,
        y_render=y_render,
        colors=colors,
        sigmas_px=sigmas_px,
        chunk_size=int(args.chunk_size),
        bin_size_px=float(args.uncertainty_bin_size_px),
    )
    density = rendered[:, :, 0]
    linear_rgb = rendered[:, :, 1:]
    normalization_fov = _parse_normalization_fov(str(args.normalization_fov) if args.normalization_fov else None, width=width, height=height)
    _, imax, imax_source = _normalize_display(
        density,
        mode=str(args.display_mode),
        fixed_imax=args.display_imax,
        imax_min=float(args.display_imax_min),
        gamma=float(args.gamma),
        brightness=float(args.brightness),
        normalize_roi=normalization_fov,
    )
    image = _colorize_density_display(
        density,
        linear_rgb,
        imax=imax,
        gamma=float(args.gamma),
        brightness=float(args.brightness),
    )
    rgb = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    png_path = args.output_dir / f"reconstruction_{args.suffix}.png"
    tiff_path = args.output_dir / f"reconstruction_{args.suffix}.tiff"
    linear_density_path = args.output_dir / f"reconstruction_{args.suffix}_density_linear_float32.tiff"
    linear_rgb_path = args.output_dir / f"reconstruction_{args.suffix}_rgb_linear_float32.tiff"
    colorbar_path = args.output_dir / f"reconstruction_{args.suffix}_z_colorbar.png"
    Image.fromarray(rgb, mode="RGB").save(png_path)
    tifffile.imwrite(tiff_path, rgb, photometric="rgb")
    tifffile.imwrite(linear_density_path, density.astype(np.float32, copy=False))
    tifffile.imwrite(linear_rgb_path, linear_rgb.astype(np.float32, copy=False), photometric="rgb")
    _draw_colorbar(colorbar_path, z_min=float(args.z_min_nm), z_max=float(args.z_max_nm))
    summary = {
        "renderer": "infer_recon_subpixel_gaussian_v2_smap_display_contract",
        "predictions": str(args.predictions),
        "rendered_localizations": int(x_render.size),
        "render_weight": str(args.render_weight),
        "prob_threshold": float(args.prob_threshold),
        "render_pixel_nm": float(args.render_pixel_nm),
        "spot_radius_nm": float(args.spot_radius_nm),
        "radius_mode": str(args.radius_mode),
        "uncertainty_cap_mode": str(getattr(args, "uncertainty_cap_mode", "fixed")),
        "uncertainty_scale": float(args.uncertainty_scale),
        "uncertainty_min_sigma_px": float(args.uncertainty_min_sigma_px),
        "uncertainty_max_sigma_px": float(args.uncertainty_max_sigma_px),
        "uncertainty_bin_size_px": float(args.uncertainty_bin_size_px),
        "sigma_render_px_mean": float(sigmas_px.mean()) if sigmas_px.size else 0.0,
        "sigma_render_px_min": float(sigmas_px.min()) if sigmas_px.size else 0.0,
        "sigma_render_px_max": float(sigmas_px.max()) if sigmas_px.size else 0.0,
        "z_unit": "nm",
        "z_min_nm": float(args.z_min_nm),
        "z_max_nm": float(args.z_max_nm),
        "width": int(width),
        "height": int(height),
        "gamma": float(args.gamma),
        "brightness": float(args.brightness),
        "display_mode": str(args.display_mode),
        "display_imax": float(imax),
        "display_imax_source": imax_source,
        "display_imax_min": float(args.display_imax_min),
        "display_quantile": _smap_quantile_from_imax_min(float(args.display_imax_min)) if str(args.display_mode) == "quantile" else None,
        "normalization_fov_rendered_px": list(normalization_fov) if normalization_fov is not None else None,
        "linear_density_path": str(linear_density_path),
        "linear_rgb_path": str(linear_rgb_path),
        "png_path": str(png_path),
        "tiff_path": str(tiff_path),
        "colorbar_path": str(colorbar_path),
    }
    (args.output_dir / f"reconstruction_{args.suffix}_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    print(json.dumps(render(_parse_args()), indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
