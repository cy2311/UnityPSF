from __future__ import annotations

import argparse
import errno
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np


@dataclass(frozen=True)
class LunarOffsetRescaleResult:
    x_offset_px: np.ndarray
    y_offset_px: np.ndarray
    changed: np.ndarray
    processed_bins: int
    bin_records: tuple[dict[str, object], ...]


def default_reconstruction_predictions(*, raw: Path, degrid: Path, degrid_enabled: bool) -> Path:
    selected = Path(degrid) if bool(degrid_enabled) else Path(raw)
    if not selected.is_file():
        raise FileNotFoundError(errno.ENOENT, "reconstruction predictions not found", str(selected))
    return selected


def lunar_histogram_equalization(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return values.copy()
    clipped = np.clip(values, -0.99, 0.99)
    counts = np.histogram(clipped, bins=np.linspace(-1, 1, 201))[0]
    cdf = np.cumsum(counts, dtype=np.float64) / float(np.sum(counts))
    indices = (clipped + 1.0) / 2.0 * 200.0 - 1.0
    lower = np.floor(indices).astype(np.int64)
    fraction = indices - lower
    return fraction * cdf[lower + 1] + (1.0 - fraction) * cdf[lower] - 0.5


def lunar_rescale_offsets(
    *,
    x_offset_px: np.ndarray,
    y_offset_px: np.ndarray,
    x_sig_nm: np.ndarray,
    y_sig_nm: np.ndarray,
    pixel_size_nm_x: float,
    pixel_size_nm_y: float,
    rescale_bins: int = 20,
    threshold: float = 0.01,
    min_bin_count: int = 32,
) -> LunarOffsetRescaleResult:
    x_offset_px = np.asarray(x_offset_px, dtype=np.float64)
    y_offset_px = np.asarray(y_offset_px, dtype=np.float64)
    x_sig_nm = np.asarray(x_sig_nm, dtype=np.float64)
    y_sig_nm = np.asarray(y_sig_nm, dtype=np.float64)
    if not (x_offset_px.shape == y_offset_px.shape == x_sig_nm.shape == y_sig_nm.shape):
        raise ValueError("offset and uncertainty arrays must have identical shapes")
    if int(rescale_bins) <= 0:
        raise ValueError("rescale_bins must be positive")
    if float(pixel_size_nm_x) <= 0 or float(pixel_size_nm_y) <= 0:
        raise ValueError("pixel sizes must be positive")

    finite_sig = np.isfinite(x_sig_nm) & np.isfinite(y_sig_nm)
    x_var = float(np.var(x_sig_nm[finite_sig])) if np.any(finite_sig) else 0.0
    y_var = float(np.var(y_sig_nm[finite_sig])) if np.any(finite_sig) else 0.0
    y_scale = np.sqrt(x_var / y_var) if x_var > 0 and y_var > 0 else 1.0
    total_sig = np.sqrt(x_sig_nm**2 + (y_scale * y_sig_nm) ** 2)
    eligible = finite_sig & np.isfinite(x_offset_px) & np.isfinite(y_offset_px) & (total_sig != 0)

    x_rescaled = x_offset_px.copy()
    y_rescaled = y_offset_px.copy()
    records: list[dict[str, object]] = []
    processed_bins = 0
    finite_total = np.sort(total_sig[eligible])
    if finite_total.size > 0:
        quantile_positions = np.linspace(0, finite_total.size, int(rescale_bins) + 1)
        bin_edges = np.interp(quantile_positions, np.arange(finite_total.size), finite_total)
        physical_threshold = float(threshold) * np.sqrt(float(pixel_size_nm_x) ** 2 + float(pixel_size_nm_y) ** 2)
        for bin_index in range(int(rescale_bins)):
            lower = float(bin_edges[bin_index])
            upper = float(bin_edges[bin_index + 1])
            indices = np.where(eligible & (total_sig > lower) & (total_sig < upper))[0]
            status = "processed"
            if lower < physical_threshold:
                status = "below_uncertainty_threshold"
            elif indices.size < int(min_bin_count):
                status = "skipped_insufficient_records"
            else:
                x_rescaled[indices] = lunar_histogram_equalization(x_offset_px[indices]) + np.mean(x_offset_px[indices])
                y_rescaled[indices] = lunar_histogram_equalization(y_offset_px[indices]) + np.mean(y_offset_px[indices])
                processed_bins += 1
            records.append(
                {
                    "bin_index": int(bin_index),
                    "lower_total_sig_nm": lower,
                    "upper_total_sig_nm": upper,
                    "count": int(indices.size),
                    "status": status,
                }
            )

    changed = (np.abs(x_rescaled - x_offset_px) > 1e-12) | (np.abs(y_rescaled - y_offset_px) > 1e-12)
    return LunarOffsetRescaleResult(
        x_offset_px=x_rescaled,
        y_offset_px=y_rescaled,
        changed=changed,
        processed_bins=int(processed_bins),
        bin_records=tuple(records),
    )


def spatial_lunar_rescale_offsets(
    *,
    x_px: np.ndarray,
    y_px: np.ndarray,
    x_offset_px: np.ndarray,
    y_offset_px: np.ndarray,
    x_sig_nm: np.ndarray,
    y_sig_nm: np.ndarray,
    pixel_size_nm_x: float,
    pixel_size_nm_y: float,
    field_width_px: float,
    field_height_px: float,
    spatial_bins_x: int,
    spatial_bins_y: int,
    rescale_bins: int = 20,
    threshold: float = 0.01,
    min_bin_count: int = 32,
) -> LunarOffsetRescaleResult:
    x_px = np.asarray(x_px, dtype=np.float64)
    y_px = np.asarray(y_px, dtype=np.float64)
    x_offset_px = np.asarray(x_offset_px, dtype=np.float64)
    y_offset_px = np.asarray(y_offset_px, dtype=np.float64)
    x_sig_nm = np.asarray(x_sig_nm, dtype=np.float64)
    y_sig_nm = np.asarray(y_sig_nm, dtype=np.float64)
    if not (x_px.shape == y_px.shape == x_offset_px.shape == y_offset_px.shape == x_sig_nm.shape == y_sig_nm.shape):
        raise ValueError("coordinates, offsets, and uncertainty arrays must have identical shapes")
    if int(spatial_bins_x) <= 0 or int(spatial_bins_y) <= 0:
        raise ValueError("spatial bin counts must be positive")
    if float(field_width_px) <= 0 or float(field_height_px) <= 0:
        raise ValueError("field dimensions must be positive")

    x_bin = np.clip(
        np.floor(x_px / float(field_width_px) * int(spatial_bins_x)).astype(np.int64),
        0,
        int(spatial_bins_x) - 1,
    )
    y_bin = np.clip(
        np.floor(y_px / float(field_height_px) * int(spatial_bins_y)).astype(np.int64),
        0,
        int(spatial_bins_y) - 1,
    )
    x_rescaled = x_offset_px.copy()
    y_rescaled = y_offset_px.copy()
    records: list[dict[str, object]] = []
    processed_bins = 0
    for spatial_y in range(int(spatial_bins_y)):
        for spatial_x in range(int(spatial_bins_x)):
            indices = np.where((x_bin == spatial_x) & (y_bin == spatial_y))[0]
            if indices.size < int(min_bin_count):
                records.append(
                    {
                        "spatial_x": int(spatial_x),
                        "spatial_y": int(spatial_y),
                        "count": int(indices.size),
                        "status": "skipped_insufficient_records",
                    }
                )
                continue
            local = lunar_rescale_offsets(
                x_offset_px=x_offset_px[indices],
                y_offset_px=y_offset_px[indices],
                x_sig_nm=x_sig_nm[indices],
                y_sig_nm=y_sig_nm[indices],
                pixel_size_nm_x=float(pixel_size_nm_x),
                pixel_size_nm_y=float(pixel_size_nm_y),
                rescale_bins=int(rescale_bins),
                threshold=float(threshold),
                min_bin_count=int(min_bin_count),
            )
            x_rescaled[indices] = local.x_offset_px
            y_rescaled[indices] = local.y_offset_px
            processed_bins += int(local.processed_bins)
            records.append(
                {
                    "spatial_x": int(spatial_x),
                    "spatial_y": int(spatial_y),
                    "count": int(indices.size),
                    "status": "processed" if local.processed_bins else "skipped",
                    "processed_uncertainty_bins": int(local.processed_bins),
                }
            )

    changed = (np.abs(x_rescaled - x_offset_px) > 1e-12) | (np.abs(y_rescaled - y_offset_px) > 1e-12)
    return LunarOffsetRescaleResult(
        x_offset_px=x_rescaled,
        y_offset_px=y_rescaled,
        changed=changed,
        processed_bins=int(processed_bins),
        bin_records=tuple(records),
    )


def degrid_predictions_h5(
    *,
    predictions: Path,
    output: Path,
    summary_json: Path,
    pixel_size_nm_x: float,
    pixel_size_nm_y: float,
    rescale_bins: int = 20,
    threshold: float = 0.01,
    min_bin_count: int = 32,
    histogram_png: Path | None,
    spatial_bins_x: int = 1,
    spatial_bins_y: int = 1,
    field_width_px: float | None = None,
    field_height_px: float | None = None,
) -> dict[str, object]:
    predictions = Path(predictions)
    output = Path(output)
    summary_json = Path(summary_json)
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    summary_json.parent.mkdir(parents=True, exist_ok=True)

    shutil.copy2(predictions, output)
    with h5py.File(output, "r+") as handle:
        group = handle["locs"]
        required = ["x_px", "y_px"]
        missing = [key for key in required if key not in group]
        if missing:
            raise KeyError(f"predictions H5 is missing required degrid columns: {missing}")
        x_px = np.asarray(group["x_px"][:], dtype=np.float64)
        y_px = np.asarray(group["y_px"][:], dtype=np.float64)
        x_offset_px = _read_or_derive_offset(group, "x_offset_px", x_px)
        y_offset_px = _read_or_derive_offset(group, "y_offset_px", y_px)
        x_sig_nm = _read_uncertainty_nm(group, axis="x", pixel_size_nm=float(pixel_size_nm_x))
        y_sig_nm = _read_uncertainty_nm(group, axis="y", pixel_size_nm=float(pixel_size_nm_y))

        spatial_enabled = int(spatial_bins_x) > 1 or int(spatial_bins_y) > 1
        if spatial_enabled:
            if field_width_px is None or field_height_px is None:
                raise ValueError("spatial degrid requires field dimensions")
            result = spatial_lunar_rescale_offsets(
                x_px=x_px,
                y_px=y_px,
                x_offset_px=x_offset_px,
                y_offset_px=y_offset_px,
                x_sig_nm=x_sig_nm,
                y_sig_nm=y_sig_nm,
                pixel_size_nm_x=float(pixel_size_nm_x),
                pixel_size_nm_y=float(pixel_size_nm_y),
                field_width_px=float(field_width_px),
                field_height_px=float(field_height_px),
                spatial_bins_x=int(spatial_bins_x),
                spatial_bins_y=int(spatial_bins_y),
                rescale_bins=int(rescale_bins),
                threshold=float(threshold),
                min_bin_count=int(min_bin_count),
            )
        else:
            result = lunar_rescale_offsets(
                x_offset_px=x_offset_px,
                y_offset_px=y_offset_px,
                x_sig_nm=x_sig_nm,
                y_sig_nm=y_sig_nm,
                pixel_size_nm_x=float(pixel_size_nm_x),
                pixel_size_nm_y=float(pixel_size_nm_y),
                rescale_bins=int(rescale_bins),
                threshold=float(threshold),
                min_bin_count=int(min_bin_count),
            )
        delta_x_px = result.x_offset_px - x_offset_px
        delta_y_px = result.y_offset_px - y_offset_px
        _replace(group, "x_px", x_px + delta_x_px)
        _replace(group, "y_px", y_px + delta_y_px)
        _shift_if_present(group, "x_px_full", delta_x_px)
        _shift_if_present(group, "y_px_full", delta_y_px)
        _replace_if_present(group, "x_nm", (x_px + delta_x_px) * float(pixel_size_nm_x))
        _replace_if_present(group, "y_nm", (y_px + delta_y_px) * float(pixel_size_nm_y))
        _shift_if_present(group, "x_nm_full", delta_x_px * float(pixel_size_nm_x))
        _shift_if_present(group, "y_nm_full", delta_y_px * float(pixel_size_nm_y))
        _replace_if_present(group, "x_offset_px", result.x_offset_px)
        _replace_if_present(group, "y_offset_px", result.y_offset_px)
        _replace_if_present(group, "x_offset_nm", result.x_offset_px * float(pixel_size_nm_x))
        _replace_if_present(group, "y_offset_nm", result.y_offset_px * float(pixel_size_nm_y))

        handle.attrs["derived_kind"] = "lunar_spatial_offset_degrid" if spatial_enabled else "lunar_offset_degrid"
        handle.attrs["source_predictions"] = str(predictions)
        handle.attrs["degrid_contract"] = (
            "lunar_spatial_rescale_offset_v1" if spatial_enabled else "lunar_rescale_offset_v1"
        )
        handle.attrs["degrid_rescale_bins"] = int(rescale_bins)
        handle.attrs["degrid_threshold"] = float(threshold)
        handle.attrs["degrid_min_bin_count"] = int(min_bin_count)
        handle.attrs["degrid_spatial_bins_x"] = int(spatial_bins_x)
        handle.attrs["degrid_spatial_bins_y"] = int(spatial_bins_y)
        handle.attrs["count"] = int(x_px.size)

    shift_x_nm = delta_x_px * float(pixel_size_nm_x)
    shift_y_nm = delta_y_px * float(pixel_size_nm_y)
    radial_shift_nm = np.hypot(shift_x_nm, shift_y_nm)
    histogram_edges = np.linspace(-0.5, 0.5, 101)
    x_hist_before = np.histogram(x_offset_px, bins=histogram_edges)[0]
    x_hist_after = np.histogram(result.x_offset_px, bins=histogram_edges)[0]
    y_hist_before = np.histogram(y_offset_px, bins=histogram_edges)[0]
    y_hist_after = np.histogram(result.y_offset_px, bins=histogram_edges)[0]
    payload = {
        "contract": "lunar_spatial_rescale_offset_v1" if spatial_enabled else "lunar_rescale_offset_v1",
        "source_predictions": str(predictions),
        "output_predictions": str(output),
        "total_rows": int(x_px.size),
        "changed_rows": int(np.count_nonzero(result.changed)),
        "processed_bins": int(result.processed_bins),
        "rescale_bins": int(rescale_bins),
        "threshold": float(threshold),
        "min_bin_count": int(min_bin_count),
        "spatial_bins_x": int(spatial_bins_x),
        "spatial_bins_y": int(spatial_bins_y),
        "pixel_size_nm_x": float(pixel_size_nm_x),
        "pixel_size_nm_y": float(pixel_size_nm_y),
        "shift_nm": _shift_summary(radial_shift_nm),
        "x_shift_nm": _shift_summary(np.abs(shift_x_nm)),
        "y_shift_nm": _shift_summary(np.abs(shift_y_nm)),
        "offset_uniformity_cv": {
            "x_before": _histogram_cv(x_hist_before),
            "x_after": _histogram_cv(x_hist_after),
            "y_before": _histogram_cv(y_hist_before),
            "y_after": _histogram_cv(y_hist_after),
        },
        "offset_histogram": {
            "edges_px": histogram_edges.tolist(),
            "x_before": x_hist_before.astype(int).tolist(),
            "x_after": x_hist_after.astype(int).tolist(),
            "y_before": y_hist_before.astype(int).tolist(),
            "y_after": y_hist_after.astype(int).tolist(),
        },
        "bins": list(result.bin_records),
    }
    summary_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    if histogram_png is not None:
        _write_histogram_png(
            histogram_png=Path(histogram_png),
            x_before=x_offset_px,
            x_after=result.x_offset_px,
            y_before=y_offset_px,
            y_after=result.y_offset_px,
        )
    return payload


def _read_or_derive_offset(group: h5py.Group, key: str, coordinates_px: np.ndarray) -> np.ndarray:
    if key in group:
        return np.asarray(group[key][:], dtype=np.float64)
    return coordinates_px - np.floor(coordinates_px) - 0.5


def _read_uncertainty_nm(group: h5py.Group, *, axis: str, pixel_size_nm: float) -> np.ndarray:
    nm_key = f"{axis}_sig_nm"
    px_key = f"{axis}_sig_px"
    legacy_key = f"{axis}_sig"
    if nm_key in group:
        return np.asarray(group[nm_key][:], dtype=np.float64)
    for key in (px_key, legacy_key):
        if key in group:
            return np.asarray(group[key][:], dtype=np.float64) * float(pixel_size_nm)
    raise KeyError(f"predictions H5 is missing {axis} uncertainty")


def _replace(group: h5py.Group, key: str, values: np.ndarray) -> None:
    group[key][:] = np.asarray(values, dtype=group[key].dtype)


def _replace_if_present(group: h5py.Group, key: str, values: np.ndarray) -> None:
    if key in group:
        _replace(group, key, values)


def _shift_if_present(group: h5py.Group, key: str, delta: np.ndarray) -> None:
    if key in group:
        _replace(group, key, np.asarray(group[key][:], dtype=np.float64) + delta)


def _shift_summary(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {"mean": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "mean": float(np.mean(values)),
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


def _histogram_cv(counts: np.ndarray) -> float:
    counts = np.asarray(counts, dtype=np.float64)
    mean = float(np.mean(counts)) if counts.size else 0.0
    return float(np.std(counts) / mean) if mean > 0 else 0.0


def _write_histogram_png(
    *,
    histogram_png: Path,
    x_before: np.ndarray,
    x_after: np.ndarray,
    y_before: np.ndarray,
    y_after: np.ndarray,
) -> None:
    import matplotlib.pyplot as plt

    histogram_png.parent.mkdir(parents=True, exist_ok=True)
    bins = np.linspace(-0.5, 0.5, 101)
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    for axis, values, title in (
        (axes[0, 0], x_before, "x offset before"),
        (axes[0, 1], x_after, "x offset after"),
        (axes[1, 0], y_before, "y offset before"),
        (axes[1, 1], y_after, "y offset after"),
    ):
        axis.hist(values, bins=bins, color="#2f6f8f")
        axis.set_xlim(-0.5, 0.5)
        axis.set_title(title)
        axis.set_xlabel("sub-pixel offset (px)")
        axis.set_ylabel("count")
    fig.savefig(histogram_png, dpi=180)
    plt.close(fig)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply LUNAR-style xy offset degrid to Neptune predictions H5.")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--histogram-png", type=Path, default=None)
    parser.add_argument("--pixel-size-nm-x", type=float, required=True)
    parser.add_argument("--pixel-size-nm-y", type=float, required=True)
    parser.add_argument("--rescale-bins", type=int, default=20)
    parser.add_argument("--threshold", type=float, default=0.01)
    parser.add_argument("--min-bin-count", type=int, default=32)
    parser.add_argument("--spatial-bins-x", type=int, default=1)
    parser.add_argument("--spatial-bins-y", type=int, default=1)
    parser.add_argument("--field-width-px", type=float, default=None)
    parser.add_argument("--field-height-px", type=float, default=None)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    payload = degrid_predictions_h5(
        predictions=args.predictions,
        output=args.output,
        summary_json=args.summary_json,
        histogram_png=args.histogram_png,
        pixel_size_nm_x=float(args.pixel_size_nm_x),
        pixel_size_nm_y=float(args.pixel_size_nm_y),
        rescale_bins=int(args.rescale_bins),
        threshold=float(args.threshold),
        min_bin_count=int(args.min_bin_count),
        spatial_bins_x=int(args.spatial_bins_x),
        spatial_bins_y=int(args.spatial_bins_y),
        field_width_px=args.field_width_px,
        field_height_px=args.field_height_px,
    )
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
