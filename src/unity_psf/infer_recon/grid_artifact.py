from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np


@dataclass(frozen=True)
class GridArtifactResult:
    grid_index: float
    frequency_axis_cycles_per_pixel: np.ndarray
    frc_curve: np.ndarray
    camera_pixel_frequency_index: int
    localization_count: int


_MUTABLE_XY_FIELDS = {
    "x_px",
    "y_px",
    "x_px_full",
    "y_px_full",
    "x_nm",
    "y_nm",
    "x_nm_full",
    "y_nm_full",
    "x_offset_px",
    "y_offset_px",
    "x_offset_nm",
    "y_offset_nm",
}


def compute_liteloc_grid_artifact_index(
    *,
    frame: np.ndarray,
    x_px: np.ndarray,
    y_px: np.ndarray,
    field_width_px: int,
    field_height_px: int,
    super_res_factor: int = 10,
    split_blocks: int = 10,
) -> GridArtifactResult:
    frame = np.asarray(frame)
    x_px = np.asarray(x_px, dtype=np.float64)
    y_px = np.asarray(y_px, dtype=np.float64)
    if not (frame.ndim == x_px.ndim == y_px.ndim == 1):
        raise ValueError("frame and coordinates must be one-dimensional")
    if not (frame.shape == x_px.shape == y_px.shape):
        raise ValueError("frame and coordinates must have identical shapes")
    if int(field_width_px) <= 0 or int(field_height_px) <= 0:
        raise ValueError("field dimensions must be positive")
    if int(super_res_factor) <= 1:
        raise ValueError("super_res_factor must be greater than one")
    if int(split_blocks) <= 1 or frame.size < int(split_blocks):
        raise ValueError("localization count must be at least split_blocks")

    order = np.argsort(frame)
    x_sorted = x_px[order]
    y_sorted = y_px[order]
    first_half = np.zeros(frame.size, dtype=bool)
    block_size = frame.size // int(split_blocks)
    for block_index in range(int(split_blocks)):
        start = block_index * block_size
        stop = (block_index + 1) * block_size
        first_half[start:stop] = bool(block_index % 2)

    image_1 = _render_super_resolution_histogram(
        x_px=x_sorted[first_half],
        y_px=y_sorted[first_half],
        field_width_px=int(field_width_px),
        field_height_px=int(field_height_px),
        super_res_factor=int(super_res_factor),
    )
    fft_1 = np.fft.fftshift(np.fft.fft2(image_1))
    del image_1
    image_2 = _render_super_resolution_histogram(
        x_px=x_sorted[~first_half],
        y_px=y_sorted[~first_half],
        field_width_px=int(field_width_px),
        field_height_px=int(field_height_px),
        super_res_factor=int(super_res_factor),
    )
    fft_2 = np.fft.fftshift(np.fft.fft2(image_2))
    del image_2

    numerator = _radial_sum_chunks(fft_1, fft_2, mode="cross")
    power_1 = _radial_sum_chunks(fft_1, None, mode="power")
    power_2 = _radial_sum_chunks(fft_2, None, mode="power")
    denominator = np.sqrt(power_1 * power_2)
    frc = np.divide(numerator, denominator, out=np.full_like(numerator, np.nan), where=denominator > 0)
    frequency = np.linspace(0.0, float(super_res_factor) / 2.0, frc.size)
    camera_index = int(np.abs(frequency - 1.0).argmin())
    lower = max(camera_index - 1, 0)
    upper = min(camera_index + 2, frc.size)
    grid_index = float(np.nanmax(frc[lower:upper]))
    return GridArtifactResult(
        grid_index=grid_index,
        frequency_axis_cycles_per_pixel=frequency,
        frc_curve=frc,
        camera_pixel_frequency_index=camera_index,
        localization_count=int(frame.size),
    )


def audit_raw_degrid_predictions(
    *,
    raw_predictions: Path,
    degrid_predictions: Path,
    field_width_px: int,
    field_height_px: int,
    output_json: Path | None = None,
    output_png: Path | None = None,
) -> dict[str, object]:
    raw = _read_prediction_arrays(Path(raw_predictions))
    degrid = _read_prediction_arrays(Path(degrid_predictions))
    if raw["x_px"].size != degrid["x_px"].size:
        raise ValueError("raw/degrid localization counts differ")
    _assert_invariant_fields(raw_predictions=Path(raw_predictions), degrid_predictions=Path(degrid_predictions))

    raw_result = compute_liteloc_grid_artifact_index(
        frame=raw["frame"],
        x_px=raw["x_px"],
        y_px=raw["y_px"],
        field_width_px=int(field_width_px),
        field_height_px=int(field_height_px),
    )
    degrid_result = compute_liteloc_grid_artifact_index(
        frame=degrid["frame"],
        x_px=degrid["x_px"],
        y_px=degrid["y_px"],
        field_width_px=int(field_width_px),
        field_height_px=int(field_height_px),
    )
    raw_x_offset = _coordinate_offsets(raw["x_px"], raw.get("x_offset_px"))
    raw_y_offset = _coordinate_offsets(raw["y_px"], raw.get("y_offset_px"))
    degrid_x_offset = _coordinate_offsets(degrid["x_px"], degrid.get("x_offset_px"))
    degrid_y_offset = _coordinate_offsets(degrid["y_px"], degrid.get("y_offset_px"))
    payload = {
        "contract": "liteloc_camera_pixel_grid_artifact_audit_v1",
        "raw_predictions": str(raw_predictions),
        "degrid_predictions": str(degrid_predictions),
        "field_width_px": int(field_width_px),
        "field_height_px": int(field_height_px),
        "super_res_factor": 10,
        "split_blocks": 10,
        "count_parity": True,
        "invariant_field_parity": True,
        "acceptance_rule": "degrid_grid_artifact_index < raw_grid_artifact_index",
        "accepted": bool(degrid_result.grid_index < raw_result.grid_index),
        "raw": _result_payload(raw_result, raw_x_offset, raw_y_offset),
        "degrid": _result_payload(degrid_result, degrid_x_offset, degrid_y_offset),
    }
    if output_json is not None:
        output_json = Path(output_json)
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    if output_png is not None:
        _write_audit_plot(
            output_png=Path(output_png),
            raw_result=raw_result,
            degrid_result=degrid_result,
            raw_offsets=(raw_x_offset, raw_y_offset),
            degrid_offsets=(degrid_x_offset, degrid_y_offset),
        )
    return payload


def _render_super_resolution_histogram(
    *,
    x_px: np.ndarray,
    y_px: np.ndarray,
    field_width_px: int,
    field_height_px: int,
    super_res_factor: int,
) -> np.ndarray:
    width = int(field_width_px) * int(super_res_factor)
    height = int(field_height_px) * int(super_res_factor)
    x_index = np.trunc(x_px * int(super_res_factor)).astype(np.int64)
    y_index = np.trunc(y_px * int(super_res_factor)).astype(np.int64)
    keep = (x_index >= 0) & (x_index < width) & (y_index >= 0) & (y_index < height)
    image = np.zeros((height, width), dtype=np.float64)
    np.add.at(image, (y_index[keep], x_index[keep]), 1.0)
    return image


def _radial_sum_chunks(
    fft_1: np.ndarray,
    fft_2: np.ndarray | None,
    *,
    mode: str,
    chunk_rows: int = 128,
) -> np.ndarray:
    height, width = fft_1.shape
    center_y = (height + 1) // 2
    center_x = (width + 1) // 2
    radial_length = int(np.ceil(height / 2) + 1)
    columns = np.arange(width, dtype=np.float64)[None, :]
    output = np.zeros(radial_length, dtype=np.float64)
    for start in range(0, height, int(chunk_rows)):
        stop = min(start + int(chunk_rows), height)
        rows = np.arange(start, stop, dtype=np.float64)[:, None]
        radius = np.rint(np.hypot(rows - center_y, columns - center_x)).astype(np.int32)
        if mode == "cross":
            if fft_2 is None:
                raise ValueError("cross radial sum requires two FFT arrays")
            weights = np.real(fft_1[start:stop] * np.conjugate(fft_2[start:stop]))
        elif mode == "power":
            weights = np.abs(fft_1[start:stop]) ** 2
        else:
            raise ValueError(f"unsupported radial sum mode: {mode}")
        keep = radius < radial_length
        output += np.bincount(radius[keep], weights=weights[keep], minlength=radial_length)[:radial_length]
    return output


def _read_prediction_arrays(path: Path) -> dict[str, np.ndarray]:
    with h5py.File(path, "r") as handle:
        group = handle["locs"]
        missing = [key for key in ("frame", "x_px", "y_px") if key not in group]
        if missing:
            raise KeyError(f"predictions H5 is missing required columns: {missing}")
        keys = ["frame", "x_px", "y_px", "x_offset_px", "y_offset_px"]
        return {key: np.asarray(group[key][:]) for key in keys if key in group}


def _assert_invariant_fields(*, raw_predictions: Path, degrid_predictions: Path) -> None:
    with h5py.File(raw_predictions, "r") as raw_handle, h5py.File(degrid_predictions, "r") as degrid_handle:
        raw_group = raw_handle["locs"]
        degrid_group = degrid_handle["locs"]
        if set(raw_group.keys()) != set(degrid_group.keys()):
            raise ValueError("raw/degrid H5 columns differ")
        for key in sorted(set(raw_group.keys()) - _MUTABLE_XY_FIELDS):
            raw_values = np.asarray(raw_group[key][:])
            degrid_values = np.asarray(degrid_group[key][:])
            if not np.array_equal(raw_values, degrid_values, equal_nan=True):
                raise ValueError(f"non-XY field differs: {key}")


def _coordinate_offsets(coordinates: np.ndarray, stored_offsets: np.ndarray | None) -> np.ndarray:
    if stored_offsets is not None:
        return np.asarray(stored_offsets, dtype=np.float64)
    coordinates = np.asarray(coordinates, dtype=np.float64)
    return coordinates - np.floor(coordinates) - 0.5


def _histogram_cv(offsets: np.ndarray) -> float:
    counts = np.histogram(offsets, bins=np.linspace(-0.5, 0.5, 101))[0]
    mean = float(np.mean(counts))
    return float(np.std(counts) / mean) if mean > 0 else float("nan")


def _result_payload(result: GridArtifactResult, x_offset: np.ndarray, y_offset: np.ndarray) -> dict[str, object]:
    index = result.camera_pixel_frequency_index
    return {
        "localization_count": int(result.localization_count),
        "grid_artifact_index": float(result.grid_index),
        "camera_pixel_frequency_cycles_per_pixel": float(result.frequency_axis_cycles_per_pixel[index]),
        "offset_uniformity_cv": {
            "x": _histogram_cv(x_offset),
            "y": _histogram_cv(y_offset),
        },
    }


def _write_audit_plot(
    *,
    output_png: Path,
    raw_result: GridArtifactResult,
    degrid_result: GridArtifactResult,
    raw_offsets: tuple[np.ndarray, np.ndarray],
    degrid_offsets: tuple[np.ndarray, np.ndarray],
) -> None:
    import matplotlib.pyplot as plt

    output_png.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    for axis, raw_values, degrid_values, label in (
        (axes[0, 0], raw_offsets[0], degrid_offsets[0], "x offset (camera px)"),
        (axes[0, 1], raw_offsets[1], degrid_offsets[1], "y offset (camera px)"),
    ):
        bins = np.linspace(-0.5, 0.5, 101)
        axis.hist(raw_values, bins=bins, density=True, histtype="step", label="raw")
        axis.hist(degrid_values, bins=bins, density=True, histtype="step", label="degrid")
        axis.set_xlabel(label)
        axis.set_ylabel("density")
        axis.legend()
    axes[1, 0].plot(raw_result.frequency_axis_cycles_per_pixel, raw_result.frc_curve, label="raw")
    axes[1, 0].plot(degrid_result.frequency_axis_cycles_per_pixel, degrid_result.frc_curve, label="degrid")
    axes[1, 0].axvline(1.0, color="black", linestyle="--", linewidth=1)
    axes[1, 0].set_xlim(0.5, 1.5)
    axes[1, 0].set_xlabel("cycles / camera pixel")
    axes[1, 0].set_ylabel("FRC")
    axes[1, 0].legend()
    axes[1, 1].bar(
        ["raw", "degrid"],
        [raw_result.grid_index, degrid_result.grid_index],
        color=["#777777", "#2A6F97"],
    )
    axes[1, 1].set_ylabel("LiteLoc-style grid artifact index")
    figure.savefig(output_png, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit raw versus LUNAR-degrid camera-pixel artifacts")
    parser.add_argument("--raw-predictions", type=Path, required=True)
    parser.add_argument("--degrid-predictions", type=Path, required=True)
    parser.add_argument("--field-width-px", type=int, required=True)
    parser.add_argument("--field-height-px", type=int, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-png", type=Path, required=True)
    args = parser.parse_args()
    payload = audit_raw_degrid_predictions(
        raw_predictions=args.raw_predictions,
        degrid_predictions=args.degrid_predictions,
        field_width_px=args.field_width_px,
        field_height_px=args.field_height_px,
        output_json=args.output_json,
        output_png=args.output_png,
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
