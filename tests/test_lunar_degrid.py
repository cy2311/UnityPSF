import h5py
import numpy as np

from neptune_v03.infer_recon.degrid import (
    default_reconstruction_predictions,
    degrid_predictions_h5,
    lunar_histogram_equalization,
    lunar_rescale_offsets,
    spatial_lunar_rescale_offsets,
)
from neptune_v03.infer_recon.predictions_io import H5PredictionWriter


def _lunar_reference_histogram_equalization(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -0.99, 0.99)
    counts = np.histogram(clipped, bins=np.linspace(-1, 1, 201))[0]
    cdf = np.cumsum(counts) / np.sum(counts)
    indices = (clipped + 1) / 2 * 200 - 1
    fractions = indices - np.floor(indices)
    lower = np.floor(indices).astype(int)
    return fractions * cdf[lower + 1] + (1 - fractions) * cdf[lower] - 0.5


def test_histogram_equalization_matches_lunar_reference() -> None:
    values = np.asarray([-1.2, -0.8, -0.25, -0.05, 0.0, 0.1, 0.1, 0.4, 0.95, 1.2], dtype=np.float64)

    actual = lunar_histogram_equalization(values)
    expected = _lunar_reference_histogram_equalization(values)

    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)


def test_default_reconstruction_uses_degrid_without_silent_fallback(tmp_path) -> None:
    raw = tmp_path / "predictions_merged.h5"
    degrid = tmp_path / "predictions_degrid.h5"
    raw.touch()

    assert default_reconstruction_predictions(raw=raw, degrid=degrid, degrid_enabled=False) == raw
    try:
        default_reconstruction_predictions(raw=raw, degrid=degrid, degrid_enabled=True)
    except FileNotFoundError as exc:
        assert exc.filename == str(degrid)
    else:
        raise AssertionError("enabled degrid must not silently fall back to raw predictions")

    degrid.touch()
    assert default_reconstruction_predictions(raw=raw, degrid=degrid, degrid_enabled=True) == degrid


def test_rescale_preserves_low_uncertainty_and_skips_sparse_bins() -> None:
    offsets = np.asarray([-0.08, -0.02, 0.01, 0.03, 0.05], dtype=np.float64)
    low_sig = np.full(5, 0.2, dtype=np.float64)

    low_result = lunar_rescale_offsets(
        x_offset_px=offsets,
        y_offset_px=offsets,
        x_sig_nm=low_sig,
        y_sig_nm=low_sig,
        pixel_size_nm_x=100.0,
        pixel_size_nm_y=100.0,
        rescale_bins=2,
        threshold=0.01,
        min_bin_count=2,
    )
    sparse_result = lunar_rescale_offsets(
        x_offset_px=offsets,
        y_offset_px=offsets,
        x_sig_nm=np.linspace(10.0, 30.0, 5),
        y_sig_nm=np.linspace(11.0, 31.0, 5),
        pixel_size_nm_x=100.0,
        pixel_size_nm_y=100.0,
        rescale_bins=5,
        threshold=0.01,
        min_bin_count=4,
    )

    np.testing.assert_array_equal(low_result.x_offset_px, offsets)
    np.testing.assert_array_equal(sparse_result.x_offset_px, offsets)
    assert low_result.processed_bins == 0
    assert sparse_result.processed_bins == 0


def test_degrid_h5_is_derived_and_preserves_non_xy_fields(tmp_path) -> None:
    source = tmp_path / "predictions_raw.h5"
    output = tmp_path / "predictions_degrid.h5"
    summary = tmp_path / "degrid_summary.json"
    count = 400
    offsets = np.resize(np.asarray([-0.08, -0.04, -0.02, 0.0, 0.01, 0.03, 0.05], dtype=np.float32), count)
    x_px = 10.5 + offsets
    y_px = 20.5 + offsets[::-1]
    rows = []
    for index in range(count):
        rows.append(
            {
                "frame": index // 10,
                "x_px": x_px[index],
                "y_px": y_px[index],
                "x_px_full": x_px[index] + 700,
                "y_px_full": y_px[index] + 400,
                "x_nm": x_px[index] * 100,
                "y_nm": y_px[index] * 100,
                "x_nm_full": (x_px[index] + 700) * 100,
                "y_nm_full": (y_px[index] + 400) * 100,
                "x_offset_px": offsets[index],
                "y_offset_px": offsets[::-1][index],
                "x_offset_nm": offsets[index] * 100,
                "y_offset_nm": offsets[::-1][index] * 100,
                "x_sig_nm": 5.0 + index * 0.1,
                "y_sig_nm": 6.0 + index * 0.1,
                "z_nm": -300.0 + index,
                "photon": 1000.0 + index,
                "prob": 0.75 + (index % 20) * 0.01,
            }
        )
    fields = list(rows[0])
    with H5PredictionWriter(source, fieldnames=fields, schema="infer_recon_predictions_h5_v0.2") as writer:
        writer.append_rows(rows)

    with h5py.File(source, "r") as handle:
        source_x_before = np.asarray(handle["locs/x_px"][:])
        source_invariants = {key: np.asarray(handle[f"locs/{key}"][:]) for key in ("frame", "z_nm", "photon", "prob")}

    payload = degrid_predictions_h5(
        predictions=source,
        output=output,
        summary_json=summary,
        pixel_size_nm_x=100.0,
        pixel_size_nm_y=100.0,
        rescale_bins=20,
        threshold=0.01,
        min_bin_count=8,
        histogram_png=None,
    )

    with h5py.File(source, "r") as handle:
        np.testing.assert_array_equal(handle["locs/x_px"][:], source_x_before)
    with h5py.File(output, "r") as handle:
        assert int(handle.attrs["count"]) == count
        assert handle.attrs["derived_kind"] == "lunar_offset_degrid"
        assert handle.attrs["source_predictions"] == str(source)
        for key, expected in source_invariants.items():
            np.testing.assert_array_equal(handle[f"locs/{key}"][:], expected)
        assert np.any(np.asarray(handle["locs/x_px"][:]) != source_x_before)
        np.testing.assert_allclose(
            handle["locs/x_px_full"][:] - handle["locs/x_px"][:],
            np.full(count, 700.0),
            rtol=0,
            atol=1e-4,
        )

    assert payload["total_rows"] == count
    assert payload["changed_rows"] > 0
    assert payload["offset_uniformity_cv"]["x_after"] < payload["offset_uniformity_cv"]["x_before"]
    assert payload["offset_uniformity_cv"]["y_after"] < payload["offset_uniformity_cv"]["y_before"]
    assert summary.is_file()


def test_periodic_degrid_uses_coordinate_phase_and_preserves_native_offsets(tmp_path) -> None:
    source = tmp_path / "predictions_period2.h5"
    output = tmp_path / "predictions_period2_degrid.h5"
    summary = tmp_path / "period2_degrid_summary.json"
    count = 4000
    rng = np.random.default_rng(23)
    cell_x = 2.0 * rng.integers(2, 40, count)
    cell_y = 2.0 * rng.integers(2, 40, count)
    biased_phase_x = 2.0 * (rng.beta(6.0, 2.0, count) - 0.5)
    biased_phase_y = 2.0 * (rng.beta(5.0, 2.5, count) - 0.5)
    x_px = cell_x + 1.0 + biased_phase_x
    y_px = cell_y + 1.0 + biased_phase_y
    rows = []
    for index in range(count):
        rows.append(
            {
                "frame": index // 10,
                "x_px": x_px[index],
                "y_px": y_px[index],
                "x_offset_px": x_px[index] - np.floor(x_px[index]) - 0.5,
                "y_offset_px": y_px[index] - np.floor(y_px[index]) - 0.5,
                "x_sig_nm": 10.0 + index % 20,
                "y_sig_nm": 11.0 + index % 20,
                "z_nm": 0.0,
                "prob": 0.9,
            }
        )
    with H5PredictionWriter(
        source,
        fieldnames=list(rows[0]),
        schema="infer_recon_predictions_h5_v0.2",
    ) as writer:
        writer.append_rows(rows)
    source_y = y_px.astype(np.float32)
    source_y_offset = (y_px - np.floor(y_px) - 0.5).astype(np.float32)

    payload = degrid_predictions_h5(
        predictions=source,
        output=output,
        summary_json=summary,
        pixel_size_nm_x=100.0,
        pixel_size_nm_y=100.0,
        rescale_bins=4,
        threshold=0.01,
        min_bin_count=32,
        histogram_png=None,
        period_px_x=2.0,
        period_px_y=2.0,
        axes="x",
    )

    with h5py.File(output, "r") as handle:
        output_x = np.asarray(handle["locs/x_px"][:], dtype=np.float64)
        output_y = np.asarray(handle["locs/y_px"][:], dtype=np.float64)
        np.testing.assert_allclose(
            handle["locs/x_offset_px"][:],
            output_x - np.floor(output_x) - 0.5,
            rtol=0,
            atol=1e-5,
        )
        np.testing.assert_array_equal(output_y, source_y)
        np.testing.assert_array_equal(handle["locs/y_offset_px"][:], source_y_offset)
        assert handle.attrs["degrid_contract"] == "lunar_periodic_rescale_offset_v1"
        assert handle.attrs["degrid_axes"] == "x"
        assert float(handle.attrs["degrid_period_px_x"]) == 2.0
        assert float(handle.attrs["degrid_period_px_y"]) == 2.0

    assert payload["period_px"] == {"x": 2.0, "y": 2.0}
    assert payload["axes"] == ["x"]
    assert payload["offset_uniformity_cv"]["x_after"] < payload["offset_uniformity_cv"]["x_before"] * 0.75
    assert payload["offset_uniformity_cv"]["y_after"] == payload["offset_uniformity_cv"]["y_before"]


def test_spatial_rescale_removes_local_grid_bias_hidden_by_global_cancellation() -> None:
    rng = np.random.default_rng(7)
    count = 40000
    x_px = rng.uniform(0.0, 200.0, count)
    y_px = rng.uniform(0.0, 100.0, count)
    left = x_px < 100.0
    base = rng.beta(5.0, 2.0, count) - 0.5
    x_offset = np.where(left, base, -base)
    y_offset = rng.uniform(-0.5, 0.5, count)
    sigma = rng.uniform(10.0, 20.0, count)

    result = spatial_lunar_rescale_offsets(
        x_px=x_px,
        y_px=y_px,
        x_offset_px=x_offset,
        y_offset_px=y_offset,
        x_sig_nm=sigma,
        y_sig_nm=sigma,
        pixel_size_nm_x=100.0,
        pixel_size_nm_y=100.0,
        field_width_px=200.0,
        field_height_px=100.0,
        spatial_bins_x=2,
        spatial_bins_y=1,
        rescale_bins=1,
        threshold=0.01,
        min_bin_count=32,
    )

    for mask in (left, ~left):
        before = abs(np.mean(np.exp(2j * np.pi * x_offset[mask])))
        after = abs(np.mean(np.exp(2j * np.pi * result.x_offset_px[mask])))
        assert before > 0.2
        assert after < 0.02

