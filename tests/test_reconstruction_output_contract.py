import json

import h5py
import numpy as np

from neptune_v03.infer_recon.predictions_io import H5PredictionWriter, read_render_arrays
from neptune_v03.infer_recon.recon.render_subpixel import (
    DEFAULT_Z_MAX_NM,
    DEFAULT_Z_MIN_NM,
    _fixed_sigma_render_px,
    _normalize_display,
    _colorize_density_display,
    _resolve_render_weights,
    _smap_quantile_from_imax_min,
)


def test_default_z_range_is_explicitly_in_nanometers() -> None:
    assert DEFAULT_Z_MIN_NM == -600.0
    assert DEFAULT_Z_MAX_NM == 600.0


def test_fixed_sigma_supports_smap_default_minimum_of_point_seven_render_pixel() -> None:
    assert _fixed_sigma_render_px(spot_radius_nm=28.0, render_pixel_nm=20.0) == 0.7


def test_count_render_weight_is_independent_of_integrated_probability() -> None:
    probability = np.asarray([0.71, 1.2, 2.4], dtype=np.float32)
    photon = np.asarray([1000.0, 2000.0, 3000.0], dtype=np.float32)

    weights = _resolve_render_weights(
        mode="count",
        count=3,
        probability=probability,
        photon=photon,
    )

    np.testing.assert_array_equal(weights, np.ones(3, dtype=np.float32))


def test_probability_and_photon_render_weights_are_explicit_modes() -> None:
    probability = np.asarray([0.71, 1.2], dtype=np.float32)
    photon = np.asarray([1200.0, 2400.0], dtype=np.float32)

    np.testing.assert_array_equal(
        _resolve_render_weights(mode="probability", count=2, probability=probability, photon=photon),
        probability,
    )
    np.testing.assert_array_equal(
        _resolve_render_weights(mode="photon", count=2, probability=probability, photon=photon),
        photon,
    )


def test_smap_default_imax_min_maps_to_the_documented_quantile() -> None:
    assert _smap_quantile_from_imax_min(-3.5) == 1.0 - 10.0**-3.5


def test_fixed_imax_display_normalization_does_not_modify_linear_density() -> None:
    density = np.asarray([[0.0, 2.0], [4.0, 8.0]], dtype=np.float32)
    display, imax, source = _normalize_display(
        density,
        mode="fixed_imax",
        fixed_imax=4.0,
        imax_min=-3.5,
        gamma=1.0,
        brightness=1.0,
        normalize_roi=None,
    )

    assert imax == 4.0
    assert source == "fixed_imax"
    np.testing.assert_array_equal(density, np.asarray([[0.0, 2.0], [4.0, 8.0]], dtype=np.float32))
    np.testing.assert_array_equal(display, np.asarray([[0.0, 0.5], [1.0, 1.0]], dtype=np.float32))


def test_density_is_the_display_brightness_while_rgb_only_carries_hue() -> None:
    density = np.asarray([[2.0]], dtype=np.float32)
    linear_rgb = np.asarray([[[0.2, 1.0, 0.1]]], dtype=np.float32)

    display_rgb = _colorize_density_display(
        density,
        linear_rgb,
        imax=4.0,
        gamma=1.0,
        brightness=1.0,
    )

    np.testing.assert_allclose(display_rgb, np.asarray([[[0.1, 0.5, 0.05]]], dtype=np.float32))


def test_h5_v02_contract_records_units_and_explicit_fields(tmp_path) -> None:
    path = tmp_path / "predictions.h5"
    with H5PredictionWriter(
        path,
        fieldnames=["x_px", "y_px", "z", "z_nm", "prob", "photon", "x_sig", "y_sig", "x_sig_px", "y_sig_px"],
        schema="infer_recon_predictions_h5_v0.2",
        attributes={"units": {"z_nm": "nm", "x_sig_px": "camera_pixel"}},
    ) as writer:
        writer.append_rows(
            [
                {
                    "x_px": 1.5,
                    "y_px": 2.5,
                    "z": 12.0,
                    "z_nm": 120.0,
                    "prob": 0.8,
                    "photon": 2000.0,
                    "x_sig": 9.0,
                    "y_sig": 9.0,
                    "x_sig_px": 0.2,
                    "y_sig_px": 0.3,
                }
            ]
        )

    with h5py.File(path, "r") as handle:
        assert handle.attrs["schema"] == "infer_recon_predictions_h5_v0.2"
        assert json.loads(handle.attrs["units"])["z_nm"] == "nm"

    _, _, z_nm, _, photon, x_sig_px, y_sig_px = read_render_arrays(path, 0.7)
    np.testing.assert_array_equal(z_nm, np.asarray([120.0], dtype=np.float32))
    np.testing.assert_array_equal(photon, np.asarray([2000.0], dtype=np.float32))
    np.testing.assert_array_equal(x_sig_px, np.asarray([0.2], dtype=np.float32))
    np.testing.assert_array_equal(y_sig_px, np.asarray([0.3], dtype=np.float32))
