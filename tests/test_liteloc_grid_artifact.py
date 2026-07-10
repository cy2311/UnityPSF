import h5py
import numpy as np
import pytest

from neptune_v03.infer_recon.grid_artifact import (
    audit_raw_degrid_predictions,
    compute_liteloc_grid_artifact_index,
)


def _brute_force_liteloc_reference(
    *,
    frame: np.ndarray,
    x_px: np.ndarray,
    y_px: np.ndarray,
    field_size_px: int,
    super_res_factor: int = 10,
    split_blocks: int = 10,
) -> float:
    order = np.argsort(frame)
    x_px = x_px[order]
    y_px = y_px[order]
    block_type = np.zeros(frame.size, dtype=bool)
    block_size = frame.size // split_blocks
    for block_index in range(split_blocks):
        block_type[block_index * block_size : (block_index + 1) * block_size] = bool(block_index % 2)

    images = []
    for keep in (block_type, ~block_type):
        image = np.zeros((field_size_px * super_res_factor,) * 2, dtype=np.float64)
        for x_value, y_value in zip(x_px[keep], y_px[keep]):
            x_index = int(x_value * super_res_factor)
            y_index = int(y_value * super_res_factor)
            if 0 <= x_index < image.shape[1] and 0 <= y_index < image.shape[0]:
                image[y_index, x_index] += 1.0
        images.append(image)

    fft_1 = np.fft.fftshift(np.fft.fft2(images[0]))
    fft_2 = np.fft.fftshift(np.fft.fft2(images[1]))
    center = ((fft_1.shape[0] + 1) // 2, (fft_1.shape[1] + 1) // 2)
    radial_length = int(np.ceil(fft_1.shape[0] / 2) + 1)

    def radial_sum(values: np.ndarray) -> np.ndarray:
        output = np.zeros(radial_length, dtype=np.float64)
        for row in range(values.shape[0]):
            for column in range(values.shape[1]):
                radius = int(np.round(np.hypot(row - center[0], column - center[1])))
                if radius < radial_length:
                    output[radius] += values[row, column]
        return output

    numerator = radial_sum(np.real(fft_1 * np.conjugate(fft_2)))
    denominator = np.sqrt(radial_sum(np.abs(fft_1) ** 2) * radial_sum(np.abs(fft_2) ** 2))
    frc = np.divide(numerator, denominator, out=np.full_like(numerator, np.nan), where=denominator > 0)
    frequency = np.linspace(0.0, super_res_factor / 2.0, frc.size)
    camera_index = int(np.abs(frequency - 1.0).argmin())
    return float(np.nanmax(frc[camera_index - 1 : camera_index + 2]))


def test_vectorized_metric_matches_small_liteloc_reference() -> None:
    rng = np.random.default_rng(7)
    count = 240
    frame = np.repeat(np.arange(24), 10)
    x_px = rng.uniform(0.0, 7.999, count)
    y_px = rng.uniform(0.0, 7.999, count)

    actual = compute_liteloc_grid_artifact_index(
        frame=frame,
        x_px=x_px,
        y_px=y_px,
        field_width_px=8,
        field_height_px=8,
    )
    expected = _brute_force_liteloc_reference(
        frame=frame,
        x_px=x_px,
        y_px=y_px,
        field_size_px=8,
    )

    np.testing.assert_allclose(actual.grid_index, expected, rtol=0, atol=1e-12)


def test_camera_grid_points_have_higher_artifact_index_than_uniform_offsets() -> None:
    rng = np.random.default_rng(19)
    emitters_per_frame = 80
    frame = np.repeat(np.arange(100), emitters_per_frame)
    base_x = rng.integers(0, 32, frame.size)
    base_y = rng.integers(0, 32, frame.size)
    grid_x = base_x + 0.5
    grid_y = base_y + 0.5
    uniform_x = base_x + rng.uniform(0.0, 1.0, frame.size)
    uniform_y = base_y + rng.uniform(0.0, 1.0, frame.size)

    grid = compute_liteloc_grid_artifact_index(
        frame=frame,
        x_px=grid_x,
        y_px=grid_y,
        field_width_px=32,
        field_height_px=32,
    )
    uniform = compute_liteloc_grid_artifact_index(
        frame=frame,
        x_px=uniform_x,
        y_px=uniform_y,
        field_width_px=32,
        field_height_px=32,
    )

    assert grid.grid_index > uniform.grid_index


def test_metric_does_not_modify_inputs() -> None:
    frame = np.arange(100, dtype=np.int32)
    x_px = np.linspace(0.1, 9.9, 100)
    y_px = np.linspace(9.9, 0.1, 100)
    originals = (frame.copy(), x_px.copy(), y_px.copy())

    result = compute_liteloc_grid_artifact_index(
        frame=frame,
        x_px=x_px,
        y_px=y_px,
        field_width_px=10,
        field_height_px=10,
    )

    assert result.localization_count == 100
    for actual, expected in zip((frame, x_px, y_px), originals):
        np.testing.assert_array_equal(actual, expected)


def _write_predictions(path, *, x_px: np.ndarray, y_px: np.ndarray, prob: np.ndarray) -> None:
    with h5py.File(path, "w") as handle:
        group = handle.create_group("locs")
        group.create_dataset("frame", data=np.arange(x_px.size, dtype=np.int32) // 4)
        group.create_dataset("x_px", data=x_px)
        group.create_dataset("y_px", data=y_px)
        group.create_dataset("prob", data=prob)
        group.create_dataset("z_nm", data=np.linspace(-100.0, 100.0, x_px.size))


def test_raw_degrid_audit_requires_identical_non_xy_rows(tmp_path) -> None:
    count = 100
    raw = tmp_path / "raw.h5"
    degrid = tmp_path / "degrid.h5"
    x_px = np.linspace(0.1, 7.9, count)
    y_px = np.linspace(7.9, 0.1, count)
    prob = np.full(count, 0.95)
    _write_predictions(raw, x_px=x_px, y_px=y_px, prob=prob)
    _write_predictions(degrid, x_px=x_px + 0.01, y_px=y_px - 0.01, prob=prob)

    payload = audit_raw_degrid_predictions(
        raw_predictions=raw,
        degrid_predictions=degrid,
        field_width_px=8,
        field_height_px=8,
    )

    assert payload["count_parity"] is True
    assert payload["invariant_field_parity"] is True
    assert payload["raw"]["localization_count"] == count
    assert payload["degrid"]["localization_count"] == count

    with h5py.File(degrid, "r+") as handle:
        handle["locs/prob"][0] = 0.1
    with pytest.raises(ValueError, match="non-XY field differs: prob"):
        audit_raw_degrid_predictions(
            raw_predictions=raw,
            degrid_predictions=degrid,
            field_width_px=8,
            field_height_px=8,
        )
