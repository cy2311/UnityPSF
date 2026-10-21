from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from .vector_model import evaluate_normalized_zernike


def recover_carrier_complex(
    wavefront_path: str | Path,
    zmap_path: str | Path,
    output_path: str | Path,
    *,
    wavelength_nm: float,
) -> np.ndarray:
    """Recover the fixed DH carrier from a saved total phase and residual maps."""
    with np.load(Path(wavefront_path), allow_pickle=False) as payload:
        total_phase = np.asarray(payload["wrapped_phase_rad"], dtype=np.float32)
        pupil_mask = np.asarray(payload["pupil_mask"], dtype=bool)
    with np.load(Path(zmap_path), allow_pickle=False) as payload:
        zernike_maps = np.asarray(payload["zernike_maps_nm"], dtype=np.float32)
        mode_order = tuple(tuple(int(value) for value in row) for row in np.asarray(payload["mode_order"]))

    if total_phase.ndim != 2 or pupil_mask.shape != total_phase.shape:
        raise ValueError("wavefront phase and pupil mask must be matching two-dimensional arrays")
    if zernike_maps.ndim != 3 or min(zernike_maps.shape[1:]) <= 0:
        raise ValueError("zernike maps must have shape (modes, height, width)")
    if len(mode_order) != int(zernike_maps.shape[0]):
        raise ValueError("mode_order length must match the zernike map count")
    if float(wavelength_nm) <= 0.0:
        raise ValueError("wavelength_nm must be positive")

    height, width = total_phase.shape
    step = 2.0 / float(width)
    axis = torch.arange(-1.0 + step / 2.0, 1.0, step, dtype=torch.float32)
    x_pupil, y_pupil = torch.meshgrid(axis, axis, indexing="xy")
    basis = evaluate_normalized_zernike(mode_order, x_pupil, y_pupil).numpy()
    # The residual maps live on the camera field grid; the saved wavefront lives
    # on the pupil grid. The carrier recovery uses the calibrated center-field
    # coefficients and evaluates the basis on the pupil grid.
    field_height, field_width = (int(value) for value in zernike_maps.shape[1:])
    coefficients_nm = zernike_maps[:, field_height // 2, field_width // 2]
    residual_opd_nm = np.einsum("c,cyx->yx", coefficients_nm, basis, optimize=True)
    residual_phase = 2.0 * np.pi * residual_opd_nm / float(wavelength_nm)
    carrier = np.exp(1j * (total_phase.astype(np.float64) - residual_phase)).astype(np.complex64)
    carrier[~pupil_mask] = np.complex64(1.0 + 0.0j)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        carrier_complex=carrier,
        complex_pupil=carrier,
        pupil_mask=pupil_mask,
        source_wavefront=np.asarray(str(Path(wavefront_path).resolve())),
        source_zmap=np.asarray(str(Path(zmap_path).resolve())),
        wavelength_nm=np.asarray(float(wavelength_nm), dtype=np.float32),
    )
    return carrier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recover a DH carrier for Neptune gamma physical update.")
    parser.add_argument("--wavefront", type=Path, required=True)
    parser.add_argument("--zmap", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wavelength-nm", type=float, default=660.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    carrier = recover_carrier_complex(
        args.wavefront,
        args.zmap,
        args.output,
        wavelength_nm=float(args.wavelength_nm),
    )
    print(f"carrier_saved={args.output} shape={carrier.shape} dtype={carrier.dtype}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
