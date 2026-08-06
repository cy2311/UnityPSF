from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .._paths import PROJECT_ROOT as V04_ROOT
from ..dataset import Microscope1Dataset
from ..gamma_field import DirectGammaZernikeField
from ..localization import LocalizationConfig
from ..vector_model import DoubleHelixVectorPSF


DEFAULT_DATASET_ROOT = V04_ROOT.parent / "datasets/training_sets/double_helix/Simulated_datasets_Microscope1"
DEFAULT_GAMMA_PATH = V04_ROOT / "output/double_helix/microscope1/field_gamma/arrays/gamma_coefficients.npz"
DEFAULT_EVALUATION_DIR = V04_ROOT / "output/double_helix/microscope1/evaluation"


def compose_frame_reconstruction(
    psfs_unit_flux: np.ndarray,
    x_px: np.ndarray,
    y_px: np.ndarray,
    photons_adu: np.ndarray,
    background_adu: np.ndarray,
    *,
    frame_shape_hw: tuple[int, int],
) -> tuple[np.ndarray, float]:
    psfs = np.asarray(psfs_unit_flux, dtype=np.float32)
    x = np.asarray(x_px, dtype=np.float32).reshape(-1)
    y = np.asarray(y_px, dtype=np.float32).reshape(-1)
    photons = np.asarray(photons_adu, dtype=np.float32).reshape(-1)
    backgrounds = np.asarray(background_adu, dtype=np.float32).reshape(-1)
    if psfs.ndim != 3 or psfs.shape[1] != psfs.shape[2] or psfs.shape[1] % 2 != 1:
        raise ValueError("psfs_unit_flux must have shape (N,S,S) with an odd patch size.")
    if not (psfs.shape[0] == x.size == y.size == photons.size == backgrounds.size):
        raise ValueError("PSF and localization arrays must have matching leading dimensions.")
    height, width = (int(value) for value in frame_shape_hw)
    background = float(np.median(backgrounds)) if backgrounds.size else 0.0
    reconstruction = np.full((height, width), background, dtype=np.float32)
    radius = psfs.shape[1] // 2
    for psf, center_x, center_y, emitter_photons in zip(psfs, x, y, photons, strict=True):
        pixel_x = int(np.floor(float(center_x) + 0.5))
        pixel_y = int(np.floor(float(center_y) + 0.5))
        image_x0 = max(pixel_x - radius, 0)
        image_y0 = max(pixel_y - radius, 0)
        image_x1 = min(pixel_x + radius + 1, width)
        image_y1 = min(pixel_y + radius + 1, height)
        patch_x0 = image_x0 - (pixel_x - radius)
        patch_y0 = image_y0 - (pixel_y - radius)
        patch_x1 = patch_x0 + image_x1 - image_x0
        patch_y1 = patch_y0 + image_y1 - image_y0
        reconstruction[image_y0:image_y1, image_x0:image_x1] += (
            float(emitter_photons) * psf[patch_y0:patch_y1, patch_x0:patch_x1]
        )
    return reconstruction, background


def orthogonal_max_projections(psf_volume: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    volume = np.asarray(psf_volume, dtype=np.float32)
    if volume.ndim != 3:
        raise ValueError("psf_volume must have shape (Z,Y,X).")
    return volume.max(axis=1), volume.max(axis=2)


def render_diagnostics(
    *,
    dataset_root: Path,
    gamma_path: Path,
    localizations_path: Path,
    evaluation_manifest_path: Path,
    output_dir: Path,
    frame_indices: np.ndarray,
    device: str,
    carrier_path: Path | None = None,
) -> dict[str, Any]:
    dataset = Microscope1Dataset(dataset_root)
    dataset.validate()
    manifest = json.loads(evaluation_manifest_path.read_text(encoding="utf-8"))
    config = _localization_config(manifest["config"], device=device)
    with np.load(gamma_path, allow_pickle=False) as gamma_payload:
        gamma_nm = np.asarray(gamma_payload["gamma_nm"], dtype=np.float32)
        mode_order = tuple(tuple(int(value) for value in row) for row in gamma_payload["mode_order"])
    carrier_complex = None
    if carrier_path is not None:
        with np.load(carrier_path, allow_pickle=False) as carrier_payload:
            key = "carrier_complex" if "carrier_complex" in carrier_payload else "complex_pupil"
            carrier_complex = np.asarray(carrier_payload[key], dtype=np.complex64)
    with np.load(localizations_path, allow_pickle=False) as localization_payload:
        localizations = {key: np.asarray(localization_payload[key]) for key in localization_payload.files}

    selected_frames = np.asarray(dataset.open_frames()[frame_indices], dtype=np.float32)
    model = DoubleHelixVectorPSF(
        mode_order=mode_order,
        na=config.na,
        wavelength_nm=config.wavelength_nm,
        pixel_size_nm=config.pixel_size_nm,
        refractive_index=config.refractive_index,
        npupil=config.npupil,
        psf_size=config.psf_size,
        device=device,
    )
    field = DirectGammaZernikeField(
        gamma_nm=torch.as_tensor(gamma_nm, dtype=torch.float32, device=device),
        mode_order=mode_order,
    )
    reconstructions, backgrounds = _reconstruct_frames(
        frame_indices,
        localizations=localizations,
        model=model,
        field=field,
        config=config,
        carrier_complex=carrier_complex,
    )
    frame_ncc = np.asarray(
        [_signal_ncc(raw, recon, background) for raw, recon, background in zip(selected_frames, reconstructions, backgrounds, strict=True)],
        dtype=np.float32,
    )
    frame_nrmse = np.asarray(
        [_signal_nrmse(raw, recon, background) for raw, recon, background in zip(selected_frames, reconstructions, backgrounds, strict=True)],
        dtype=np.float32,
    )

    z_nm = (
        np.arange(dataset.config.calibration_planes, dtype=np.float32) + dataset.z_index_origin
    ) * dataset.z_sign * dataset.z_step_nm
    center_x = torch.full((z_nm.size,), config.image_shape_hw[1] / 2.0, device=device)
    center_y = torch.full((z_nm.size,), config.image_shape_hw[0] / 2.0, device=device)
    center_coefficients = field.evaluate(
        -1.0 + 2.0 * center_x / float(config.image_shape_hw[1]),
        -1.0 + 2.0 * center_y / float(config.image_shape_hw[0]),
    )
    with torch.no_grad():
        psf_volume = model.render(
            coefficients_nm=center_coefficients,
            z_nm=torch.as_tensor(z_nm, dtype=torch.float32, device=device),
            carrier_complex=carrier_complex,
        ).cpu().numpy()
    xz_projection, yz_projection = orthogonal_max_projections(psf_volume)
    sample_indices = np.linspace(0, z_nm.size - 1, 7, dtype=np.int64)

    arrays_dir = output_dir / "arrays"
    figures_dir = output_dir / "figures"
    metadata_dir = output_dir / "metadata"
    for directory in (arrays_dir, figures_dir, metadata_dir):
        directory.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        arrays_dir / "raw_vs_wavefront_reconstruction.npz",
        frame_index_0based=frame_indices,
        raw_adu=selected_frames,
        reconstruction_adu=reconstructions,
        fitted_background_adu=backgrounds,
        frame_signal_ncc=frame_ncc,
        frame_signal_nrmse=frame_nrmse,
    )
    np.savez_compressed(
        arrays_dir / "recovered_wavefront_psf_z_volume.npz",
        z_nm=z_nm,
        psf_unit_flux=psf_volume.astype(np.float32),
        sample_plane_index=sample_indices,
        sample_z_nm=z_nm[sample_indices],
        xz_max_projection=xz_projection,
        yz_max_projection=yz_projection,
        center_coefficients_nm=center_coefficients[0].detach().cpu().numpy().astype(np.float32),
        mode_order=np.asarray(mode_order, dtype=np.int64),
    )
    _plot_raw_vs_reconstruction(
        selected_frames,
        reconstructions,
        frame_indices=frame_indices,
        frame_ncc=frame_ncc,
        frame_nrmse=frame_nrmse,
        path=figures_dir / "raw_frames_vs_recovered_wavefront_reconstruction.png",
    )
    _plot_psf_z_views(
        psf_volume,
        z_nm=z_nm,
        sample_indices=sample_indices,
        pixel_size_nm=config.pixel_size_nm,
        path=figures_dir / "recovered_wavefront_psf_z_and_xz_yz_views.png",
    )
    result = {
        "device": device,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE",
        "torch_cuda_available": bool(torch.cuda.is_available()),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", "not-under-slurm"),
        "gamma_sha256": _sha256(gamma_path),
        "shared_carrier_input": str(carrier_path.resolve()) if carrier_path is not None else None,
        "shared_carrier_sha256": _sha256(carrier_path) if carrier_path is not None else None,
        "localizations_sha256": _sha256(localizations_path),
        "frame_indices_0based": frame_indices.tolist(),
        "frame_signal_ncc": frame_ncc.tolist(),
        "frame_signal_nrmse": frame_nrmse.tolist(),
        "reconstruction_semantics": (
            "sum of shared-carrier plus field-residual-gamma vector PSFs at independently "
            "fitted x/y/z, scaled by fitted photons and median fitted background; GT is not used"
            if carrier_path is not None
            else "sum of direct-gamma vector PSFs at independently fitted x/y/z, scaled by "
            "fitted photons and median fitted background; GT is not used"
        ),
        "psf_volume_semantics": (
            "unit-flux shared-carrier plus field-residual-gamma vector PSF at field center "
            "over all 119 calibrated z planes"
            if carrier_path is not None
            else "unit-flux direct-gamma vector PSF at field center over all 119 calibrated z planes"
        ),
        "xz_semantics": "maximum projection over y",
        "yz_semantics": "maximum projection over x",
    }
    (metadata_dir / "wavefront_diagnostics.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def _reconstruct_frames(
    frame_indices: np.ndarray,
    *,
    localizations: dict[str, np.ndarray],
    model: DoubleHelixVectorPSF,
    field: DirectGammaZernikeField,
    config: LocalizationConfig,
    carrier_complex: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    reconstructions = []
    backgrounds = []
    for frame_index in frame_indices:
        rows = localizations["frame_index"] == frame_index
        x_px = localizations["x_px"][rows]
        y_px = localizations["y_px"][rows]
        center_x = np.floor(x_px + 0.5)
        center_y = np.floor(y_px + 0.5)
        x_t = torch.as_tensor(x_px, dtype=torch.float32, device=model.device)
        y_t = torch.as_tensor(y_px, dtype=torch.float32, device=model.device)
        coefficients = field.evaluate(
            -1.0 + 2.0 * x_t / float(config.image_shape_hw[1]),
            -1.0 + 2.0 * y_t / float(config.image_shape_hw[0]),
        )
        with torch.no_grad():
            psfs = model.render(
                coefficients_nm=coefficients,
                z_nm=torch.as_tensor(localizations["z_nm"][rows], dtype=torch.float32, device=model.device),
                carrier_complex=carrier_complex,
                dx_px=torch.as_tensor(x_px - center_x, dtype=torch.float32, device=model.device),
                dy_px=torch.as_tensor(y_px - center_y, dtype=torch.float32, device=model.device),
            ).cpu().numpy()
        reconstruction, background = compose_frame_reconstruction(
            psfs,
            x_px,
            y_px,
            localizations["photons_adu"][rows],
            localizations["background_adu"][rows],
            frame_shape_hw=config.image_shape_hw,
        )
        reconstructions.append(reconstruction)
        backgrounds.append(background)
    return np.stack(reconstructions), np.asarray(backgrounds, dtype=np.float32)


def _signal_ncc(raw: np.ndarray, reconstruction: np.ndarray, background: float) -> float:
    observed = np.maximum(np.asarray(raw, dtype=np.float64) - background, 0.0)
    expected = np.maximum(np.asarray(reconstruction, dtype=np.float64) - background, 0.0)
    observed -= observed.mean()
    expected -= expected.mean()
    denominator = np.linalg.norm(observed) * np.linalg.norm(expected)
    return float(np.dot(observed.ravel(), expected.ravel()) / denominator) if denominator else 0.0


def _signal_nrmse(raw: np.ndarray, reconstruction: np.ndarray, background: float) -> float:
    observed = np.maximum(np.asarray(raw, dtype=np.float64) - background, 0.0)
    expected = np.maximum(np.asarray(reconstruction, dtype=np.float64) - background, 0.0)
    denominator = np.linalg.norm(observed)
    return float(np.linalg.norm(observed - expected) / denominator) if denominator else 0.0


def _plot_raw_vs_reconstruction(
    raw: np.ndarray,
    reconstruction: np.ndarray,
    *,
    frame_indices: np.ndarray,
    frame_ncc: np.ndarray,
    frame_nrmse: np.ndarray,
    path: Path,
) -> None:
    plt = _plotting()
    with plt.rc_context(_figure_style()):
        fig, axes = plt.subplots(2, raw.shape[0], figsize=(7.2, 4.15), constrained_layout=True)
        for column, frame_index in enumerate(frame_indices):
            vmin = float(np.percentile(raw[column], 1.0))
            vmax = float(np.percentile(raw[column], 99.9))
            axes[0, column].imshow(raw[column], cmap="gray", vmin=vmin, vmax=vmax)
            axes[1, column].imshow(reconstruction[column], cmap="gray", vmin=vmin, vmax=vmax)
            axes[0, column].set_title(f"Frame {int(frame_index) + 1}")
            axes[1, column].set_xlabel(f"NCC {frame_ncc[column]:.3f} | NRMSE {frame_nrmse[column]:.3f}")
            for row in range(2):
                axes[row, column].set_xticks([])
                axes[row, column].set_yticks([])
        axes[0, 0].set_ylabel("Raw frame (ADU)")
        axes[1, 0].set_ylabel("Recovered-wavefront\nPSF reconstruction")
        fig.suptitle("Raw frames versus reconstruction from independent localizations", fontsize=10)
        _save_png_pdf(fig, path)
        plt.close(fig)


def _plot_psf_z_views(
    psf_volume: np.ndarray,
    *,
    z_nm: np.ndarray,
    sample_indices: np.ndarray,
    pixel_size_nm: float,
    path: Path,
) -> None:
    plt = _plotting()
    from matplotlib.colors import PowerNorm

    xz_projection, yz_projection = orthogonal_max_projections(psf_volume)
    radius_um = (psf_volume.shape[-1] // 2) * pixel_size_nm / 1000.0
    lateral_extent = (-radius_um, radius_um, radius_um, -radius_um)
    side_extent = (-radius_um, radius_um, float(z_nm[-1] / 1000.0), float(z_nm[0] / 1000.0))
    intensity_norm = PowerNorm(gamma=0.5, vmin=0.0, vmax=float(psf_volume.max()))
    with plt.rc_context(_figure_style()):
        fig = plt.figure(figsize=(8.4, 5.0))
        grid = fig.add_gridspec(
            2,
            14,
            height_ratios=(1.0, 1.25),
            hspace=0.5,
            wspace=0.3,
            left=0.07,
            right=0.98,
            bottom=0.1,
            top=0.84,
        )
        for position, plane_index in enumerate(sample_indices):
            axis = fig.add_subplot(grid[0, 2 * position : 2 * position + 2])
            axis.imshow(
                psf_volume[plane_index],
                cmap="viridis",
                norm=intensity_norm,
                extent=lateral_extent,
            )
            axis.set_title(f"z = {z_nm[plane_index] / 1000.0:.2f} um", fontsize=7)
            axis.set_xticks([])
            axis.set_yticks([])
        xz_axis = fig.add_subplot(grid[1, :6])
        yz_axis = fig.add_subplot(grid[1, 8:])
        xz_axis.imshow(xz_projection, cmap="viridis", aspect="auto", extent=side_extent, norm=intensity_norm)
        yz_axis.imshow(yz_projection, cmap="viridis", aspect="auto", extent=side_extent, norm=intensity_norm)
        xz_axis.set_title("XZ maximum projection over Y")
        yz_axis.set_title("YZ maximum projection over X")
        xz_axis.set_xlabel("X (um)")
        yz_axis.set_xlabel("Y (um)")
        xz_axis.set_ylabel("Z (um)")
        yz_axis.set_ylabel("Z (um)")
        fig.suptitle("Recovered double-helix PSF across Z", fontsize=10, y=0.96)
        _save_png_pdf(fig, path)
        plt.close(fig)


def _localization_config(values: dict[str, Any], *, device: str) -> LocalizationConfig:
    return LocalizationConfig(
        batch_size=int(values["batch_size"]),
        refinement_steps=int(values["refinement_steps"]),
        xy_learning_rate_px=float(values["xy_learning_rate_px"]),
        z_learning_rate_nm=float(values["z_learning_rate_nm"]),
        minimum_ncc=float(values["minimum_ncc"]),
        z_range_nm=tuple(float(value) for value in values["z_range_nm"]),
        pixel_size_nm=float(values["pixel_size_nm"]),
        image_origin_xy_px=tuple(float(value) for value in values["image_origin_xy_px"]),
        image_shape_hw=tuple(int(value) for value in values["image_shape_hw"]),
        na=float(values["na"]),
        wavelength_nm=float(values["wavelength_nm"]),
        refractive_index=float(values["refractive_index"]),
        npupil=int(values["npupil"]),
        psf_size=int(values["psf_size"]),
        device=device,
    )


def _figure_style() -> dict[str, Any]:
    return {
        "font.family": "sans-serif",
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
    }


def _plotting() -> Any:
    cache_dir = V04_ROOT / ".local/cache/matplotlib"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
    import matplotlib.pyplot as plt

    return plt


def _save_png_pdf(fig: Any, path: Path) -> None:
    fig.savefig(path, dpi=300, facecolor="white")
    fig.savefig(path.with_suffix(".pdf"), facecolor="white")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render raw-frame and axial PSF diagnostics from a recovered DH pupil.")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--gamma-path", type=Path, default=DEFAULT_GAMMA_PATH)
    parser.add_argument("--carrier-complex", type=Path)
    parser.add_argument("--localizations-path", type=Path, default=DEFAULT_EVALUATION_DIR / "arrays/independent_localizations.npz")
    parser.add_argument("--evaluation-manifest-path", type=Path, default=DEFAULT_EVALUATION_DIR / "metadata/manifest.json")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_EVALUATION_DIR)
    parser.add_argument("--frames", default="1,1667,3333,5000", help="Comma-separated 1-based TIFF frame numbers.")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for formal diagnostics but is unavailable.")
    frame_indices = np.asarray([int(value) - 1 for value in args.frames.split(",")], dtype=np.int64)
    if frame_indices.size == 0 or np.any(frame_indices < 0):
        raise ValueError("--frames must contain positive 1-based frame numbers.")
    result = render_diagnostics(
        dataset_root=args.dataset_root,
        gamma_path=args.gamma_path,
        localizations_path=args.localizations_path,
        evaluation_manifest_path=args.evaluation_manifest_path,
        output_dir=args.output_dir,
        frame_indices=frame_indices,
        device=args.device,
        carrier_path=args.carrier_complex,
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["compose_frame_reconstruction", "orthogonal_max_projections", "render_diagnostics"]
