from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import tifffile
import torch
import torch.nn.functional as F
import yaml

from unity_psf.localization.online import (
    OnlineBatchProviderConfig,
    build_online_batch_provider,
    build_sliding_window_origin_bank,
)
from unity_psf.localization.runtime_config import build_localization_runtime_config
from unity_psf.optics.vector_psf import (
    VectorPSFParams,
    build_vector_psf_context,
    noll_to_nm,
    render_vector_psf_bank,
)

from .._paths import PROJECT_ROOT
from ..vector_model import evaluate_normalized_zernike


@dataclass(frozen=True)
class ReferenceSimulationContract:
    image_shape_hw: tuple[int, int]
    frames_per_sample: int
    training_batch_size: int
    pupil_size_px: int
    psf_support_px: int
    simulation_backend: str
    photon_mean: float
    photon_std: float
    photon_clip: tuple[float, float]
    background_photons: tuple[float, float]
    emitter_density_um2: float
    camera_qe: float
    camera_spurious_charge: float
    camera_baseline_adu: float
    camera_e_per_adu: float
    pixel_size_nm_xy: tuple[float, float]
    wavelength_nm: float
    numerical_aperture: float
    medium_refractive_index: float


@dataclass(frozen=True)
class PreparedZernikeMaps:
    source_path: Path
    output_path: Path
    source_mode_count: int
    exported_mode_count: int
    source_field_shape_hw: tuple[int, int]
    field_shape_hw: tuple[int, int]
    source_pixel_size_nm: float | None
    target_pixel_size_nm_xy: tuple[float, float] | None


@dataclass(frozen=True)
class SimulationOutputs:
    output_dir: Path
    manifest_path: Path
    batch_arrays_path: Path
    training_tiff_path: Path
    gamma_roi_bank_tiff_path: Path
    psf_volume_path: Path
    figure_paths: tuple[Path, ...]


def _load_yaml(path: str | Path) -> dict[str, Any]:
    resolved_path = Path(path).resolve()
    payload = yaml.safe_load(resolved_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a mapping in {resolved_path}")
    return payload


def export_first_full_roi_frames(
    frames_adu: np.ndarray,
    output_path: str | Path,
    *,
    frame_count: int = 100,
) -> Path:
    frames = np.asarray(frames_adu)
    if frames.ndim != 4:
        raise ValueError(f"frames_adu must have shape (N,T,H,W), got {frames.shape}")
    flattened = frames.reshape(-1, frames.shape[-2], frames.shape[-1])
    if flattened.shape[0] < int(frame_count):
        raise ValueError(f"requested {frame_count} frames but only {flattened.shape[0]} are available")
    output = Path(output_path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(
        output,
        flattened[: int(frame_count)],
        photometric="minisblack",
        metadata={"axes": "TYX"},
    )
    return output


def load_reference_contract(path: str | Path) -> ReferenceSimulationContract:
    config = _load_yaml(path)
    train = config["train"]
    online = train["online_generation"]
    simulation = config["simulation"]
    emitter = simulation["emitter"]
    vector = simulation["psf"]["vector"]
    lut = online["lut_simulation"]
    camera = config["camera"]
    optical = config["optical"]

    return ReferenceSimulationContract(
        image_shape_hw=(int(online["height"]), int(online["width"])),
        frames_per_sample=int(online["channels"]),
        training_batch_size=int(train["batch_size"]),
        pupil_size_px=int(vector["npupil"]),
        psf_support_px=int(lut["psf_size"]),
        simulation_backend=str(online["simulation_backend"]),
        photon_mean=float(emitter["intensity_mu_sig"][0]),
        photon_std=float(emitter["intensity_mu_sig"][1]),
        photon_clip=tuple(float(value) for value in emitter["intensity_clip"]),
        background_photons=tuple(float(value) for value in simulation["background_uniform"]),
        emitter_density_um2=float(online["emitter_density_um2"]),
        camera_qe=float(camera["qe"]),
        camera_spurious_charge=float(camera["spurious_charge"]),
        camera_baseline_adu=float(camera["baseline"]),
        camera_e_per_adu=float(camera["e_per_adu"]),
        pixel_size_nm_xy=(float(optical["pixel_size_nm_x"]), float(optical["pixel_size_nm_y"])),
        wavelength_nm=float(optical["wavelength_nm"]),
        numerical_aperture=float(optical["NA"]),
        medium_refractive_index=float(optical["n_medium"]),
    )


def prepare_zero_defocus_zmap(
    source_path: str | Path,
    output_path: str | Path,
    *,
    source_pixel_size_nm: float | None = None,
    target_pixel_size_nm_xy: tuple[float, float] | None = None,
) -> PreparedZernikeMaps:
    source = Path(source_path).resolve()
    output = Path(output_path).resolve()
    with np.load(source, allow_pickle=False) as payload:
        maps_nm = np.asarray(payload["zernike_maps_nm"], dtype=np.float32)
        mode_order = np.asarray(payload["mode_order"], dtype=np.int64)

    if maps_nm.ndim != 3 or mode_order.shape != (maps_nm.shape[0], 2):
        raise ValueError("Expected zernike_maps_nm (C,H,W) and matching mode_order (C,2)")
    if np.any(np.all(mode_order == np.asarray((2, 0), dtype=np.int64), axis=1)):
        raise ValueError("Source w5434 zmap unexpectedly contains the fixed (2,0) defocus mode")

    source_shape = (int(maps_nm.shape[1]), int(maps_nm.shape[2]))
    if (source_pixel_size_nm is None) != (target_pixel_size_nm_xy is None):
        raise ValueError("source and target pixel sizes must be provided together")
    if source_pixel_size_nm is not None and target_pixel_size_nm_xy is not None:
        source_pixel = float(source_pixel_size_nm)
        target_x, target_y = (float(value) for value in target_pixel_size_nm_xy)
        if min(source_pixel, target_x, target_y) <= 0.0:
            raise ValueError("physical pixel sizes must be positive")
        target_h = max(1, int(round(source_shape[0] * source_pixel / target_y)))
        target_w = max(1, int(round(source_shape[1] * source_pixel / target_x)))
        maps_nm = (
            F.interpolate(
                torch.from_numpy(maps_nm)[None],
                size=(target_h, target_w),
                mode="bilinear",
                align_corners=False,
            )[0]
            .numpy()
            .astype(np.float32, copy=False)
        )

    zero_defocus = np.zeros((1, maps_nm.shape[1], maps_nm.shape[2]), dtype=np.float32)
    exported_maps = np.concatenate((zero_defocus, maps_nm), axis=0)
    exported_order = np.concatenate((np.asarray(((2, 0),), dtype=np.int64), mode_order), axis=0)
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {}
    if source_pixel_size_nm is not None and target_pixel_size_nm_xy is not None:
        metadata = {
            "source_field_shape_hw": np.asarray(source_shape, dtype=np.int64),
            "source_pixel_size_nm": np.asarray(source_pixel_size_nm, dtype=np.float32),
            "target_pixel_size_nm_xy": np.asarray(target_pixel_size_nm_xy, dtype=np.float32),
            "resampling_semantics": np.asarray("bilinear pixel-cell physical-coordinate resampling"),
        }
    np.savez_compressed(
        output,
        zernike_maps_nm=exported_maps,
        mode_order=exported_order,
        **metadata,
    )
    return PreparedZernikeMaps(
        source_path=source,
        output_path=output,
        source_mode_count=int(maps_nm.shape[0]),
        exported_mode_count=int(exported_maps.shape[0]),
        source_field_shape_hw=source_shape,
        field_shape_hw=(int(maps_nm.shape[1]), int(maps_nm.shape[2])),
        source_pixel_size_nm=None if source_pixel_size_nm is None else float(source_pixel_size_nm),
        target_pixel_size_nm_xy=(
            None
            if target_pixel_size_nm_xy is None
            else tuple(float(value) for value in target_pixel_size_nm_xy)
        ),
    )


def calibrated_label_z_range_nm(z_min_nm: float, z_max_nm: float) -> tuple[float, float]:
    center_nm = 0.5 * (float(z_min_nm) + float(z_max_nm))
    return float(z_min_nm) - center_nm, float(z_max_nm) - center_nm


def model_z_from_label_z_nm(
    label_z_nm: np.ndarray | float,
    *,
    model_center_nm: float,
) -> np.ndarray:
    return np.asarray(label_z_nm, dtype=np.float64) + float(model_center_nm)


def lut_steps_preserving_reference_spacing(
    *,
    reference_z_range_nm: tuple[float, float],
    reference_z_steps: int,
    requested_z_range_nm: tuple[float, float],
) -> int:
    if int(reference_z_steps) < 2:
        raise ValueError("reference_z_steps must be at least two")
    reference_span = float(reference_z_range_nm[1]) - float(reference_z_range_nm[0])
    requested_span = float(requested_z_range_nm[1]) - float(requested_z_range_nm[0])
    if min(reference_span, requested_span) <= 0.0:
        raise ValueError("z ranges must be increasing")
    reference_spacing = reference_span / float(int(reference_z_steps) - 1)
    nominal_intervals = requested_span / reference_spacing
    interval_count = max(1, int(round(nominal_intervals)))
    if math.isclose(
        float(requested_z_range_nm[0]),
        -float(requested_z_range_nm[1]),
        abs_tol=1e-9,
    ):
        candidates = {max(2, interval_count - 1), max(2, interval_count), interval_count + 1}
        even_candidates = [value for value in candidates if value % 2 == 0]
        interval_count = min(
            even_candidates,
            key=lambda value: abs(requested_span / float(value) - reference_spacing),
        )
    return interval_count + 1


def scaled_odd_psf_support(
    *,
    source_support_px: int,
    source_pixel_size_nm: float,
    target_pixel_size_nm_xy: tuple[float, float],
) -> int:
    target_min = min(float(value) for value in target_pixel_size_nm_xy)
    scaled = int(math.ceil(int(source_support_px) * float(source_pixel_size_nm) / target_min))
    return scaled if scaled % 2 == 1 else scaled + 1


def build_export_provider_config(
    reference_config_path: str | Path,
    prepared_zmap_path: str | Path,
    *,
    seed: int,
    emitter_density_um2: float | None = None,
    shared_carrier_path: str | Path | None = None,
    sample_count: int | None = None,
) -> OnlineBatchProviderConfig:
    reference_path = Path(reference_config_path).resolve()
    prepared_path = Path(prepared_zmap_path).resolve()
    config = _load_yaml(reference_path)
    runtime = build_localization_runtime_config(
        config,
        config_base_dir=reference_path.parent,
        seed=int(seed),
    )
    params = dict(runtime["batch_provider"]["params"])
    density = float(params["emitter_density_um2"] if emitter_density_um2 is None else emitter_density_um2)
    if density <= 0.0:
        raise ValueError("emitter_density_um2 must be positive")
    requested_samples = int(params["batch_size"] if sample_count is None else sample_count)
    if requested_samples <= 0:
        raise ValueError("sample_count must be positive")
    carrier_complex = None
    if shared_carrier_path is not None:
        with np.load(Path(shared_carrier_path).resolve(), allow_pickle=False) as payload:
            key = "carrier_complex" if "carrier_complex" in payload else "complex_pupil"
            carrier_complex = torch.from_numpy(np.asarray(payload[key], dtype=np.complex64))
        if tuple(carrier_complex.shape) != (int(params["npupil"]), int(params["npupil"])):
            raise ValueError("Shared carrier shape must match the Neptune pupil sampling")
    steps_per_epoch = int(math.ceil(requested_samples / int(params["batch_size"])))
    sequence_count = steps_per_epoch if shared_carrier_path is not None else 1
    z_range = (-2.0, 2.0)
    dh_overrides: dict[str, Any] = {}
    if shared_carrier_path is not None:
        with np.load(prepared_path, allow_pickle=False) as payload:
            field_shape = tuple(int(value) for value in payload["zernike_maps_nm"].shape[1:])
        z_range_nm = calibrated_label_z_range_nm(33.3, 3962.7)
        reference_z_range_nm = tuple(float(value) * 1000.0 for value in params["z_range"])
        z_range = tuple(float(value) / 1000.0 for value in z_range_nm)
        support = scaled_odd_psf_support(
            source_support_px=31,
            source_pixel_size_nm=200.0,
            target_pixel_size_nm_xy=(float(params["pixel_size_nm_x"]), float(params["pixel_size_nm_y"])),
        )
        origins = build_sliding_window_origin_bank(
            field_width_px=field_shape[1],
            field_height_px=field_shape[0],
            roi_width_px=int(params["width"]),
            roi_height_px=int(params["height"]),
            stride_px=int(params["field_origin_stride_px"]),
        )
        sequence_count = len(origins)
        dh_overrides = {
            "na": 1.27,
            "refmed": 1.33,
            "refcov": 1.33,
            "refimm": 1.33,
            "vector_psf_size": support,
            "zemit0": 1998.0,
            "lut_z_steps": lut_steps_preserving_reference_spacing(
                reference_z_range_nm=reference_z_range_nm,
                reference_z_steps=int(params["lut_z_steps"]),
                requested_z_range_nm=z_range_nm,
            ),
        }
    params.update(
        {
            "steps_per_epoch": steps_per_epoch,
            "sequence_count": sequence_count,
            "cached_window_max_gpu_sequences": 1,
            "simulation_output_device": "renderer",
            "z_range": z_range,
            "z_scale": 2.0,
            "emitter_density_um2": density,
            "domain_count": 1,
            "domain_balance_mode": "fixed",
            "append_domain_onehot": False,
            "condition_dim": int(params["condition_feature_dim"]),
            "pupil_carrier_complex": carrier_complex,
            "dual_domain_coeff_maps": (
                {
                    "name": "double_helix" if carrier_complex is not None else "double_helix_w5434",
                    "coeff_maps_npz": str(prepared_path),
                },
            ),
            **dh_overrides,
        }
    )
    return OnlineBatchProviderConfig(**params)


def selected_z_planes_nm() -> np.ndarray:
    return np.asarray((-1964.7, -1500.0, -1000.0, -500.0, 0.0, 500.0, 1000.0, 1500.0, 1964.7), dtype=np.float32)


def physicalize_legacy_targets(
    normalized_targets: np.ndarray,
    *,
    photon_scale: float,
    z_scale_um: float,
) -> np.ndarray:
    targets = np.asarray(normalized_targets, dtype=np.float32)
    if targets.shape[-1] != 4:
        raise ValueError("Legacy targets must end in (photons, x_px, y_px, z)")
    return np.stack(
        (
            targets[..., 1],
            targets[..., 2],
            targets[..., 3] * float(z_scale_um),
            targets[..., 0] * float(photon_scale),
        ),
        axis=-1,
    ).astype(np.float32, copy=False)


def orthogonal_max_projections(volume_zyx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    volume = np.asarray(volume_zyx)
    if volume.ndim != 3:
        raise ValueError("PSF volume must have shape (Z,Y,X)")
    return volume.max(axis=1), volume.max(axis=2)


def _noll_indices(mode_order: np.ndarray) -> list[int]:
    mode_to_noll = {noll_to_nm(index): index for index in range(1, 128)}
    return [mode_to_noll[tuple(int(value) for value in row)] for row in mode_order]


def _load_prepared_zmap(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        maps_nm = np.asarray(payload["zernike_maps_nm"], dtype=np.float32)
        mode_order = np.asarray(payload["mode_order"], dtype=np.int64)
    return maps_nm, mode_order


def _render_psf_volume(
    coefficient_nm: np.ndarray,
    mode_order: np.ndarray,
    z_nm: np.ndarray,
    *,
    contract: ReferenceSimulationContract,
    device: torch.device,
    carrier_complex: np.ndarray | None = None,
) -> np.ndarray:
    context = build_vector_psf_context(
        NA=contract.numerical_aperture,
        wavelength_nm=contract.wavelength_nm,
        pixel_size_nm_x=contract.pixel_size_nm_xy[0],
        pixel_size_nm_y=contract.pixel_size_nm_xy[1],
        noll_indices=_noll_indices(mode_order),
        params=VectorPSFParams(
            npupil=contract.pupil_size_px,
            psf_size=contract.psf_support_px,
            refmed=contract.medium_refractive_index,
            refcov=contract.medium_refractive_index,
            refimm=contract.medium_refractive_index,
            objstage0=0.0,
            otf_rescale_xy=(0.0, 0.0),
            zemit0=None,
            batch_size=96,
        ),
        device=device,
    )
    coefficients = torch.as_tensor(coefficient_nm, device=device, dtype=torch.float32)
    coefficients_rad = (
        coefficients[None]
        * (2.0 * math.pi / contract.wavelength_nm)
        * context.normfac[None]
    ).expand(len(z_nm), -1)
    return np.asarray(
        render_vector_psf_bank(
            context,
            coefficients_rad,
            np.asarray(z_nm, dtype=np.float32) * 1e-9,
            pupil_carrier_complex=carrier_complex,
            out_size=contract.psf_support_px,
            batch_size=96,
        ),
        dtype=np.float32,
    )


def _pupil_wavefront(
    coefficient_nm: np.ndarray,
    mode_order: np.ndarray,
    *,
    npupil: int,
    wavelength_nm: float,
    device: torch.device,
    carrier_complex: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    step = 2.0 / float(npupil)
    axis = torch.arange(-1.0 + step / 2.0, 1.0, step, dtype=torch.float32, device=device)
    x_pupil, y_pupil = torch.meshgrid(axis, axis, indexing="xy")
    modes = tuple(tuple(int(value) for value in row) for row in mode_order)
    basis = evaluate_normalized_zernike(modes, x_pupil, y_pupil)
    opd_nm = torch.einsum(
        "c,cyx->yx",
        torch.as_tensor(coefficient_nm, dtype=torch.float32, device=device),
        basis,
    )
    aperture = x_pupil.square() + y_pupil.square() < 1.0
    pupil = torch.exp(2j * math.pi * opd_nm / float(wavelength_nm))
    if carrier_complex is not None:
        pupil = pupil * torch.as_tensor(carrier_complex, dtype=torch.complex64, device=device)
    phase = torch.angle(pupil)
    return (
        opd_nm.detach().cpu().numpy(),
        phase.detach().cpu().numpy(),
        aperture.detach().cpu().numpy(),
    )


def _pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _save_wavefront_figure(
    opd_nm: np.ndarray,
    phase_rad: np.ndarray,
    aperture: np.ndarray,
    path: Path,
) -> None:
    plt = _pyplot()
    masked_opd = np.ma.masked_where(~aperture, opd_nm)
    masked_phase = np.ma.masked_where(~aperture, phase_rad)
    figure, axes = plt.subplots(1, 2, figsize=(10.2, 4.4), constrained_layout=True)
    opd_limit = float(np.max(np.abs(masked_opd)))
    first = axes[0].imshow(masked_opd, cmap="RdBu_r", vmin=-opd_limit, vmax=opd_limit, origin="lower")
    second = axes[1].imshow(masked_phase, cmap="twilight", vmin=-math.pi, vmax=math.pi, origin="lower")
    axes[0].set_title("Field-dependent residual OPD")
    axes[1].set_title("Shared carrier + residual phase")
    figure.colorbar(first, ax=axes[0], label="OPD (nm)", shrink=0.85)
    figure.colorbar(second, ax=axes[1], label="Phase (rad)", shrink=0.85)
    for axis in axes:
        axis.set_xlabel("Pupil x")
        axis.set_ylabel("Pupil y")
        axis.set_xticks([])
        axis.set_yticks([])
    figure.savefig(path, dpi=300, facecolor="white")
    plt.close(figure)


def _save_psf_slices_figure(volume: np.ndarray, z_nm: np.ndarray, path: Path) -> None:
    plt = _pyplot()
    selected = selected_z_planes_nm()
    indices = [int(np.argmin(np.abs(z_nm - target))) for target in selected]
    figure, axes = plt.subplots(3, 3, figsize=(9.2, 9.0), constrained_layout=True)
    for axis, index in zip(axes.flat, indices, strict=True):
        plane = volume[index] / max(float(volume[index].max()), 1e-12)
        axis.imshow(plane, cmap="magma", vmin=0.0, vmax=1.0, origin="lower")
        axis.set_title(f"z = {z_nm[index]:+.0f} nm")
        axis.set_xticks([])
        axis.set_yticks([])
    figure.suptitle(f"1000-step field-dependent double-helix PSF, Neptune LUT{volume.shape[-1]}")
    figure.savefig(path, dpi=300, facecolor="white")
    plt.close(figure)


def _save_axial_figure(
    volume: np.ndarray,
    z_nm: np.ndarray,
    *,
    contract: ReferenceSimulationContract,
    path: Path,
) -> None:
    plt = _pyplot()
    from matplotlib.colors import PowerNorm

    xz, yz = orthogonal_max_projections(volume)
    half_x_um = volume.shape[2] * contract.pixel_size_nm_xy[0] / 2000.0
    half_y_um = volume.shape[1] * contract.pixel_size_nm_xy[1] / 2000.0
    extent_xz = (-half_x_um, half_x_um, z_nm[0] / 1000.0, z_nm[-1] / 1000.0)
    extent_yz = (-half_y_um, half_y_um, z_nm[0] / 1000.0, z_nm[-1] / 1000.0)
    figure, axes = plt.subplots(1, 2, figsize=(10.2, 5.8), constrained_layout=True)
    axes[0].imshow(xz, cmap="magma", norm=PowerNorm(gamma=0.45), origin="lower", extent=extent_xz, aspect="auto")
    axes[1].imshow(yz, cmap="magma", norm=PowerNorm(gamma=0.45), origin="lower", extent=extent_yz, aspect="auto")
    axes[0].set_title("XZ max projection")
    axes[1].set_title("YZ max projection")
    axes[0].set_xlabel("x (um)")
    axes[1].set_xlabel("y (um)")
    for axis in axes:
        axis.set_ylabel("z (um)")
        axis.axhline(0.0, color="white", linewidth=0.7, alpha=0.7)
    figure.savefig(path, dpi=300, facecolor="white")
    plt.close(figure)


def _save_training_frames_figure(
    frames_adu: np.ndarray,
    targets_physical: np.ndarray,
    mask: np.ndarray,
    path: Path,
    *,
    emitter_density_um2: float,
) -> None:
    plt = _pyplot()
    sample_indices = np.linspace(0, len(frames_adu) - 1, 4, dtype=int)
    selected = frames_adu[sample_indices]
    vmin = float(np.percentile(selected, 1.0))
    vmax = float(np.percentile(selected, 99.8))
    figure, axes = plt.subplots(4, 3, figsize=(9.0, 11.5), constrained_layout=True)
    for row, sample_idx in enumerate(sample_indices.tolist()):
        for channel in range(3):
            axis = axes[row, channel]
            axis.imshow(frames_adu[sample_idx, channel], cmap="gray", vmin=vmin, vmax=vmax, origin="lower")
            axis.set_title(f"sample {sample_idx}, t{channel - 1:+d}")
            axis.set_xticks([])
            axis.set_yticks([])
        active = targets_physical[sample_idx][mask[sample_idx]]
        axes[row, 1].scatter(active[:, 0], active[:, 1], s=18, facecolors="none", edgecolors="#00d7ff", linewidths=0.7)
    figure.suptitle(
        f"Neptune test simulation: {emitter_density_um2:g} emitter/um^2 "
        "(cyan = center-frame target)"
    )
    figure.savefig(path, dpi=300, facecolor="white")
    plt.close(figure)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    return value


def run_training_simulation(
    *,
    reference_config_path: str | Path,
    source_zmap_path: str | Path,
    source_metrics_path: str | Path,
    output_dir: str | Path,
    seed: int,
    device: str,
    emitter_density_um2: float = 1.0,
    shared_carrier_path: str | Path | None = None,
    sample_count: int = 96,
) -> SimulationOutputs:
    torch_device = torch.device(device)
    if torch_device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Formal double-helix training simulation requires an available CUDA device")

    output = Path(output_dir).resolve()
    arrays_dir = output / "arrays"
    figures_dir = output / "figures"
    stacks_dir = output / "stacks"
    metadata_dir = output / "metadata"
    for directory in (arrays_dir, figures_dir, stacks_dir, metadata_dir):
        directory.mkdir(parents=True, exist_ok=True)

    reference_path = Path(reference_config_path).resolve()
    source_zmap = Path(source_zmap_path).resolve()
    source_metrics = Path(source_metrics_path).resolve()
    shared_carrier = None if shared_carrier_path is None else Path(shared_carrier_path).resolve()
    contract = load_reference_contract(reference_path)
    prepared_path = arrays_dir / "residual21_zero_defocus_zernike_maps_nm.npz"
    prepared = prepare_zero_defocus_zmap(
        source_zmap,
        prepared_path,
        source_pixel_size_nm=200.0 if shared_carrier is not None else None,
        target_pixel_size_nm_xy=contract.pixel_size_nm_xy if shared_carrier is not None else None,
    )
    provider_config = build_export_provider_config(
        reference_path,
        prepared_path,
        seed=seed,
        emitter_density_um2=emitter_density_um2,
        shared_carrier_path=shared_carrier,
        sample_count=sample_count,
    )

    torch.manual_seed(int(seed))
    torch.cuda.manual_seed_all(int(seed))
    provider = build_online_batch_provider(provider_config)
    loc_batches = [training_batch.inputs for training_batch in provider(1)]
    frames_adu = np.concatenate(
        [
            (batch.model_input[0] if isinstance(batch.model_input, tuple) else batch.model_input)
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32, copy=False)
            for batch in loc_batches
        ],
        axis=0,
    )[:sample_count]
    target_width = max(int(batch.pxyz_tar.shape[1]) for batch in loc_batches)
    target_parts = []
    mask_parts = []
    for batch in loc_batches:
        targets = batch.pxyz_tar.detach().cpu().numpy().astype(np.float32, copy=False)
        batch_mask = batch.mask_tar.detach().cpu().numpy().astype(bool, copy=False)
        if targets.shape[1] < target_width:
            padding = target_width - targets.shape[1]
            targets = np.pad(targets, ((0, 0), (0, padding), (0, 0)))
            batch_mask = np.pad(batch_mask, ((0, 0), (0, padding)))
        target_parts.append(targets)
        mask_parts.append(batch_mask)
    targets_normalized = np.concatenate(target_parts, axis=0)[:sample_count]
    mask = np.concatenate(mask_parts, axis=0)[:sample_count]
    targets_physical = physicalize_legacy_targets(
        targets_normalized,
        photon_scale=float(provider_config.photon_scale),
        z_scale_um=float(provider_config.z_scale),
    )

    batch_arrays_path = arrays_dir / "simulated_training_batch.npz"
    np.savez_compressed(
        batch_arrays_path,
        frames_adu=frames_adu,
        detect_target=np.concatenate(
            [batch.detect_tar.detach().cpu().numpy().astype(np.float32) for batch in loc_batches], axis=0
        )[:sample_count],
        background_target=np.concatenate(
            [batch.bkg_tar.detach().cpu().numpy().astype(np.float32) for batch in loc_batches], axis=0
        )[:sample_count],
        emitter_xy_z_um_photons=targets_physical,
        emitter_mask=mask,
        legacy_training_targets_normalized=targets_normalized,
    )
    training_tiff_path = stacks_dir / "simulated_training_triplets_adu.tif"
    tifffile.imwrite(
        training_tiff_path,
        frames_adu,
        photometric="minisblack",
        metadata={"axes": "TCYX"},
    )
    gamma_roi_bank_tiff_path = export_first_full_roi_frames(
        frames_adu,
        stacks_dir / "simulated_training_first100_full_roi_adu.tif",
    )
    (metadata_dir / "online_batch_metadata.json").write_text(
        json.dumps(_jsonable([batch.metadata for batch in loc_batches]), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    maps_nm, mode_order = _load_prepared_zmap(prepared_path)
    carrier_complex = None
    if shared_carrier is not None:
        with np.load(shared_carrier, allow_pickle=False) as payload:
            key = "carrier_complex" if "carrier_complex" in payload else "complex_pupil"
            carrier_complex = np.asarray(payload[key], dtype=np.complex64)
    coefficient_nm = maps_nm[:, maps_nm.shape[1] // 2, maps_nm.shape[2] // 2]
    z_nm = np.linspace(
        float(provider_config.z_range[0]) * 1000.0,
        float(provider_config.z_range[1]) * 1000.0,
        int(provider_config.lut_z_steps),
        dtype=np.float32,
    )
    model_z_nm = model_z_from_label_z_nm(
        z_nm,
        model_center_nm=float(provider_config.zemit0 or 0.0),
    ).astype(np.float32)
    dh_contract = replace(
        contract,
        psf_support_px=int(provider_config.vector_psf_size),
        numerical_aperture=float(provider_config.na),
        medium_refractive_index=float(provider_config.refmed),
    )
    psf_volume = _render_psf_volume(
        coefficient_nm,
        mode_order,
        model_z_nm,
        contract=dh_contract,
        device=torch_device,
        carrier_complex=carrier_complex,
    )
    psf_volume_path = arrays_dir / "double_helix_psf_volume_calibrated_z1965.npz"
    np.savez_compressed(
        psf_volume_path,
        psf_unit_flux=psf_volume,
        z_label_nm=z_nm,
        z_model_nm=model_z_nm,
        z_model_offset_nm=np.asarray(provider_config.zemit0, dtype=np.float32),
        coefficient_nm=coefficient_nm,
        mode_order=mode_order,
    )
    tifffile.imwrite(
        stacks_dir / "double_helix_psf_volume_calibrated_z1965.tif",
        psf_volume,
        photometric="minisblack",
        metadata={"axes": "ZYX"},
    )

    opd_nm, phase_rad, aperture = _pupil_wavefront(
        coefficient_nm,
        mode_order,
        npupil=contract.pupil_size_px,
        wavelength_nm=contract.wavelength_nm,
        device=torch_device,
        carrier_complex=carrier_complex,
    )
    np.savez_compressed(
        arrays_dir / "double_helix_pupil_wavefront.npz",
        opd_nm=opd_nm,
        wrapped_phase_rad=phase_rad,
        pupil_mask=aperture,
    )
    figure_paths = (
        figures_dir / "double_helix_pupil_wavefront.png",
        figures_dir / "double_helix_psf_z_slices.png",
        figures_dir / "double_helix_psf_xz_yz.png",
        figures_dir / "simulated_training_frames.png",
    )
    _save_wavefront_figure(opd_nm, phase_rad, aperture, figure_paths[0])
    _save_psf_slices_figure(psf_volume, z_nm, figure_paths[1])
    _save_axial_figure(psf_volume, z_nm, contract=dh_contract, path=figure_paths[2])
    _save_training_frames_figure(
        frames_adu,
        targets_physical,
        mask,
        figure_paths[3],
        emitter_density_um2=emitter_density_um2,
    )

    active_targets = targets_physical[mask]
    manifest = {
        "schema_version": "double_helix_neptune_calibrated_lut_simulation.v3",
        "reference_run": str(reference_path.parents[1]),
        "reference_resolved_config": str(reference_path),
        "reference_resolved_config_sha256": _sha256(reference_path),
        "source_residual_zmap": str(source_zmap),
        "source_residual_zmap_sha256": _sha256(source_zmap),
        "source_calibration_metadata": str(source_metrics),
        "source_calibration_metadata_sha256": _sha256(source_metrics),
        "source_shared_carrier": None if shared_carrier is None else str(shared_carrier),
        "source_shared_carrier_sha256": None if shared_carrier is None else _sha256(shared_carrier),
        "reference_contract": asdict(contract),
        "requested_overrides": {
            "psf_model": "fixed independent shared DH carrier + 21 field-dependent residual Zernike modes",
            "physical_update_total_steps": 1000,
            "z_range_um": list(provider_config.z_range),
            "photon_mean": 20000.0,
            "photon_std": 1000.0,
            "emitter_density_um2": provider_config.emitter_density_um2,
        },
        "dimension_contract": {
            "training_image_roi_yx_px": [provider_config.height, provider_config.width],
            "frames_per_sample": provider_config.channels,
            "batch_size": provider_config.batch_size,
            "test_sample_count": int(frames_adu.shape[0]),
            "psf_support_yx_px": [provider_config.vector_psf_size, provider_config.vector_psf_size],
            "pupil_sampling_yx_px": [provider_config.npupil, provider_config.npupil],
        },
        "simulation_contract": {
            "backend": provider_config.simulation_backend,
            "batch_strategy": provider_config.batch_strategy,
            "lut_field_mode": provider_config.lut_field_mode,
            "lut_storage_dtype": provider_config.lut_storage_dtype,
            "camera_output_domain": "raw ADU after Poisson camera sampling",
            "photon_sampling": {
                "distribution": "normal emitter intensity before lifetime-overlap scaling",
                "mean": provider_config.photon_mean,
                "std": provider_config.photon_sigma,
                "clip": provider_config.photon_range,
            },
            "defocus_2_0_nm": 0.0,
            "shared_carrier_applied": carrier_complex is not None,
            "source_mode_count": prepared.source_mode_count,
            "exported_mode_count_including_zero_defocus": prepared.exported_mode_count,
            "source_field_shape_hw": prepared.source_field_shape_hw,
            "resampled_field_shape_hw": prepared.field_shape_hw,
            "source_pixel_size_nm": prepared.source_pixel_size_nm,
            "target_pixel_size_nm_xy": prepared.target_pixel_size_nm_xy,
            "z_label_range_nm": [float(z_nm[0]), float(z_nm[-1])],
            "z_model_range_nm": [float(model_z_nm[0]), float(model_z_nm[-1])],
            "z_model_offset_nm": provider_config.zemit0,
            "lut_z_steps": provider_config.lut_z_steps,
            "lut_z_spacing_nm": float((z_nm[-1] - z_nm[0]) / max(len(z_nm) - 1, 1)),
            "field_origin_sampling_mode": provider_config.field_origin_sampling_mode,
            "field_origin_sequence_count": provider_config.sequence_count,
            "dh_optics": {
                "NA": provider_config.na,
                "refmed": provider_config.refmed,
                "refcov": provider_config.refcov,
                "refimm": provider_config.refimm,
                "calibrated_support_px": 31,
                "calibrated_pixel_size_nm": 200.0,
                "simulation_support_px": provider_config.vector_psf_size,
            },
        },
        "formal_runtime": {
            "device": str(torch_device),
            "cuda_available": bool(torch.cuda.is_available()),
            "gpu_name": torch.cuda.get_device_name(torch_device),
            "torch_version": torch.__version__,
            "seed": int(seed),
        },
        "simulated_test_statistics": {
            "active_emitters": int(mask.sum()),
            "effective_center_frame_photon_mean_after_lifetime_overlap": float(active_targets[:, 3].mean()),
            "effective_center_frame_photon_std_after_lifetime_overlap": float(active_targets[:, 3].std()),
            "z_min_um": float(active_targets[:, 2].min()),
            "z_max_um": float(active_targets[:, 2].max()),
            "raw_adu_min": float(frames_adu.min()),
            "raw_adu_max": float(frames_adu.max()),
        },
        "optical_contract_note": (
            "The accepted 1000-step Microscope1 DH pupil keeps its calibrated NA and refractive indices; "
            "the FD map and PSF support are resampled onto the Neptune camera grid before the strict LUT pipeline."
        ),
        "artifacts": {
            "batch_arrays": str(batch_arrays_path),
            "training_tiff": str(training_tiff_path),
            "gamma_roi_bank_tiff": str(gamma_roi_bank_tiff_path),
            "psf_volume": str(psf_volume_path),
            "figures": [str(path) for path in figure_paths],
        },
    }
    manifest_path = metadata_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(_jsonable(manifest), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return SimulationOutputs(
        output_dir=output,
        manifest_path=manifest_path,
        batch_arrays_path=batch_arrays_path,
        training_tiff_path=training_tiff_path,
        gamma_roi_bank_tiff_path=gamma_roi_bank_tiff_path,
        psf_volume_path=psf_volume_path,
        figure_paths=figure_paths,
    )


def parse_args() -> argparse.Namespace:
    root = PROJECT_ROOT.parent
    unity_root = PROJECT_ROOT
    parser = argparse.ArgumentParser(description="Export a Neptune-contract double-helix test simulation.")
    parser.add_argument(
        "--reference-config",
        type=Path,
        default=root
        / "results/microtube_right_benchmark_20260711/runs/neptune_v03_fd_nat"
        / "training_dual_anchor99_gamma10_conditionfix/resolved_config.yaml",
    )
    parser.add_argument(
        "--source-zmap",
        type=Path,
        default=unity_root
        / "output/double_helix/microscope1_shared_carrier_residual21_physical_update_first100_totalsteps1000_20260725"
        / "physical_update/arrays/alternating_full_roi_zernike_maps_nm.npz",
    )
    parser.add_argument(
        "--source-metrics",
        type=Path,
        default=unity_root
        / "output/double_helix/microscope1_shared_carrier_residual21_physical_update_first100_totalsteps1000_20260725"
        / "physical_update/metadata/manifest.json",
    )
    parser.add_argument(
        "--shared-carrier",
        type=Path,
        default=unity_root
        / "output/double_helix/microscope1_shared_carrier_residual21_20260725"
        / "calibration/arrays/shared_double_helix_carrier.npz",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=unity_root / "output/double_helix/microscope1_1000step_neptune_sim_testsets_20260725/density1p0",
    )
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--emitter-density-um2", type=float, default=1.0)
    parser.add_argument("--sample-count", type=int, default=96)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    outputs = run_training_simulation(
        reference_config_path=args.reference_config,
        source_zmap_path=args.source_zmap,
        source_metrics_path=args.source_metrics,
        output_dir=args.output_dir,
        seed=args.seed,
        device=args.device,
        emitter_density_um2=args.emitter_density_um2,
        shared_carrier_path=args.shared_carrier,
        sample_count=args.sample_count,
    )
    print(
        json.dumps(
            {
                "output_dir": str(outputs.output_dir),
                "manifest": str(outputs.manifest_path),
                "figures": [str(path) for path in outputs.figure_paths],
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


__all__ = [
    "PreparedZernikeMaps",
    "ReferenceSimulationContract",
    "SimulationOutputs",
    "build_export_provider_config",
    "export_first_full_roi_frames",
    "load_reference_contract",
    "orthogonal_max_projections",
    "physicalize_legacy_targets",
    "prepare_zero_defocus_zmap",
    "run_training_simulation",
    "selected_z_planes_nm",
]


if __name__ == "__main__":
    raise SystemExit(main())
