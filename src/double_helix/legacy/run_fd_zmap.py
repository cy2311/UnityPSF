from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import tifffile

from .._paths import PROJECT_ROOT as V04_ROOT
from ..dataset import Microscope1Dataset
from ..field_fit import FieldFitConfig, FieldFitResult, fit_field_dependent_z
from ..local_fit import LocalZFit, OracleObservations, estimate_local_z, harvest_oracle_patches
from ..lut import CalibrationLUT


DEFAULT_DATASET_ROOT = Path(
    "/home/guest/Others/main/race/datasets/training_sets/double_helix/"
    "Simulated_datasets_Microscope1"
)
DEFAULT_OUTPUT_DIR = Path("output/double_helix_microscope1_zmap")


@dataclass(frozen=True)
class RunConfig:
    dataset_root: Path = DEFAULT_DATASET_ROOT
    output_dir: Path = DEFAULT_OUTPUT_DIR
    min_neighbor_distance_px: float = 31.0
    max_emitters: int | None = None
    search_radius_planes: int = 3
    min_ncc: float = 0.94
    bootstrap_iterations: int = 200
    random_seed: int = 20260723


@dataclass(frozen=True)
class RunResult:
    output_dir: Path
    summary_path: Path
    zmap_path: Path
    diagnostics_path: Path
    field_accepted: bool


def run_pipeline(config: RunConfig) -> RunResult:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = Microscope1Dataset(config.dataset_root)
    contract = dataset.validate()
    lut = CalibrationLUT.from_array(
        dataset.read_calibration(),
        z_step_nm=dataset.z_step_nm,
        z_index_origin=dataset.z_index_origin,
        z_sign=dataset.z_sign,
    )
    observations = harvest_oracle_patches(
        dataset,
        min_neighbor_distance_px=config.min_neighbor_distance_px,
        max_emitters=config.max_emitters,
    )
    local_fit = estimate_local_z(
        observations.patches,
        observations.z_gt_nm,
        lut,
        search_radius_planes=config.search_radius_planes,
    )
    quality = _quality_mask(observations, local_fit, lut, min_ncc=config.min_ncc)
    if int(quality.sum()) < 100:
        raise ValueError(f"Only {int(quality.sum())} oracle observations passed quality filters.")

    field_config = FieldFitConfig(
        bootstrap_iterations=config.bootstrap_iterations,
        random_seed=config.random_seed,
    )
    field_fit = fit_field_dependent_z(
        observations.x_px[quality],
        observations.y_px[quality],
        observations.z_gt_nm[quality],
        local_fit.z_residual_nm[quality],
        observations.frame_index[quality],
        config=field_config,
    )

    summary = _build_summary(config, contract, observations, local_fit, quality, field_fit, field_config)
    _write_local_fits(output_dir / "local_fits.npz", observations, local_fit, quality)
    zmap_path = output_dir / "fd_zmap.npz"
    _write_zmaps(zmap_path, output_dir, field_fit, summary)
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    diagnostics_path = output_dir / "fd_zmap_diagnostics.png"
    _render_diagnostics(field_fit, diagnostics_path)
    return RunResult(
        output_dir=output_dir,
        summary_path=summary_path,
        zmap_path=zmap_path,
        diagnostics_path=diagnostics_path,
        field_accepted=field_fit.field_accepted,
    )


def _quality_mask(
    observations: OracleObservations,
    local_fit: LocalZFit,
    lut: CalibrationLUT,
    *,
    min_ncc: float,
) -> np.ndarray:
    z_min = float(lut.z_nm[1])
    z_max = float(lut.z_nm[-2])
    return (
        (local_fit.ncc >= float(min_ncc))
        & (local_fit.photons_adu > 0.0)
        & (observations.z_gt_nm >= z_min)
        & (observations.z_gt_nm <= z_max)
    )


def _build_summary(
    config: RunConfig,
    contract: Any,
    observations: OracleObservations,
    local_fit: LocalZFit,
    quality: np.ndarray,
    field_fit: FieldFitResult,
    field_config: FieldFitConfig,
) -> dict[str, Any]:
    selected_residual = local_fit.z_residual_nm[quality]
    active = field_fit.active_fov_mask
    candidate = field_fit.candidate_map_nm
    uncertainty = field_fit.uncertainty_std_nm
    return {
        "dataset": {
            "root": str(Path(config.dataset_root).resolve()),
            "frame_shape": list(contract.frame_shape),
            "gt_rows": int(contract.gt_rows),
            "pixel_size_nm": float(contract.pixel_size_nm),
            "frame_convention": "GT Frame is 1-based; TIFF page and exported frame_index are 0-based",
            "xy_mapping": "x_px=x_nm/200+15; y_px=y_nm/200+15; centers use floor(value+0.5)",
        },
        "calibration": {
            "shape": list(contract.calibration_shape),
            "z_step_nm": float(contract.z_step_nm),
            "z_index_origin": float(contract.z_index_origin),
            "z_sign": int(contract.z_sign),
            "plane_z_formula": "z_nm = z_sign * (plane_index_0based + z_index_origin) * z_step_nm",
            "preserves_z_dependent_midpoint_drift": True,
        },
        "method": {
            "training_mode": "oracle_supervised",
            "psf_model": "31x31 calibration LUT with continuous axial interpolation and Fourier XY alignment",
            "likelihood": "Gaussian raw-ADU profile fit",
            "field_model": "degree-2 zero-mean Legendre scalar axial residual after degree-3 global Z bias",
            "selection_gate": "frame-held-out and spatial-block-held-out bootstrap 95% CI lower bounds must exceed zero",
            "source_simulation_expected_field_dependence": "none described in the publication",
        },
        "map_semantics": {
            "fd_z_offset_nm": "selected apparent axial bias z_lut - z_GT; zero when field gate rejects",
            "candidate_map_nm": "ungated fitted apparent axial bias",
            "fd_z_correction_nm": "negative of fd_z_offset_nm",
            "correction_formula": "z_corrected = z_lut - fd_z_offset_nm",
            "axes": "YX",
            "gauge": "mean zero over the 120x120 active FOV",
            "padding": "15 pixels on every side; map values are extrapolated but active_fov_mask marks validity",
        },
        "selection": {
            "harvested_emitters": int(observations.patches.shape[0]),
            "quality_emitters": int(quality.sum()),
            "min_neighbor_distance_px": float(config.min_neighbor_distance_px),
            "min_ncc": float(config.min_ncc),
        },
        "local_fit": {
            "median_ncc": float(np.median(local_fit.ncc[quality])),
            "p10_ncc": float(np.percentile(local_fit.ncc[quality], 10.0)),
            "median_abs_z_error_nm": float(np.median(np.abs(selected_residual))),
            "z_rmse_nm": float(np.sqrt(np.mean(selected_residual**2))),
            "median_z_bias_nm": float(np.median(selected_residual)),
            "median_photons_adu": float(np.median(local_fit.photons_adu[quality])),
            "median_background_adu": float(np.median(local_fit.background_adu[quality])),
            "median_residual_variance_adu2": float(np.median(local_fit.residual_variance_adu2[quality])),
            "empirical_blank_variance_reference_adu2": 2504.757,
        },
        "field_fit": {
            "accepted": bool(field_fit.field_accepted),
            "metrics": field_fit.metrics,
            "spatial_terms_px_py": [list(term) for term in field_fit.spatial_terms],
            "spatial_coefficients_nm": field_fit.spatial_coefficients_nm.tolist(),
            "global_z_coefficients_nm": field_fit.global_z_coefficients_nm.tolist(),
            "candidate_active_rms_nm": float(np.sqrt(np.mean(candidate[active] ** 2))),
            "candidate_active_peak_to_peak_nm": float(np.ptp(candidate[active])),
            "uncertainty_active_median_nm": float(np.median(uncertainty[active])),
            "uncertainty_active_max_nm": float(np.max(uncertainty[active])),
            **_adjacent_jump_metrics(candidate, active),
        },
        "config": _jsonable(asdict(config)),
        "field_config": _jsonable(asdict(field_config)),
    }


def _write_local_fits(
    path: Path,
    observations: OracleObservations,
    local_fit: LocalZFit,
    quality: np.ndarray,
) -> None:
    np.savez_compressed(
        path,
        gt_index=observations.gt_index,
        frame_index=observations.frame_index,
        x_px=observations.x_px,
        y_px=observations.y_px,
        z_gt_nm=observations.z_gt_nm,
        nearest_neighbor_px=observations.nearest_neighbor_px,
        z_fit_nm=local_fit.z_fit_nm,
        z_residual_nm=local_fit.z_residual_nm,
        ncc=local_fit.ncc,
        photons_adu=local_fit.photons_adu,
        background_adu=local_fit.background_adu,
        residual_variance_adu2=local_fit.residual_variance_adu2,
        quality_mask=quality,
    )


def _write_zmaps(path: Path, output_dir: Path, result: FieldFitResult, summary: dict[str, Any]) -> None:
    correction = -result.fd_z_offset_nm
    np.savez_compressed(
        path,
        fd_z_offset_nm=result.fd_z_offset_nm,
        fd_z_correction_nm=correction,
        candidate_map_nm=result.candidate_map_nm,
        uncertainty_std_nm=result.uncertainty_std_nm,
        active_fov_mask=result.active_fov_mask,
        spatial_terms_px_py=np.asarray(result.spatial_terms, dtype=np.int64),
        spatial_coefficients_nm=result.spatial_coefficients_nm,
        global_z_coefficients_nm=result.global_z_coefficients_nm,
        field_accepted=np.asarray(result.field_accepted),
        metadata_json=np.asarray(json.dumps(summary, sort_keys=True)),
    )
    description = json.dumps(summary["map_semantics"], sort_keys=True)
    tifffile.imwrite(output_dir / "fd_z_offset_nm.tif", result.fd_z_offset_nm, description=description)
    tifffile.imwrite(output_dir / "fd_z_correction_nm.tif", correction, description=description)
    tifffile.imwrite(output_dir / "fd_z_offset_candidate_nm.tif", result.candidate_map_nm, description=description)
    tifffile.imwrite(output_dir / "fd_z_uncertainty_std_nm.tif", result.uncertainty_std_nm, description=description)
    tifffile.imwrite(output_dir / "active_fov_mask.tif", result.active_fov_mask.astype(np.uint8), description="1=active FOV")


def _render_diagnostics(result: FieldFitResult, path: Path) -> None:
    cache_dir = V04_ROOT / ".local" / "cache" / "matplotlib"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    active = result.active_fov_mask
    candidate_active = result.candidate_map_nm[active]
    limit = max(float(np.percentile(np.abs(candidate_active), 99.0)), 1.0)
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
    extent_um = (0.0, 30.0, 30.0, 0.0)
    final_display = np.where(active, result.fd_z_offset_nm, np.nan)
    candidate_display = np.where(active, result.candidate_map_nm, np.nan)
    uncertainty_display = np.where(active, result.uncertainty_std_nm, np.nan)

    with plt.rc_context(
        {
            "font.family": "sans-serif",
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
        }
    ):
        fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.65), constrained_layout=True)
        panels = (
            (final_display, "Selected FD offset", "PuOr_r", norm, "Axial bias (nm)"),
            (candidate_display, "Ungated candidate", "PuOr_r", norm, "Axial bias (nm)"),
            (uncertainty_display, "Block-bootstrap uncertainty", "cividis", None, "SD (nm)"),
        )
        for label, (ax, (image, title, cmap, panel_norm, colorbar_label)) in zip(
            ("A", "B", "C"), zip(axes, panels, strict=True), strict=True
        ):
            artist = ax.imshow(image, origin="upper", extent=extent_um, cmap=cmap, norm=panel_norm)
            ax.set_title(title)
            ax.set_xlabel("X (um)")
            ax.set_ylabel("Y (um)")
            ax.text(-0.16, 1.06, label, transform=ax.transAxes, fontweight="bold", fontsize=10, va="top")
            colorbar = fig.colorbar(artist, ax=ax, fraction=0.046, pad=0.03)
            colorbar.set_label(colorbar_label)
        status = "accepted" if result.field_accepted else "rejected by held-out gate; selected map is zero"
        fig.suptitle(f"Microscope 1 double-helix FD axial map ({status})", fontsize=10)
        fig.savefig(path, dpi=300, facecolor="white")
        fig.savefig(path.with_suffix(".pdf"), facecolor="white")
        plt.close(fig)


def _adjacent_jump_metrics(values: np.ndarray, active: np.ndarray) -> dict[str, float]:
    horizontal_mask = active[:, 1:] & active[:, :-1]
    vertical_mask = active[1:, :] & active[:-1, :]
    jumps = np.concatenate(
        [
            np.abs(np.diff(values, axis=1))[horizontal_mask],
            np.abs(np.diff(values, axis=0))[vertical_mask],
        ]
    )
    return {
        "candidate_adjacent_jump_p99_nm": float(np.percentile(jumps, 99.0)),
        "candidate_adjacent_jump_max_nm": float(np.max(jumps)),
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit a field-dependent axial map for DHPSFU Microscope 1.")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--min-neighbor-distance-px", type=float, default=31.0)
    parser.add_argument("--max-emitters", type=int, default=None)
    parser.add_argument("--search-radius-planes", type=int, default=3)
    parser.add_argument("--min-ncc", type=float, default=0.94)
    parser.add_argument("--bootstrap-iterations", type=int, default=200)
    parser.add_argument("--random-seed", type=int, default=20260723)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run_pipeline(
        RunConfig(
            dataset_root=args.dataset_root,
            output_dir=args.output_dir,
            min_neighbor_distance_px=args.min_neighbor_distance_px,
            max_emitters=args.max_emitters,
            search_radius_planes=args.search_radius_planes,
            min_ncc=args.min_ncc,
            bootstrap_iterations=args.bootstrap_iterations,
            random_seed=args.random_seed,
        )
    )
    print(f"summary_path={result.summary_path}")
    print(f"zmap_path={result.zmap_path}")
    print(f"field_accepted={result.field_accepted}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["RunConfig", "RunResult", "run_pipeline"]
