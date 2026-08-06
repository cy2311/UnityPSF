from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .._paths import PROJECT_ROOT as V04_ROOT
from ..dataset import Microscope1Dataset
from ..field_gamma import FieldGammaFitConfig, FieldGammaFitResult, fit_field_gamma
from ..gamma_field import DirectGammaZernikeField
from ..local_fit import harvest_oracle_patches


DEFAULT_DATASET_ROOT = V04_ROOT.parent / "datasets/training_sets/double_helix/Simulated_datasets_Microscope1"
DEFAULT_CALIBRATION_GAMMA = V04_ROOT / "output/double_helix/microscope1/calibration/arrays/gamma_coefficients.npz"
DEFAULT_OUTPUT_DIR = V04_ROOT / "output/double_helix/microscope1/field_gamma"


@dataclass(frozen=True)
class FieldGammaOutputs:
    output_dir: Path
    gamma_path: Path
    zmap_path: Path
    candidate_gamma_path: Path
    metrics_path: Path
    manifest_path: Path
    coefficient_maps_figure_path: Path
    candidate_maps_figure_path: Path


def write_field_gamma_outputs(
    output_dir: str | Path,
    *,
    result: FieldGammaFitResult,
    config: FieldGammaFitConfig,
    carrier_path: str | Path | None = None,
) -> FieldGammaOutputs:
    root = Path(output_dir)
    arrays_dir = root / "arrays"
    figures_dir = root / "figures"
    metadata_dir = root / "metadata"
    for directory in (arrays_dir, figures_dir, metadata_dir):
        directory.mkdir(parents=True, exist_ok=True)
    mode_order = np.asarray(result.mode_order, dtype=np.int64)
    gamma_path = arrays_dir / "gamma_coefficients.npz"
    np.savez_compressed(
        gamma_path,
        gamma_nm=np.asarray(result.gamma_nm, dtype=np.float32),
        mode_order=mode_order,
        spatial_order=np.asarray(config.spatial_degree, dtype=np.int64),
    )
    candidate_gamma_path = arrays_dir / "candidate_gamma_coefficients.npz"
    np.savez_compressed(
        candidate_gamma_path,
        gamma_nm=np.asarray(result.candidate_gamma_nm, dtype=np.float32),
        mode_order=mode_order,
        spatial_order=np.asarray(config.spatial_degree, dtype=np.int64),
    )
    field = DirectGammaZernikeField(
        gamma_nm=torch.from_numpy(np.asarray(result.gamma_nm, dtype=np.float32)),
        mode_order=result.mode_order,
    )
    zmap_path = arrays_dir / "alternating_full_roi_zernike_maps_nm.npz"
    field.export_zmap(zmap_path, image_shape_hw=config.image_shape_hw)
    np.savez_compressed(
        arrays_dir / "observation_partition.npz",
        train=result.partition.train,
        frame_holdout=result.partition.frame_holdout,
        spatial_holdout=result.partition.spatial_holdout,
        block_ids=result.partition.block_ids,
    )
    metrics_path = metadata_dir / "metrics.json"
    metrics_path.write_text(json.dumps(result.metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    residual_model = carrier_path is not None
    manifest = {
        "field_accepted": result.field_accepted,
        "gamma_semantics": (
            "field-dependent residual OPD above fixed shared DH carrier"
            if residual_model
            else "direct total pupil coefficient field; no residual decomposition"
        ),
        "selected_model": (
            "shared carrier plus degree-2 residual gamma field"
            if residual_model and result.field_accepted
            else "shared carrier plus spatially constant residual gamma"
            if residual_model
            else "degree-2 direct gamma field"
            if result.field_accepted
            else "spatially constant direct gamma"
        ),
        "shared_carrier_input": str(Path(carrier_path).resolve()) if carrier_path is not None else None,
        "shared_carrier_sha256": _sha256(Path(carrier_path)) if carrier_path is not None else None,
        "source_simulation_expected_field_dependence": "none described in the publication",
        "supervision": "GT x/y/z used only to harvest pupil-calibration patches",
        "localization_contract": "not an emitter-localization result",
        "spatial_terms_px_py": [list(term) for term in result.spatial_terms],
        "config": asdict(config),
    }
    manifest_path = metadata_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    coefficient_maps_path = figures_dir / "gamma_coefficient_maps.png"
    candidate_maps_path = figures_dir / "candidate_gamma_coefficient_maps.png"
    _render_gamma_maps(
        result.gamma_nm,
        result.mode_order,
        coefficient_maps_path,
        title="Selected residual gamma maps" if residual_model else "Selected direct gamma maps",
    )
    _render_gamma_maps(
        result.candidate_gamma_nm,
        result.mode_order,
        candidate_maps_path,
        title=(
            "Candidate residual gamma maps (held-out diagnostic)"
            if residual_model
            else "Candidate direct gamma maps (held-out diagnostic)"
        ),
    )
    return FieldGammaOutputs(
        output_dir=root,
        gamma_path=gamma_path,
        zmap_path=zmap_path,
        candidate_gamma_path=candidate_gamma_path,
        metrics_path=metrics_path,
        manifest_path=manifest_path,
        coefficient_maps_figure_path=coefficient_maps_path,
        candidate_maps_figure_path=candidate_maps_path,
    )


def _render_gamma_maps(
    gamma_nm: np.ndarray,
    mode_order: tuple[tuple[int, int], ...],
    path: Path,
    *,
    title: str,
) -> None:
    cache_dir = V04_ROOT / ".local/cache/matplotlib"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    field = DirectGammaZernikeField(gamma_nm=torch.from_numpy(np.asarray(gamma_nm)), mode_order=mode_order)
    maps = field.coefficient_stack(image_shape_hw=(150, 150)).numpy()
    variation = np.ptp(maps, axis=(1, 2))
    selected = np.argsort(variation)[::-1][: min(12, maps.shape[0])]
    columns = 4
    rows = int(np.ceil(selected.size / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(11.0, 2.8 * rows), constrained_layout=True)
    axes = np.asarray(axes).reshape(-1)
    for axis, mode_index in zip(axes, selected, strict=False):
        values = maps[mode_index]
        midpoint = float(np.mean(values))
        spread = max(float(np.max(np.abs(values - midpoint))), 1e-3)
        artist = axis.imshow(
            values,
            cmap="PuOr_r",
            norm=TwoSlopeNorm(vmin=midpoint - spread, vcenter=midpoint, vmax=midpoint + spread),
            extent=(0.0, 30.0, 30.0, 0.0),
        )
        n, m = mode_order[mode_index]
        axis.set_title(f"Z({n},{m:+d}), range {variation[mode_index]:.2f} nm", fontsize=8)
        axis.set_xlabel("X (um)", fontsize=7)
        axis.set_ylabel("Y (um)", fontsize=7)
        axis.tick_params(labelsize=6)
        fig.colorbar(artist, ax=axis, fraction=0.046, pad=0.03).ax.tick_params(labelsize=6)
    for axis in axes[selected.size :]:
        axis.set_visible(False)
    fig.suptitle(title, fontsize=10)
    fig.savefig(path, dpi=300, facecolor="white")
    fig.savefig(path.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit held-out-validated direct field gamma for Microscope1.")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--calibration-gamma", type=Path, default=DEFAULT_CALIBRATION_GAMMA)
    parser.add_argument("--carrier-complex", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--npupil", type=int, default=128)
    parser.add_argument("--adam-steps", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-training-emitters", type=int, default=3000)
    parser.add_argument("--max-validation-emitters", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=20260723)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = FieldGammaFitConfig(
        device=args.device,
        npupil=args.npupil,
        adam_steps=args.adam_steps,
        batch_size=args.batch_size,
        max_training_emitters=args.max_training_emitters,
        max_validation_emitters=args.max_validation_emitters,
        seed=args.seed,
    )
    dataset = Microscope1Dataset(args.dataset_root)
    dataset.validate()
    with np.load(args.calibration_gamma, allow_pickle=False) as payload:
        global_gamma = np.asarray(payload["gamma_nm"], dtype=np.float32)[:, 0, 0]
        mode_order = tuple(tuple(int(value) for value in row) for row in payload["mode_order"])
    carrier_complex = None
    if args.carrier_complex is not None:
        with np.load(args.carrier_complex, allow_pickle=False) as payload:
            key = "carrier_complex" if "carrier_complex" in payload else "complex_pupil"
            carrier_complex = np.asarray(payload[key], dtype=np.complex64)
    provenance = {
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", "not-under-slurm"),
        "torch_cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE",
        "calibration_gamma_sha256": _sha256(args.calibration_gamma),
        "shared_carrier_sha256": _sha256(args.carrier_complex) if args.carrier_complex is not None else None,
        "config": asdict(config),
    }
    print(json.dumps(provenance, indent=2, sort_keys=True), flush=True)
    observations = harvest_oracle_patches(dataset, min_neighbor_distance_px=31.0)
    result = fit_field_gamma(
        observations,
        global_gamma_nm=global_gamma,
        mode_order=mode_order,
        config=config,
        carrier_complex=carrier_complex,
    )
    outputs = write_field_gamma_outputs(
        args.output_dir,
        result=result,
        config=config,
        carrier_path=args.carrier_complex,
    )
    print(
        json.dumps(
            {"output_dir": str(outputs.output_dir), "field_accepted": result.field_accepted, "metrics": result.metrics},
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["FieldGammaOutputs", "write_field_gamma_outputs"]
