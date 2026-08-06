from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from .._paths import PROJECT_ROOT as V04_ROOT
from ..calibration import calibration_mode_order
from ..dataset import Microscope1Dataset
from ..localization import (
    AngleZCalibration,
    IndependentLocalizations,
    LocalizationConfig,
    localize_frames,
)
from ..physical_update import (
    FullFOVPhysicalUpdateConfig,
    evaluate_full_fov_poisson_loss,
    evaluate_residual_coefficients,
    fit_full_fov_physical_update,
    gamma_terms_to_tensor,
    spatial_gamma_terms,
)
from ..vector_model import DoubleHelixVectorPSF


DEFAULT_DATASET_ROOT = (
    V04_ROOT.parent / "datasets/training_sets/double_helix/Simulated_datasets_Microscope1"
)
DEFAULT_GLOBAL_ROOT = (
    V04_ROOT / "output/double_helix/microscope1_shared_carrier_residual21_20260725"
)
DEFAULT_OUTPUT_ROOT = (
    V04_ROOT
    / "output/double_helix/microscope1_shared_carrier_residual21_physical_update_20260725"
)


@dataclass(frozen=True)
class FixedFullFOVBank:
    frame_indices: np.ndarray
    train_frame_indices: np.ndarray
    heldout_frame_indices: np.ndarray
    localization_count_per_frame: np.ndarray
    median_ncc_per_frame: np.ndarray
    image_shape_hw: tuple[int, int]
    projected_emitter_count: int
    selection_strategy: str
    roi_bank_count: int = 1


def select_fixed_full_fov_bank_frames(
    localizations: IndependentLocalizations,
    *,
    frame_count: int,
    image_shape_hw: tuple[int, int],
    expected_emitters_per_frame: int,
    target_projected_emitters: int,
    heldout_fraction: float,
    seed: int,
) -> FixedFullFOVBank:
    if frame_count <= 0 or expected_emitters_per_frame <= 0 or target_projected_emitters <= 0:
        raise ValueError("Frame and emitter counts must be positive.")
    if not 0.0 < heldout_fraction < 1.0:
        raise ValueError("heldout_fraction must be between zero and one.")
    if localizations.frame_index.size == 0:
        raise ValueError("Initial localizations are required to build the fixed full-FOV bank.")
    if np.any(localizations.frame_index < 0) or np.any(localizations.frame_index >= frame_count):
        raise ValueError("Localization frame indices are outside the TIFF frame range.")

    counts = np.bincount(localizations.frame_index, minlength=frame_count)
    median_ncc = np.full(frame_count, -np.inf, dtype=np.float64)
    for frame_index in np.flatnonzero(counts):
        rows = localizations.frame_index == frame_index
        median_ncc[frame_index] = float(np.median(localizations.ncc[rows]))
    frame_indices = np.arange(frame_count, dtype=np.int64)
    priority = np.lexsort(
        (
            frame_indices,
            -median_ncc,
            np.abs(counts - int(expected_emitters_per_frame)),
        )
    )
    target_frame_count = min(
        frame_count,
        int(np.ceil(target_projected_emitters / float(expected_emitters_per_frame))),
    )
    selected = np.sort(priority[:target_frame_count]).astype(np.int64)
    selected_emitter_count = int(counts[selected].sum())
    if selected_emitter_count == 0:
        raise ValueError("The selected full-FOV bank contains no localized emitters.")

    shuffled = np.random.default_rng(int(seed)).permutation(selected)
    heldout_count = max(1, int(round(selected.size * heldout_fraction)))
    heldout_count = min(heldout_count, selected.size - 1)
    heldout = np.sort(shuffled[:heldout_count]).astype(np.int64)
    train = np.sort(shuffled[heldout_count:]).astype(np.int64)
    return FixedFullFOVBank(
        frame_indices=selected,
        train_frame_indices=train,
        heldout_frame_indices=heldout,
        localization_count_per_frame=counts[selected].astype(np.int64),
        median_ncc_per_frame=median_ncc[selected].astype(np.float32),
        image_shape_hw=tuple(int(value) for value in image_shape_hw),
        projected_emitter_count=selected_emitter_count,
        selection_strategy="quality_ranked",
    )


def select_fixed_first_full_fov_bank_frames(
    localizations: IndependentLocalizations,
    *,
    frame_count: int,
    first_frame_count: int,
    image_shape_hw: tuple[int, int],
    heldout_stride: int,
) -> FixedFullFOVBank:
    if not 1 < first_frame_count <= frame_count:
        raise ValueError("first_frame_count must be between two and the TIFF frame count.")
    if heldout_stride < 2 or heldout_stride > first_frame_count:
        raise ValueError("heldout_stride must leave both training and heldout frames.")
    if localizations.frame_index.size == 0:
        raise ValueError("Initial localizations are required to build the fixed full-FOV bank.")
    if np.any(localizations.frame_index < 0) or np.any(localizations.frame_index >= frame_count):
        raise ValueError("Localization frame indices are outside the TIFF frame range.")

    selected = np.arange(first_frame_count, dtype=np.int64)
    heldout = np.arange(heldout_stride - 1, first_frame_count, heldout_stride, dtype=np.int64)
    train = np.setdiff1d(selected, heldout, assume_unique=True)
    counts = np.bincount(localizations.frame_index, minlength=frame_count)
    median_ncc = np.full(first_frame_count, np.nan, dtype=np.float32)
    for frame_index in selected:
        rows = localizations.frame_index == frame_index
        if np.any(rows):
            median_ncc[frame_index] = float(np.median(localizations.ncc[rows]))
    return FixedFullFOVBank(
        frame_indices=selected,
        train_frame_indices=train,
        heldout_frame_indices=heldout,
        localization_count_per_frame=counts[selected].astype(np.int64),
        median_ncc_per_frame=median_ncc,
        image_shape_hw=tuple(int(value) for value in image_shape_hw),
        projected_emitter_count=int(counts[selected].sum()),
        selection_strategy=f"fixed_first_{first_frame_count}_frames",
    )


def write_physical_update_outputs(
    output_dir: str | Path,
    *,
    gamma_terms_nm: np.ndarray,
    spatial_terms: Sequence[tuple[int, int]],
    mode_order: Sequence[tuple[int, int]],
    image_shape_hw: tuple[int, int],
    bank: FixedFullFOVBank,
    update_records: list[dict[str, Any]],
    carrier_path: str | Path,
    initial_gamma_path: str | Path,
    dataset_root: str | Path,
) -> dict[str, Path]:
    root = Path(output_dir)
    arrays_dir = root / "arrays"
    figures_dir = root / "figures"
    metrics_dir = root / "metrics"
    metadata_dir = root / "metadata"
    for directory in (arrays_dir, figures_dir, metrics_dir, metadata_dir):
        directory.mkdir(parents=True, exist_ok=True)

    parameters = np.asarray(gamma_terms_nm, dtype=np.float32)
    modes = tuple(tuple(int(value) for value in mode) for mode in mode_order)
    terms = tuple(tuple(int(value) for value in term) for term in spatial_terms)
    if parameters.shape != (21, len(terms)) or len(modes) != 21:
        raise ValueError("Physical-update output requires exactly 21 residual Zernike modes.")
    if tuple(int(value) for value in image_shape_hw) != bank.image_shape_hw:
        raise ValueError("Output image shape must match the single full-FOV bank shape.")
    dense_gamma = gamma_terms_to_tensor(
        torch.as_tensor(parameters), terms=terms, degree=max(max(term) for term in terms)
    ).numpy()
    height, width = bank.image_shape_hw
    yy, xx = np.indices((height, width), dtype=np.float32)
    maps = evaluate_residual_coefficients(
        torch.as_tensor(parameters),
        x_px=torch.as_tensor(xx.reshape(-1)),
        y_px=torch.as_tensor(yy.reshape(-1)),
        image_shape_hw=bank.image_shape_hw,
        terms=terms,
    ).numpy().T.reshape(21, height, width)

    gamma_path = arrays_dir / "final_gamma_coefficients.npz"
    np.savez_compressed(
        gamma_path,
        gamma_nm=dense_gamma.astype(np.float32),
        gamma_terms_nm=parameters,
        spatial_terms=np.asarray(terms, dtype=np.int64),
        mode_order=np.asarray(modes, dtype=np.int64),
        spatial_order=np.asarray(max(max(term) for term in terms), dtype=np.int64),
        semantics=np.asarray("Neptune-style feedback-updated FD residual OPD above fixed DH carrier"),
    )
    zmap_path = arrays_dir / "alternating_full_roi_zernike_maps_nm.npz"
    np.savez_compressed(
        zmap_path,
        zernike_maps_nm=maps.astype(np.float32),
        mode_order=np.asarray(modes, dtype=np.int64),
        x_px=np.arange(width, dtype=np.float32),
        y_px=np.arange(height, dtype=np.float32),
        semantics=np.asarray("feedback-updated field-dependent residual OPD maps in nanometers"),
    )
    bank_path = arrays_dir / "fixed_full_fov_roi_bank.npz"
    np.savez_compressed(
        bank_path,
        frame_indices_0based=bank.frame_indices,
        train_frame_indices_0based=bank.train_frame_indices,
        heldout_frame_indices_0based=bank.heldout_frame_indices,
        localization_count_per_frame=bank.localization_count_per_frame,
        median_ncc_per_frame=bank.median_ncc_per_frame,
        selection_strategy=np.asarray(bank.selection_strategy),
        roi_bank_count=np.asarray(1, dtype=np.int64),
        roi_bank_shape_hw=np.asarray(bank.image_shape_hw, dtype=np.int64),
    )
    metrics_path = metrics_dir / "gamma_update_metrics.jsonl"
    metrics_path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in update_records),
        encoding="utf-8",
    )

    physical_state = {
        "source": "double_helix_full_fov_physical_update",
        "gamma_path": str(gamma_path.resolve()),
        "zmap_path": str(zmap_path.resolve()),
        "update_count": len(update_records),
        "heldout_accept_policy": "monitor",
        "feedback_applied": True,
    }
    state_path = metadata_dir / "current_physical_state.json"
    state_path.write_text(json.dumps(physical_state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest = {
        "parameterization": "fixed independent 128x128 shared DH carrier plus exactly 21 residual Zernike modes",
        "roi_bank_count": 1,
        "roi_bank_shape_hw": list(bank.image_shape_hw),
        "roi_bank_semantics": "one fixed bank containing complete 150x150 TIFF frames; no spatial bank splitting",
        "roi_bank_selection_strategy": bank.selection_strategy,
        "train_heldout_semantics": "temporal frame split within the same fixed full-FOV bank",
        "selected_frame_count": int(bank.frame_indices.size),
        "projected_emitter_count": int(bank.projected_emitter_count),
        "carrier_counted_as_zernike_mode": False,
        "residual_mode_count": 21,
        "residual_mode_order": [list(mode) for mode in modes],
        "defocus_zernike_2_0_fixed": True,
        "heldout_accept_policy": "monitor",
        "physical_updates_are_feedback_applied": True,
        "fd_map_interpretation": "Neptune-style feedback-updated FD residual zmap; not independently validated ground truth",
        "dataset_root": str(Path(dataset_root).resolve()),
        "shared_carrier_input": str(Path(carrier_path).resolve()),
        "shared_carrier_sha256": _sha256(Path(carrier_path)),
        "initial_gamma_input": str(Path(initial_gamma_path).resolve()),
        "initial_gamma_sha256": _sha256(Path(initial_gamma_path)),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", "not-under-slurm"),
    }
    manifest_path = metadata_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    coefficient_figure = figures_dir / "final_gamma_coefficient_maps.png"
    loss_figure = figures_dir / "gamma_update_loss.png"
    _render_coefficient_maps(maps, modes, coefficient_figure)
    _render_update_loss(update_records, loss_figure)
    return {
        "gamma_path": gamma_path,
        "zmap_path": zmap_path,
        "bank_path": bank_path,
        "metrics_path": metrics_path,
        "state_path": state_path,
        "manifest_path": manifest_path,
        "coefficient_figure": coefficient_figure,
        "loss_figure": loss_figure,
    }


def run_physical_updates(
    *,
    dataset_root: Path,
    carrier_path: Path,
    initial_gamma_path: Path,
    initial_localizations_path: Path,
    output_dir: Path,
    update_count: int,
    target_projected_emitters: int,
    heldout_fraction: float,
    gamma_steps: int,
    gamma_lr: float,
    frame_batch_size: int,
    localization_refinement_steps: int,
    seed: int,
    device: str,
    fixed_first_frame_count: int | None = None,
) -> dict[str, Path]:
    dataset = Microscope1Dataset(dataset_root)
    contract = dataset.validate()
    mode_order, initial_terms = _load_initial_gamma(initial_gamma_path)
    if mode_order != calibration_mode_order(21) or (2, 0) in mode_order:
        raise ValueError("The physical update requires the exact 21-mode residual basis with Z(2,0) excluded.")
    with np.load(carrier_path, allow_pickle=False) as payload:
        carrier_complex = np.asarray(payload["carrier_complex"], dtype=np.complex64)
    if carrier_complex.shape != (128, 128):
        raise ValueError("The independent shared DH carrier must have shape 128x128.")
    initial_localizations = _load_localizations(initial_localizations_path)
    if fixed_first_frame_count is None:
        bank = select_fixed_full_fov_bank_frames(
            initial_localizations,
            frame_count=contract.frame_shape[0],
            image_shape_hw=contract.frame_shape[1:],
            expected_emitters_per_frame=dataset.config.emitters_per_frame,
            target_projected_emitters=target_projected_emitters,
            heldout_fraction=heldout_fraction,
            seed=seed,
        )
    else:
        bank = select_fixed_first_full_fov_bank_frames(
            initial_localizations,
            frame_count=contract.frame_shape[0],
            first_frame_count=fixed_first_frame_count,
            image_shape_hw=contract.frame_shape[1:],
            heldout_stride=int(round(1.0 / heldout_fraction)),
        )
    frames = np.asarray(dataset.open_frames()[bank.frame_indices], dtype=np.float32)
    current_localizations = _remap_global_localizations(initial_localizations, bank.frame_indices)
    train_local_indices = np.flatnonzero(np.isin(bank.frame_indices, bank.train_frame_indices))
    heldout_local_indices = np.flatnonzero(np.isin(bank.frame_indices, bank.heldout_frame_indices))

    calibration = dataset.read_calibration()
    calibration_z_nm = dataset.z_sign * (
        np.arange(calibration.shape[0], dtype=np.float64) + dataset.z_index_origin
    ) * dataset.z_step_nm
    angle_calibration = AngleZCalibration.from_stack(calibration, calibration_z_nm)
    localization_config = LocalizationConfig(
        refinement_steps=localization_refinement_steps,
        image_shape_hw=contract.frame_shape[1:],
        npupil=128,
        device=device,
    )
    update_config = FullFOVPhysicalUpdateConfig(
        image_shape_hw=contract.frame_shape[1:],
        spatial_degree=2,
        gamma_steps=gamma_steps,
        gamma_lr=gamma_lr,
        frame_batch_size=frame_batch_size,
        npupil=128,
        device=device,
    )
    model = DoubleHelixVectorPSF(
        mode_order=mode_order,
        na=update_config.na,
        wavelength_nm=update_config.wavelength_nm,
        pixel_size_nm=update_config.pixel_size_nm,
        refractive_index=update_config.refractive_index,
        npupil=update_config.npupil,
        psf_size=update_config.psf_size,
        device=device,
    )
    carrier_tensor = torch.as_tensor(carrier_complex, device=device)
    terms = spatial_gamma_terms(2)
    current_terms = initial_terms
    update_records: list[dict[str, Any]] = []
    updates_dir = output_dir / "updates"

    for update_index in range(1, update_count + 1):
        if update_index > 1:
            current_gamma = gamma_terms_to_tensor(
                torch.as_tensor(current_terms), terms=terms, degree=2
            ).numpy()
            current_localizations = localize_frames(
                frames,
                gamma_nm=current_gamma,
                mode_order=mode_order,
                angle_calibration=angle_calibration,
                config=localization_config,
                carrier_complex=carrier_complex,
            )
        train_frames, train_localizations = _subset_bank(
            frames, current_localizations, train_local_indices
        )
        heldout_frames, heldout_localizations = _subset_bank(
            frames, current_localizations, heldout_local_indices
        )
        heldout_before = evaluate_full_fov_poisson_loss(
            heldout_frames,
            localizations=heldout_localizations,
            gamma_terms_nm=torch.as_tensor(current_terms, device=device),
            terms=terms,
            carrier_complex=carrier_tensor,
            model=model,
            config=update_config,
        )
        result = fit_full_fov_physical_update(
            train_frames,
            localizations=train_localizations,
            initial_gamma_terms_nm=current_terms,
            mode_order=mode_order,
            carrier_complex=carrier_complex,
            config=update_config,
            model=model,
        )
        heldout_after = evaluate_full_fov_poisson_loss(
            heldout_frames,
            localizations=heldout_localizations,
            gamma_terms_nm=torch.as_tensor(result.gamma_terms_nm, device=device),
            terms=terms,
            carrier_complex=carrier_tensor,
            model=model,
            config=update_config,
        )
        current_terms = result.gamma_terms_nm
        record = dict(result.metrics)
        record.update(
            {
                "update_index": update_index,
                "roi_bank_frame_count": int(bank.frame_indices.size),
                "localized_emitter_count": int(current_localizations.frame_index.size),
                "train_frame_count": int(train_frames.shape[0]),
                "heldout_frame_count": int(heldout_frames.shape[0]),
                "heldout_poisson_nll_before": heldout_before,
                "heldout_poisson_nll_after": heldout_after,
            }
        )
        update_records.append(record)
        update_dir = updates_dir / f"update_{update_index:03d}"
        update_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            update_dir / "gamma_update.npz",
            gamma_terms_nm=current_terms,
            gamma_nm=result.gamma_nm,
            mode_order=np.asarray(mode_order, dtype=np.int64),
            spatial_terms=np.asarray(terms, dtype=np.int64),
            loss_history=result.loss_history,
        )
        (update_dir / "metrics.json").write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(json.dumps(record, sort_keys=True), flush=True)

    return write_physical_update_outputs(
        output_dir,
        gamma_terms_nm=current_terms,
        spatial_terms=terms,
        mode_order=mode_order,
        image_shape_hw=contract.frame_shape[1:],
        bank=bank,
        update_records=update_records,
        carrier_path=carrier_path,
        initial_gamma_path=initial_gamma_path,
        dataset_root=dataset_root,
    )


def _load_initial_gamma(path: Path) -> tuple[tuple[tuple[int, int], ...], np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        gamma = np.asarray(payload["gamma_nm"], dtype=np.float32)
        mode_order = tuple(tuple(int(value) for value in row) for row in payload["mode_order"])
    if gamma.shape != (21, 1, 1):
        raise ValueError("Initial residual gamma must contain 21 global coefficients.")
    terms = np.zeros((21, len(spatial_gamma_terms(2))), dtype=np.float32)
    terms[:, 0] = gamma[:, 0, 0]
    return mode_order, terms


def _load_localizations(path: Path) -> IndependentLocalizations:
    with np.load(path, allow_pickle=False) as payload:
        return IndependentLocalizations(
            frame_index=np.asarray(payload["frame_index"], dtype=np.int64),
            x_px=np.asarray(payload["x_px"], dtype=np.float32),
            y_px=np.asarray(payload["y_px"], dtype=np.float32),
            z_nm=np.asarray(payload["z_nm"], dtype=np.float32),
            photons_adu=np.asarray(payload["photons_adu"], dtype=np.float32),
            background_adu=np.asarray(payload["background_adu"], dtype=np.float32),
            ncc=np.asarray(payload["ncc"], dtype=np.float32),
            lobe_angle_rad=np.asarray(payload["lobe_angle_rad"], dtype=np.float32),
            lobe_separation_px=np.asarray(payload["lobe_separation_px"], dtype=np.float32),
        )


def _remap_global_localizations(
    localizations: IndependentLocalizations, selected_global_frames: np.ndarray
) -> IndependentLocalizations:
    mapping = {int(global_index): local_index for local_index, global_index in enumerate(selected_global_frames)}
    keep = np.isin(localizations.frame_index, selected_global_frames)
    remapped = np.asarray([mapping[int(value)] for value in localizations.frame_index[keep]], dtype=np.int64)
    return _take_localizations(localizations, keep, frame_index=remapped)


def _subset_bank(
    frames: np.ndarray,
    localizations: IndependentLocalizations,
    selected_local_frames: np.ndarray,
) -> tuple[np.ndarray, IndependentLocalizations]:
    keep = np.isin(localizations.frame_index, selected_local_frames)
    mapping = {int(old): new for new, old in enumerate(selected_local_frames)}
    remapped = np.asarray([mapping[int(value)] for value in localizations.frame_index[keep]], dtype=np.int64)
    return frames[selected_local_frames], _take_localizations(localizations, keep, frame_index=remapped)


def _take_localizations(
    values: IndependentLocalizations, keep: np.ndarray, *, frame_index: np.ndarray
) -> IndependentLocalizations:
    return IndependentLocalizations(
        frame_index=frame_index,
        x_px=values.x_px[keep],
        y_px=values.y_px[keep],
        z_nm=values.z_nm[keep],
        photons_adu=values.photons_adu[keep],
        background_adu=values.background_adu[keep],
        ncc=values.ncc[keep],
        lobe_angle_rad=values.lobe_angle_rad[keep],
        lobe_separation_px=values.lobe_separation_px[keep],
    )


def _plotting() -> Any:
    cache_dir = V04_ROOT.parent / ".local/cache/matplotlib"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
    import matplotlib.pyplot as plt

    return plt


def _render_coefficient_maps(
    maps: np.ndarray, mode_order: Sequence[tuple[int, int]], path: Path
) -> None:
    plt = _plotting()
    figure, axes = plt.subplots(3, 7, figsize=(14, 6.4), constrained_layout=True)
    for axis, coefficient_map, mode in zip(axes.ravel(), maps, mode_order, strict=True):
        limit = max(float(np.max(np.abs(coefficient_map))), 1e-6)
        artist = axis.imshow(coefficient_map, cmap="RdBu_r", vmin=-limit, vmax=limit)
        axis.set_title(f"Z({mode[0]},{mode[1]})", fontsize=8)
        axis.set_xticks([])
        axis.set_yticks([])
        figure.colorbar(artist, ax=axis, fraction=0.045, pad=0.03)
    figure.suptitle("Feedback-updated field-dependent residual Zernike maps (nm)")
    figure.savefig(path, dpi=240, facecolor="white")
    plt.close(figure)


def _render_update_loss(records: list[dict[str, Any]], path: Path) -> None:
    plt = _plotting()
    figure, axis = plt.subplots(figsize=(6.4, 4.0), constrained_layout=True)
    indices = [int(record["update_index"]) for record in records]
    before = [float(record.get("heldout_poisson_nll_before", np.nan)) for record in records]
    after = [float(record.get("heldout_poisson_nll_after", np.nan)) for record in records]
    axis.plot(indices, before, marker="o", label="Heldout before")
    axis.plot(indices, after, marker="s", label="Heldout after")
    axis.set_xlabel("Physical update")
    axis.set_ylabel("Poisson NLL / pixel")
    axis.legend(frameon=False)
    axis.spines[["top", "right"]].set_visible(False)
    figure.savefig(path, dpi=240, facecolor="white")
    plt.close(figure)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Neptune-style alternating physical updates on one fixed 150x150 full-FOV ROI bank."
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument(
        "--carrier-complex",
        type=Path,
        default=DEFAULT_GLOBAL_ROOT / "calibration/arrays/shared_double_helix_carrier.npz",
    )
    parser.add_argument(
        "--initial-gamma",
        type=Path,
        default=DEFAULT_GLOBAL_ROOT / "calibration/arrays/residual_gamma_initialization.npz",
    )
    parser.add_argument(
        "--initial-localizations",
        type=Path,
        default=DEFAULT_GLOBAL_ROOT / "evaluation/arrays/independent_localizations.npz",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_ROOT / "physical_update")
    parser.add_argument("--update-count", type=int, default=5)
    parser.add_argument("--target-projected-emitters", type=int, default=1000)
    parser.add_argument("--fixed-first-frame-count", type=int)
    parser.add_argument("--heldout-fraction", type=float, default=0.1)
    parser.add_argument("--gamma-steps", type=int, default=100)
    parser.add_argument("--gamma-lr", type=float, default=0.025)
    parser.add_argument("--frame-batch-size", type=int, default=8)
    parser.add_argument("--localization-refinement-steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for the formal physical update but is unavailable.")
    provenance = {
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE",
        "roi_bank_count": 1,
        "roi_bank_shape_hw": [150, 150],
        "config": vars(args) | {"dataset_root": str(args.dataset_root), "output_dir": str(args.output_dir)},
    }
    print(json.dumps(provenance, indent=2, sort_keys=True, default=str), flush=True)
    outputs = run_physical_updates(
        dataset_root=args.dataset_root,
        carrier_path=args.carrier_complex,
        initial_gamma_path=args.initial_gamma,
        initial_localizations_path=args.initial_localizations,
        output_dir=args.output_dir,
        update_count=args.update_count,
        target_projected_emitters=args.target_projected_emitters,
        heldout_fraction=args.heldout_fraction,
        gamma_steps=args.gamma_steps,
        gamma_lr=args.gamma_lr,
        frame_batch_size=args.frame_batch_size,
        localization_refinement_steps=args.localization_refinement_steps,
        seed=args.seed,
        device=args.device,
        fixed_first_frame_count=args.fixed_first_frame_count,
    )
    print(json.dumps({key: str(value) for key, value in outputs.items()}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "FixedFullFOVBank",
    "run_physical_updates",
    "select_fixed_first_full_fov_bank_frames",
    "select_fixed_full_fov_bank_frames",
    "write_physical_update_outputs",
]
