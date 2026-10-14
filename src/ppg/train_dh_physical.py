#!/usr/bin/env python3
"""Train the DH expert while updating a run-local PPG map from sample ROIs."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import shutil
from types import SimpleNamespace

import numpy as np
import tifffile
import torch
import yaml

from ppg.lunar_background import (
    LunarBackgroundEstimate,
    estimate_lunar_tiff_background,
)
from ppg.lunar_photon_range import estimate_lunar_photon_range
from ppg.physical_update import (
    build_sample_roi_bank,
    resolve_roi_update_budget,
    samples_from_roi_bank,
    should_update_ppg,
    update_ppg_map,
)
from ppg.condition_input import load_ppg_conditioning, ppg_model_input
from unity_psf.contracts import (
    PSFModality,
    ResolvedTrainingContract,
    ResolvedTrainingInstance,
)
from unity_psf.localization.legacy_decode import decode_liteloc_formal_infer_emitters
from unity_psf.roi_library import EmitterPosterior, ROIBank, ROIRecord, save_roi_bank
from unity_psf.runtime import ensure_run_layout
from unity_psf.training.entrypoints.train_modality_expert_parallel import (
    _train_owned_modality,
)


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _resolved_contract(*, run_id: str, run_dir: Path, config_path: Path, config: dict, epochs: int, step_budget: int) -> ResolvedTrainingContract:
    contract = ResolvedTrainingContract(
        run_id=run_id,
        run_dir=str(run_dir),
        epochs=epochs,
        required_gpus=1,
        smoke=False,
        instances=(ResolvedTrainingInstance("double_helix:main", "double_helix", "main", str(config_path), 4301, 4311, step_budget),),
        configs={"double_helix:main": config},
        inputs={},
    )
    (run_dir / "resolved").mkdir(parents=True, exist_ok=True)
    (run_dir / "resolved" / "training_contract.json").write_text(
        json.dumps(contract.to_dict(), indent=2, sort_keys=True), encoding="utf-8"
    )
    return contract


def resolve_training_epochs(config: dict, *, override: int | None) -> int:
    epochs = int(config["train"]["epochs"] if override is None else override)
    if epochs <= 0:
        raise ValueError("training epochs must be positive")
    return epochs


def apply_lunar_background_estimate(
    config: dict,
    estimate: LunarBackgroundEstimate,
) -> None:
    background_range = [float(value) for value in estimate.background_range]
    config["train"]["online_generation"]["background_range"] = background_range
    config["train"]["ppg_physical_update"]["resolved_background"] = {
        "background_range": background_range,
        "background_mean": float(estimate.background_mean),
        "frame_range": [int(value) for value in estimate.frame_range],
        "percentile": float(estimate.percentile),
        "camera_backward": estimate.camera_backward,
    }


def _select_update_records(
    records: tuple, *, max_rois: int, update_index: int
) -> tuple:
    """Return a deterministic rotating subset without modifying the ROI library."""
    if max_rois <= 0:
        raise ValueError("max_rois must be positive")
    if len(records) <= max_rois:
        return records
    start = (int(update_index) * int(max_rois)) % len(records)
    return tuple(records[(start + offset) % len(records)] for offset in range(max_rois))


def _validate_ppg_map(path: Path, *, psf_support_px: int) -> None:
    with np.load(path, allow_pickle=False) as payload:
        schema = str(payload["schema"].item())
        gamma_shape = payload["gamma"].shape
        roi_shape = tuple(int(value) for value in payload["roi_shape_yx"])
        field_shape = payload["coefficient_field"].shape
    if schema != "ppg.pca_map.v1":
        raise ValueError(f"unsupported PPG map schema: {schema}")
    if gamma_shape != (9, 4):
        raise ValueError(f"PPG map gamma shape must be (9, 4), got {gamma_shape}")
    if field_shape != (4, 40, 40):
        raise ValueError(
            f"PPG coefficient field must be (4, 40, 40), got {field_shape}"
        )
    expected = (int(psf_support_px), int(psf_support_px))
    if roi_shape != expected:
        raise ValueError(f"PPG map ROI shape {roi_shape} does not match {expected}")


def _prepare_working_map(
    initial_map: Path,
    working_map: Path,
    *,
    resume: bool,
    psf_support_px: int,
) -> None:
    if resume:
        if not working_map.is_file():
            raise FileNotFoundError(
                f"resume requires an existing working PPG map: {working_map}"
            )
        _validate_ppg_map(working_map, psf_support_px=psf_support_px)
        return
    _validate_ppg_map(initial_map, psf_support_px=psf_support_px)
    working_map.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(initial_map, working_map)


class PPGPhysicalUpdateHook:
    def __init__(self, *, config: dict, working_map: Path, sample_tiff: Path) -> None:
        self.config = dict(config)
        self.photon_scale = float(self.config.get("photon_max", 31000.0))
        self.working_map = working_map
        self.sample_tiff = sample_tiff
        with tifffile.TiffFile(sample_tiff) as handle:
            shape = tuple(int(v) for v in handle.series[0].shape)
        if len(shape) != 3:
            raise ValueError(f"sample TIFF must be (T,H,W), got {shape}")
        self.frame_shape_yx = shape[-2:]
        self.condition_provider = load_ppg_conditioning(self.working_map)
        self.roi_library: ROIBank | None = None

    @staticmethod
    def _redetect_record(
        *,
        model: torch.nn.Module,
        record: ROIRecord,
        condition_provider,
        source_field_shape_yx: tuple[int, int],
        psf_support_px: int,
        probability_threshold: float,
        z_scale_nm: float = 600.0,
        photon_scale: float = 31000.0,
    ) -> ROIRecord:
        raw = torch.as_tensor(
            record.raw_frames_photon,
            device=next(model.parameters()).device,
            dtype=torch.float32,
        )
        with torch.no_grad():
            model_input = ppg_model_input(
                raw[None],
                provider=condition_provider,
                origin_xy_px=record.roi_origin_xy_px,
                source_field_shape_yx=source_field_shape_yx,
            )
            output = model(model_input)
            decoded = decode_liteloc_formal_infer_emitters(
                output,
                accept_threshold=float(probability_threshold),
                photon_scale=float(photon_scale),
                z_scale=float(z_scale_nm),
            )
            edge = int(psf_support_px) // 2 + 1
            height, width = raw.shape[-2:]
            accepted = [
                index
                for index, xyz in enumerate(decoded.xyz_px_nm)
                if edge <= float(xyz[0]) < width - edge
                and edge <= float(xyz[1]) < height - edge
            ]
            if len(accepted) > 512:
                scores = decoded.probability[accepted]
                accepted = [accepted[index] for index in torch.topk(scores, 512).indices.tolist()]
            emitters = []
            for index in accepted:
                local_x, local_y, z_nm = (
                    float(value) for value in decoded.xyz_px_nm[index]
                )
                sigma_xy = tuple(float(value) for value in decoded.sigma_xy_px[index])
                emitters.append(
                    EmitterPosterior(
                        probability=float(decoded.probability[index]),
                        cell_xy_px=(float(np.floor(local_x)), float(np.floor(local_y))),
                        mu_xy_px=(local_x, local_y),
                        sigma_xy_px=sigma_xy,
                        mu_z_nm=z_nm,
                        sigma_z_nm=float(decoded.sigma_z_nm[index]),
                        mu_photons=float(decoded.photons[index]),
                        sigma_photons=float(decoded.sigma_photons[index]),
                        local_xy_px=(local_x, local_y),
                        full_xy_px=(
                            float(record.roi_origin_xy_px[0]) + local_x,
                            float(record.roi_origin_xy_px[1]) + local_y,
                        ),
                        frame_index=int(record.frame_window[0])
                        + len(record.raw_frames_photon) // 2,
                    )
                )
        return replace(
            record,
            emitters=tuple(emitters),
            summary={**record.summary, "redetected_with_current_model": True},
        )

    def _redetect_library(self, *, runtime, bank: ROIBank) -> ROIBank:
        was_training = runtime.model.training
        runtime.model.eval()
        records = tuple(
            self._redetect_record(
                model=runtime.model,
                record=record,
                condition_provider=self.condition_provider,
                source_field_shape_yx=self.frame_shape_yx,
                psf_support_px=int(self.config["psf_support_px"]),
                probability_threshold=float(self.config["probability_threshold"]),
                photon_scale=self.photon_scale,
            )
            for record in bank.records
        )
        if was_training:
            runtime.model.train()
        return replace(
            bank,
            records=records,
            metadata={
                **bank.metadata,
                "roi_update_reinference": True,
                "roi_update_detection_source": "current_runtime_model",
            },
        )

    def __call__(self, *, runtime, epoch: int, layout) -> dict | None:
        if not should_update_ppg(
            epoch,
            start_epoch=int(self.config["start_epoch"]),
            interval=int(self.config["interval"]),
        ):
            return None
        update_dir = (
            layout.artifacts_dir / "ppg_physical_updates" / f"epoch_{epoch:03d}"
        )
        update_dir.mkdir(parents=True, exist_ok=True)
        search_start, search_stop = (int(v) for v in self.config["frame_search_range"])
        block = int(self.config["frame_block_size"])
        block_count = int(self.config["frame_block_count"])
        pool_max_rois, update_max_rois = resolve_roi_update_budget(self.config)
        update_index = (int(epoch) - int(self.config["start_epoch"])) // int(
            self.config["interval"]
        )
        latest_start = search_stop - block
        starts = (
            np.linspace(search_start, latest_start, block_count).round().astype(int)
        )
        starts = search_start + (starts - search_start + update_index * block) % max(
            search_stop - search_start - block + 1, 1
        )
        frame_ranges = [(int(start), int(start + block)) for start in starts]
        roi_bank_path = update_dir / "sample_roi_bank.h5"
        if self.roi_library is None:
            banks = []
            per_block_rois = max(1, int(np.ceil(pool_max_rois / block_count)))
            for block_index, frame_range in enumerate(frame_ranges):
                banks.append(
                    build_sample_roi_bank(
                        model=runtime.model,
                        sample_tiff=self.sample_tiff,
                        output_h5=update_dir
                        / f"sample_roi_bank_block_{block_index:02d}.h5",
                        frame_shape_yx=self.frame_shape_yx,
                        frame_range=frame_range,
                        probability_threshold=float(
                            self.config["probability_threshold"]
                        ),
                        grid_shape=tuple(int(v) for v in self.config["grid_shape"]),
                        max_rois=per_block_rois,
                        roi_size_px=int(self.config["roi_size_px"]),
                        psf_support_px=int(self.config["psf_support_px"]),
                        photon_scale=self.photon_scale,
                        condition_provider=self.condition_provider,
                    )
                )
            records = [
                replace(record, roi_id=index)
                for index, record in enumerate(
                    record for bank_part in banks for record in bank_part.records
                )
            ]
            self.roi_library = ROIBank(
                records=tuple(records[:pool_max_rois]),
                config={
                    **banks[0].config,
                    "frame_ranges": frame_ranges,
                    "max_rois": pool_max_rois,
                    "roi_library_max_rois": pool_max_rois,
                    "roi_update_max_rois": update_max_rois,
                },
                metadata={
                    **banks[0].metadata,
                    "frame_ranges": frame_ranges,
                    "frame_sampling": "uniform_blocks",
                    "roi_library_max_rois": pool_max_rois,
                    "roi_update_max_rois": update_max_rois,
                    "roi_update_frame_block_count": block_count,
                    "roi_library_built_once": True,
                },
                empty_grid_cell_ids=tuple(
                    sorted(
                        set.intersection(
                            *(set(item.empty_grid_cell_ids) for item in banks)
                        )
                    )
                ),
            )
        library = self._redetect_library(runtime=runtime, bank=self.roi_library)
        self.roi_library = library
        update_records = _select_update_records(
            library.records,
            max_rois=update_max_rois,
            update_index=update_index,
        )
        bank = replace(
            library,
            records=update_records,
            metadata={
                **library.metadata,
                "roi_update_index": update_index,
                "roi_update_selected_rois": len(update_records),
            },
        )
        save_roi_bank(bank, roi_bank_path)
        samples = samples_from_roi_bank(bank, field_shape_yx=self.frame_shape_yx)
        occupied_cells = len({record.grid_cell_id for record in bank.records})
        if occupied_cells < int(self.config["min_occupied_grid_cells"]):
            raise RuntimeError(
                f"ROI bank covers only {occupied_cells} grid cells; "
                f"required {self.config['min_occupied_grid_cells']}"
            )
        output_map = update_dir / "ppg_map_updated.npz"
        if runtime.model.input_normalization is None:
            raise RuntimeError("PPG physical update requires LUNAR input normalization")
        metrics = update_ppg_map(
            self.working_map,
            output_map,
            samples=samples,
            device=str(next(runtime.model.parameters()).device),
            steps=int(self.config["fit_steps"]),
            learning_rate=float(self.config["learning_rate"]),
            regularization=float(self.config["regularization"]),
            normalization=runtime.model.input_normalization,
            gamma_ridge=float(self.config["gamma_ridge"]),
            gamma_boundary_anchor_strength=float(
                self.config["gamma_boundary_anchor_strength"]
            ),
            gamma_max_abs_delta=float(self.config["gamma_max_abs_delta"]),
            ncc_threshold=float(self.config["ncc_threshold"]),
            heldout_fraction=float(self.config["heldout_fraction"]),
            heldout_min_nll_improvement=float(
                self.config["heldout_min_nll_improvement"]
            ),
            heldout_min_improved_fraction=float(
                self.config["heldout_min_improved_fraction"]
            ),
            over_cut_px=int(self.config["over_cut_px"]),
        )
        if metrics["map_accepted"]:
            temporary = self.working_map.with_suffix(".next.npz")
            shutil.copyfile(output_map, temporary)
            os.replace(temporary, self.working_map)
            self.condition_provider = load_ppg_conditioning(self.working_map)
            refresh = getattr(
                runtime.channels["main"].batches, "refresh_physical_model", None
            )
            if not callable(refresh):
                raise RuntimeError(
                    "PPG provider does not expose renderer cache refresh"
                )
            refresh()
        summary = {
            "epoch": int(epoch),
            "sample_tiff": str(self.sample_tiff),
            "sample_field_shape_yx": list(self.frame_shape_yx),
            "bead_field_shape_yx": [
                int(v) for v in np.load(self.working_map)["field_shape_yx"]
            ],
            "coordinate_mapping": "normalized_xy",
            "frame_ranges": frame_ranges,
            "occupied_grid_cells": occupied_cells,
            "roi_bank": str(roi_bank_path),
            "updated_map": str(output_map),
            "roi_library_max_rois": pool_max_rois,
            "roi_update_max_rois": update_max_rois,
            "roi_update_frame_block_count": block_count,
            "roi_update_reinference": True,
            "roi_update_detection_source": "current_runtime_model",
            **metrics,
        }
        (update_dir / "metrics.json").write_text(
            json.dumps(summary, indent=2, default=_json_default) + "\n"
        )
        return {
            key: value
            for key, value in summary.items()
            if key not in {"roi_metrics", "rejected_roi_metrics"}
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--contract-hash", default=None)
    parser.add_argument("--density", type=float, default=None,
                        help="Override online simulation emitter density (emitters/um^2).")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    if args.density is not None:
        if args.density <= 0:
            raise ValueError("--density must be positive")
        config["train"]["online_generation"]["emitter_density_um2"] = float(args.density)
    epochs = resolve_training_epochs(config, override=args.epochs)
    config["train"]["epochs"] = epochs
    initial_map = Path(config["train"]["online_generation"]["ppg_map_path"]).resolve()
    physical_config = config["train"]["ppg_physical_update"]
    sample_tiff = Path(physical_config["sample_tiff"]).resolve()
    if not initial_map.is_file():
        raise FileNotFoundError(initial_map)
    if not sample_tiff.is_file():
        raise FileNotFoundError(sample_tiff)
    background_config = physical_config["background_estimation"]
    background_estimate = estimate_lunar_tiff_background(
        sample_tiff,
        percentile=float(background_config["percentile"]),
        max_size_gb=float(background_config["max_size_gb"]),
        chunk_frames=int(background_config["chunk_frames"]),
    )
    apply_lunar_background_estimate(config, background_estimate)
    photon_range, photon_metadata = estimate_lunar_photon_range(
        sample_tiff,
        psf_size=int(config["train"]["online_generation"]["vector_psf_size"]),
    )
    online = config["train"]["online_generation"]
    online["photon_range"] = [float(v) for v in photon_range]
    online["photon_scale"] = float(photon_range[1])
    config["train"]["loss"]["params"]["photon_scale"] = float(photon_range[1])
    config["train"]["scaling"]["photon_max"] = float(photon_range[1])
    config["train"]["ppg_physical_update"]["photon_range"] = [float(v) for v in photon_range]
    config["train"]["ppg_physical_update"]["photon_max"] = float(photon_range[1])
    config["train"]["ppg_physical_update"]["resolved_photon_range"] = photon_metadata
    physical = dict(config["train"]["ppg_physical_update"])
    layout = ensure_run_layout(
        args.run_root.resolve(),
        args.run_id,
        stage_names=("dh_ppg_physical_training",),
    )
    working_map = layout.artifacts_dir / "ppg_map_working.npz"
    _prepare_working_map(
        initial_map,
        working_map,
        resume=args.resume,
        psf_support_px=int(physical["psf_support_px"]),
    )
    config["train"]["online_generation"]["ppg_map_path"] = str(working_map)
    resolved_config = layout.metadata_dir / "resolved_physical_config.yaml"
    resolved_config.write_text(yaml.safe_dump(config, sort_keys=False))
    resolved_config = resolved_config.resolve()
    entry = (
        "double_helix:main",
        {
            "config": str(resolved_config),
            "model_seed": 4301,
            "data_seed": 4311,
            "step_budget": int(config["train"]["online_generation"]["steps_per_epoch"]),
        },
    )
    resolved_contract = _resolved_contract(
        run_id=args.run_id,
        run_dir=layout.run_dir,
        config_path=resolved_config,
        config=config,
        epochs=epochs,
        step_budget=int(config["train"]["online_generation"]["steps_per_epoch"]),
    )
    if args.contract_hash is not None and args.contract_hash != resolved_contract.contract_hash:
        raise ValueError("provided contract hash does not match the resolved DH training contract")
    hook = PPGPhysicalUpdateHook(
        config=physical, working_map=working_map, sample_tiff=sample_tiff
    )
    status = _train_owned_modality(
        SimpleNamespace(config=resolved_config, resume=args.resume),
        layout=layout,
        modality=PSFModality.DOUBLE_HELIX,
        entries=[entry],
        epochs=epochs,
        rank=0,
        local_rank=0,
        device=args.device,
        attempt_id=args.run_id,
        contract_hash=resolved_contract.contract_hash,
        resolved_contract=resolved_contract,
        epoch_end_hook=hook,
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "run": str(layout.run_dir),
                "metrics": status["metrics"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
