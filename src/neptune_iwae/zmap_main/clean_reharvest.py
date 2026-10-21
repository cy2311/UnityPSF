from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from .harvest import HarvestedEmitters, build_starter_harvester_config, harvest_tiff_points
from .nat_fit import build_nat_spot_fitter_config, build_patch_dataset_from_emitters
from .starter_model import build_density1_starter_model


@dataclass(frozen=True)
class CleanReharvestConfig:
    output_dir: Path
    tiff_path: Path
    pipeline_config_path: Path
    checkpoint_path: Path | None = None
    device: str = "cpu"
    frame_start: int = 0
    frame_stop: int | None = 512
    spatial_tile_size: int = 64
    temporal_batch_size: int = 64
    spatial_overlap_px: int = 8
    spatial_tile_batch_size: int = 64
    input_offset: float | None = None
    input_scale: float | None = None
    raw_th: float | None = None
    split_th: float | None = None
    accept_prob_threshold: float = 0.7
    phot_min: float = 100.0
    sigma_max_px: float = 2.5
    clean_library_target: int = 256
    patch_size_px: int = 15
    min_neighbor_distance_px: float = 15.0
    baseline_adu: float | None = None
    gallery_count: int = 16
    crop_x0: int = 0
    crop_x1: int | None = None
    crop_y0: int = 0
    crop_y1: int | None = None


@dataclass(frozen=True)
class CleanReharvestResult:
    config: CleanReharvestConfig
    harvest_metrics: dict[str, Any]
    clean_selection_metrics: dict[str, Any]
    clean_patch_count: int
    output_dir: Path
    figures: dict[str, str]

    def to_serializable_dict(self) -> dict[str, Any]:
        return {
            "config": {
                **asdict(self.config),
                "output_dir": str(self.config.output_dir),
                "tiff_path": str(self.config.tiff_path),
                "pipeline_config_path": str(self.config.pipeline_config_path),
                "checkpoint_path": None if self.config.checkpoint_path is None else str(self.config.checkpoint_path),
            },
            "harvest_metrics": self.harvest_metrics,
            "clean_selection_metrics": self.clean_selection_metrics,
            "clean_patch_count": int(self.clean_patch_count),
            "figures": self.figures,
            "output_dir": str(self.output_dir),
        }


def _save_patch_gallery(path: Path, patches: torch.Tensor, *, max_count: int) -> None:
    count = min(int(max_count), int(patches.shape[0]))
    if count <= 0:
        return
    cols = min(4, count)
    rows = int(np.ceil(count / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(2.2 * cols, 2.2 * rows), dpi=160, constrained_layout=True)
    axes = np.atleast_1d(axes).reshape(rows, cols)
    for index in range(rows * cols):
        ax = axes[index // cols, index % cols]
        ax.set_xticks([])
        ax.set_yticks([])
        if index >= count:
            ax.axis("off")
            continue
        panel = patches[index].detach().cpu().numpy()
        vmin = float(np.percentile(panel, 1))
        vmax = float(np.percentile(panel, 99))
        ax.imshow(panel, cmap="magma", vmin=vmin, vmax=vmax)
        h = panel.shape[0]
        w = panel.shape[1]
        ax.axhline((h - 1) / 2.0, color="white", lw=0.5, alpha=0.5)
        ax.axvline((w - 1) / 2.0, color="white", lw=0.5, alpha=0.5)
        ax.set_title(f"clean spot {index}", fontsize=8)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def _save_probability_hist(path: Path, probabilities: torch.Tensor) -> None:
    fig, ax = plt.subplots(figsize=(5, 3.2), dpi=160, constrained_layout=True)
    ax.hist(probabilities.detach().cpu().numpy(), bins=20, color="#4C78A8", edgecolor="white")
    ax.set_xlabel("accepted probability")
    ax.set_ylabel("count")
    ax.set_title("Harvest probability distribution")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def _save_report(path: Path, result: CleanReharvestResult) -> None:
    lines = [
        "# Clean reharvest report",
        "",
        "## Summary",
        "",
        f"- output: `{result.output_dir}`",
        f"- frame range: `{result.config.frame_start}:{result.config.frame_stop}`",
        f"- candidate_count: `{result.harvest_metrics.get('candidate_count')}`",
        f"- accepted_count: `{result.harvest_metrics.get('accepted_count')}`",
        f"- clean_patch_count: `{result.clean_patch_count}`",
        f"- isolated_count: `{result.clean_selection_metrics.get('isolated_count')}`",
        f"- image_quality_count: `{result.clean_selection_metrics.get('image_quality_count')}`",
        f"- rejected_by_image_quality: `{result.clean_selection_metrics.get('rejected_by_image_quality')}`",
        "",
        "## Figures",
        "",
    ]
    for name, figure_path in result.figures.items():
        lines.append(f"- {name}: `{figure_path}`")
    lines += [
        "",
        "## Visual Preview",
        "",
        "### Clean patch gallery",
        "",
        "![clean patches](figures/clean_patch_gallery.png)",
        "",
        "### Harvest probability histogram",
        "",
        "![harvest histogram](figures/harvest_probability_hist.png)",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_clean_reharvest(config: CleanReharvestConfig) -> CleanReharvestResult:
    output_dir = Path(config.output_dir)
    figures_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    with Path(config.pipeline_config_path).open("r", encoding="utf-8") as handle:
        pipeline_cfg = json.load(handle)
    model = build_density1_starter_model(
        checkpoint_path=config.checkpoint_path,
        device=config.device,
        cfg=pipeline_cfg,
    )
    harvester_cfg = build_starter_harvester_config(
        pipeline_config_path=config.pipeline_config_path,
        tiff_path=config.tiff_path,
        device=config.device,
        spatial_tile_size=config.spatial_tile_size,
        temporal_batch_size=config.temporal_batch_size,
        frame_start=config.frame_start,
        frame_stop=config.frame_stop,
        spatial_overlap_px=config.spatial_overlap_px,
        spatial_tile_batch_size=config.spatial_tile_batch_size,
        input_offset=config.input_offset,
        input_scale=config.input_scale,
        raw_th=config.raw_th,
        split_th=config.split_th,
        accept_prob_threshold=config.accept_prob_threshold,
        phot_min=config.phot_min,
        sigma_max_px=config.sigma_max_px,
        crop_x0=config.crop_x0,
        crop_x1=config.crop_x1,
        crop_y0=config.crop_y0,
        crop_y1=config.crop_y1,
    )
    harvest = harvest_tiff_points(model, harvester_cfg)
    harvest_payload = {
        "candidate_count": int(harvest.candidate_count),
        "accepted_count": int(harvest.accepted_count),
        "metrics": dict(harvest.metrics),
        "payload": {key: getattr(harvest.emitters, key) for key in harvest.emitters.__dataclass_fields__.keys()},
    }
    torch.save(harvest_payload, output_dir / "harvest.pt")

    fitter_cfg = build_nat_spot_fitter_config(
        pipeline_config_path=config.pipeline_config_path,
        tiff_path=config.tiff_path,
        device=config.device,
        patch_size_px=config.patch_size_px,
        max_emitters=config.clean_library_target,
        min_neighbor_distance_px=config.min_neighbor_distance_px,
        baseline_adu=config.baseline_adu,
        freeze_initial_astig_standard=True,
    )
    patch_dataset = build_patch_dataset_from_emitters(harvest.emitters, fitter_cfg)
    clean_patch_count = int(patch_dataset.patches_norm.shape[0])

    gallery_path = figures_dir / "clean_patch_gallery.png"
    hist_path = figures_dir / "harvest_probability_hist.png"
    _save_patch_gallery(gallery_path, patch_dataset.patches_norm, max_count=config.gallery_count)
    _save_probability_hist(hist_path, harvest.emitters.probability)

    result = CleanReharvestResult(
        config=config,
        harvest_metrics={
            **dict(harvest.metrics),
            "candidate_count": int(harvest.candidate_count),
            "accepted_count": int(harvest.accepted_count),
        },
        clean_selection_metrics=dict(patch_dataset.selection_metrics),
        clean_patch_count=clean_patch_count,
        output_dir=output_dir,
        figures={
            "clean_patch_gallery": str(gallery_path),
            "harvest_probability_hist": str(hist_path),
        },
    )
    (output_dir / "clean_reharvest_summary.json").write_text(
        json.dumps(result.to_serializable_dict(), indent=2) + "\n",
        encoding="utf-8",
    )
    _save_report(output_dir / "report.md", result)
    return result


__all__ = [
    "CleanReharvestConfig",
    "CleanReharvestResult",
    "run_clean_reharvest",
]
