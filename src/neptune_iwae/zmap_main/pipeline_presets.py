from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
MICROTUBE_ASSET_DIR = HERE / "assets" / "microtube"
MICROTUBE_PIPELINE_CONFIG = MICROTUBE_ASSET_DIR / "pipeline_smlm_uni_decode_aligned_fixedscale_depthwise_mu20000_sigma1000.json"
MICROTUBE_CHECKPOINT = MICROTUBE_ASSET_DIR / "checkpoint_best_jaccard.pt"
TRIPPLE_COLOR_PIPELINE_CONFIG = Path("neptune/starter/pipeline_density_1_real_sample_online_mu20000_depthwise_film.json")
TRIPPLE_COLOR_CHECKPOINT = Path("neptune/starter/checkpoint_best_jaccard.pt")
TRIPPLE_COLOR_TIFF = Path("datasets/raw_samples/sample/tripple_color/spool_400mW_30ms_3D_6_MMStack_Default.ome.tif")
PAINT_TIFF = Path("datasets/samples/PAINT/spool_200mW_30ms_3D_-lh1_1_MMStack_Default.ome.tif")
NCP_TIFF = Path("datasets/samples/sample_ncp/spool_500mw-30 ms_3D_1_1_MMStack_Default.ome.tif")
DYNAMIN_TIFF = Path("datasets/training_sets/dynamin_3d_5_1")
MEMBRANE_TIFF = Path("datasets/training_sets/membrane_3d_4_1")


@dataclass(frozen=True)
class MicrotubuleZMapPreset:
    side: str
    tiff_path: Path
    pipeline_config_path: Path
    checkpoint_path: Path
    frame_start: int
    frame_stop: int
    temporal_batch_size: int
    spatial_tile_batch_size: int
    clean_library_target: int
    gallery_count: int
    crop_x0: int
    crop_x1: int | None
    crop_y0: int
    crop_y1: int | None
    input_offset: float | None
    input_scale: float | None
    nat_config_kind: str
    max_emitters: int
    min_neighbor_distance_px: float
    global_projected_min_distance_px: float
    selection_pool_multiplier: int
    selection_mode: str
    spatial_balance_grid_px: int
    spatial_balance_min_per_cell: int
    spatial_balance_max_per_cell: int
    max_patch_peak_distance_px: float
    secondary_peak_exclusion_radius_px: float
    max_secondary_peak_fraction: float
    rank_spots_by_quality: bool
    roi_x_min_px: float
    roi_x_max_px: float
    roi_y_min_px: float
    roi_y_max_px: float
    roi_edge_margin_px: float
    alternating_rounds: int
    alternating_local_steps: int
    alternating_global_steps: int
    accept_prob_threshold: float = 0.7
    phot_min: float = 100.0
    sigma_max_px: float = 2.5
    baseline_adu: float | None = None


ZMapPipelinePreset = MicrotubuleZMapPreset


def build_microtubule_preset(side: str) -> MicrotubuleZMapPreset:
    side_norm = str(side).strip().lower()
    common = {
        "tiff_path": Path("datasets/samples/sample/spool_800mW_30ms_3D_7_1_MMStack_Default_1.ome.tif"),
        "pipeline_config_path": MICROTUBE_PIPELINE_CONFIG,
        "checkpoint_path": MICROTUBE_CHECKPOINT,
        "gallery_count": 16,
        "crop_y0": 0,
        "crop_y1": None,
        "nat_config_kind": "order2",
        "secondary_peak_exclusion_radius_px": 3.5,
        "roi_y_min_px": 0.0,
        "roi_y_max_px": 1200.0,
        "roi_edge_margin_px": 3.0,
    }
    if side_norm == "left":
        return MicrotubuleZMapPreset(
            side="left",
            frame_start=0,
            frame_stop=1000,
            temporal_batch_size=128,
            spatial_tile_batch_size=16,
            clean_library_target=256,
            crop_x0=0,
            crop_x1=None,
            input_offset=None,
            input_scale=None,
            max_emitters=1400,
            min_neighbor_distance_px=5.0,
            global_projected_min_distance_px=0.0,
            selection_pool_multiplier=10,
            selection_mode="spatial_balanced",
            spatial_balance_grid_px=128,
            spatial_balance_min_per_cell=6,
            spatial_balance_max_per_cell=0,
            max_patch_peak_distance_px=2.5,
            max_secondary_peak_fraction=0.45,
            rank_spots_by_quality=False,
            roi_x_min_px=0.0,
            roi_x_max_px=600.0,
            alternating_rounds=30,
            alternating_local_steps=100,
            alternating_global_steps=100,
            **common,
        )
    if side_norm == "right":
        return MicrotubuleZMapPreset(
            side="right",
            frame_start=0,
            frame_stop=1000,
            temporal_batch_size=128,
            spatial_tile_batch_size=16,
            clean_library_target=256,
            crop_x0=600,
            crop_x1=1200,
            input_offset=257.0,
            input_scale=1800.0,
            max_emitters=1400,
            min_neighbor_distance_px=7.0,
            global_projected_min_distance_px=0.0,
            selection_pool_multiplier=50,
            selection_mode="spatial_balanced",
            spatial_balance_grid_px=128,
            spatial_balance_min_per_cell=6,
            spatial_balance_max_per_cell=0,
            max_patch_peak_distance_px=1.2,
            max_secondary_peak_fraction=0.25,
            rank_spots_by_quality=False,
            roi_x_min_px=600.0,
            roi_x_max_px=1200.0,
            alternating_rounds=20,
            alternating_local_steps=100,
            alternating_global_steps=100,
            **common,
        )
    raise ValueError(f"Unsupported microtubule side: {side!r}")


def build_tripple_color_preset(side: str) -> MicrotubuleZMapPreset:
    side_norm = str(side).strip().lower()
    common = {
        "tiff_path": TRIPPLE_COLOR_TIFF,
        "pipeline_config_path": TRIPPLE_COLOR_PIPELINE_CONFIG,
        "checkpoint_path": TRIPPLE_COLOR_CHECKPOINT,
        "frame_start": 0,
        "frame_stop": 1000,
        "temporal_batch_size": 128,
        "spatial_tile_batch_size": 16,
        "clean_library_target": 256,
        "gallery_count": 16,
        "crop_y0": 0,
        "crop_y1": None,
        "input_offset": 257.0,
        "nat_config_kind": "order1",
        "max_emitters": 1400,
        "min_neighbor_distance_px": 0.0,
        "global_projected_min_distance_px": 0.0,
        "selection_pool_multiplier": 50,
        "selection_mode": "spatial_balanced",
        "spatial_balance_grid_px": 128,
        "spatial_balance_min_per_cell": 5,
        "spatial_balance_max_per_cell": 0,
        "max_patch_peak_distance_px": 99.0,
        "secondary_peak_exclusion_radius_px": 3.5,
        "max_secondary_peak_fraction": 99.0,
        "rank_spots_by_quality": False,
        "roi_y_min_px": 0.0,
        "roi_y_max_px": 1200.0,
        "roi_edge_margin_px": 3.0,
        "alternating_rounds": 35,
        "alternating_local_steps": 120,
        "alternating_global_steps": 120,
    }
    if side_norm == "left":
        return MicrotubuleZMapPreset(
            side="left",
            crop_x0=0,
            crop_x1=600,
            input_scale=500.0,
            roi_x_min_px=0.0,
            roi_x_max_px=600.0,
            **common,
        )
    if side_norm == "right":
        return MicrotubuleZMapPreset(
            side="right",
            crop_x0=600,
            crop_x1=1200,
            input_scale=450.0,
            roi_x_min_px=600.0,
            roi_x_max_px=1200.0,
            **common,
        )
    raise ValueError(f"Unsupported tripple_color side: {side!r}")


def build_paint_preset(side: str) -> MicrotubuleZMapPreset:
    side_norm = str(side).strip().lower()
    common = {
        "tiff_path": PAINT_TIFF,
        "pipeline_config_path": MICROTUBE_PIPELINE_CONFIG,
        "checkpoint_path": MICROTUBE_CHECKPOINT,
        "frame_start": 0,
        "frame_stop": 100,
        "temporal_batch_size": 128,
        "spatial_tile_batch_size": 16,
        "clean_library_target": 1000,
        "gallery_count": 16,
        "crop_y0": 0,
        "crop_y1": None,
        "nat_config_kind": "order2",
        "max_emitters": 1000,
        "global_projected_min_distance_px": 0.0,
        "selection_mode": "spatial_balanced",
        "spatial_balance_grid_px": 128,
        "spatial_balance_min_per_cell": 6,
        "spatial_balance_max_per_cell": 0,
        "secondary_peak_exclusion_radius_px": 3.5,
        "rank_spots_by_quality": False,
        "roi_y_min_px": 0.0,
        "roi_y_max_px": 1200.0,
        "roi_edge_margin_px": 3.0,
        "accept_prob_threshold": 0.3,
        "phot_min": 20.0,
        "sigma_max_px": 2.5,
        "alternating_local_steps": 100,
        "alternating_global_steps": 100,
    }
    if side_norm == "left":
        return MicrotubuleZMapPreset(
            side="left",
            crop_x0=0,
            crop_x1=600,
            input_offset=130.0,
            input_scale=700.0,
            min_neighbor_distance_px=5.0,
            selection_pool_multiplier=10,
            max_patch_peak_distance_px=2.5,
            max_secondary_peak_fraction=0.45,
            roi_x_min_px=0.0,
            roi_x_max_px=600.0,
            alternating_rounds=30,
            baseline_adu=135.0,
            **common,
        )
    if side_norm == "right":
        return MicrotubuleZMapPreset(
            side="right",
            crop_x0=600,
            crop_x1=1200,
            input_offset=257.0,
            input_scale=600.0,
            min_neighbor_distance_px=5.0,
            selection_pool_multiplier=10,
            max_patch_peak_distance_px=2.5,
            max_secondary_peak_fraction=0.45,
            roi_x_min_px=600.0,
            roi_x_max_px=1200.0,
            alternating_rounds=30,
            baseline_adu=125.0,
            **common,
        )
    raise ValueError(f"Unsupported paint side: {side!r}")


def build_ncp_preset(side: str) -> MicrotubuleZMapPreset:
    side_norm = str(side).strip().lower()
    common = {
        "tiff_path": NCP_TIFF,
        "pipeline_config_path": TRIPPLE_COLOR_PIPELINE_CONFIG,
        "checkpoint_path": TRIPPLE_COLOR_CHECKPOINT,
        "frame_start": 0,
        "frame_stop": 1000,
        "temporal_batch_size": 128,
        "spatial_tile_batch_size": 16,
        "clean_library_target": 450,
        "gallery_count": 16,
        "crop_y0": 400,
        "crop_y1": 800,
        "input_offset": 257.0,
        "input_scale": 1500.0,
        "nat_config_kind": "order2",
        "max_emitters": 500,
        "min_neighbor_distance_px": 9.0,
        "global_projected_min_distance_px": 0.0,
        "selection_pool_multiplier": 100,
        "selection_mode": "spatial_balanced",
        "spatial_balance_grid_px": 128,
        "spatial_balance_min_per_cell": 0,
        "spatial_balance_max_per_cell": 0,
        "max_patch_peak_distance_px": 2.5,
        "secondary_peak_exclusion_radius_px": 3.5,
        "max_secondary_peak_fraction": 0.55,
        "rank_spots_by_quality": True,
        "roi_y_min_px": 400.0,
        "roi_y_max_px": 800.0,
        "roi_edge_margin_px": 3.0,
        "alternating_rounds": 8,
        "alternating_local_steps": 40,
        "alternating_global_steps": 40,
    }
    if side_norm == "left":
        return MicrotubuleZMapPreset(
            side="left",
            crop_x0=100,
            crop_x1=500,
            roi_x_min_px=100.0,
            roi_x_max_px=500.0,
            **common,
        )
    if side_norm == "right":
        return MicrotubuleZMapPreset(
            side="right",
            crop_x0=700,
            crop_x1=1100,
            roi_x_min_px=700.0,
            roi_x_max_px=1100.0,
            **common,
        )
    raise ValueError(f"Unsupported ncp side: {side!r}")


def build_full_fov_training_sample_preset(sample: str, side: str) -> MicrotubuleZMapPreset:
    sample_norm = str(sample).strip().lower()
    side_norm = str(side).strip().lower()
    if sample_norm == "dynamin":
        tiff_path = DYNAMIN_TIFF
    elif sample_norm == "membrane":
        tiff_path = MEMBRANE_TIFF
    else:
        raise ValueError(f"Unsupported full-FOV training sample: {sample!r}")
    common = {
        "tiff_path": tiff_path,
        "pipeline_config_path": MICROTUBE_PIPELINE_CONFIG,
        "checkpoint_path": MICROTUBE_CHECKPOINT,
        "frame_start": 0,
        "frame_stop": 1000,
        "temporal_batch_size": 128,
        "spatial_tile_batch_size": 16,
        "clean_library_target": 1000,
        "gallery_count": 16,
        "crop_y0": 0,
        "crop_y1": None,
        "input_offset": None,
        "input_scale": None,
        "nat_config_kind": "order2",
        "max_emitters": 500,
        "min_neighbor_distance_px": 5.0,
        "global_projected_min_distance_px": 0.0,
        "selection_pool_multiplier": 50,
        "selection_mode": "spatial_balanced",
        "spatial_balance_grid_px": 128,
        "spatial_balance_min_per_cell": 0,
        "spatial_balance_max_per_cell": 0,
        "max_patch_peak_distance_px": 2.5,
        "secondary_peak_exclusion_radius_px": 3.5,
        "max_secondary_peak_fraction": 0.45,
        "rank_spots_by_quality": True,
        "roi_y_min_px": 0.0,
        "roi_y_max_px": 1200.0,
        "roi_edge_margin_px": 3.0,
        "alternating_rounds": 20,
        "alternating_local_steps": 100,
        "alternating_global_steps": 100,
    }
    if side_norm == "left":
        return MicrotubuleZMapPreset(
            side="left",
            crop_x0=0,
            crop_x1=600,
            roi_x_min_px=0.0,
            roi_x_max_px=600.0,
            **common,
        )
    if side_norm == "right":
        return MicrotubuleZMapPreset(
            side="right",
            crop_x0=600,
            crop_x1=1200,
            roi_x_min_px=600.0,
            roi_x_max_px=1200.0,
            **common,
        )
    raise ValueError(f"Unsupported {sample_norm} side: {side!r}")


def build_zmap_preset(sample: str, side: str) -> ZMapPipelinePreset:
    sample_norm = str(sample).strip().lower().replace("-", "_")
    if sample_norm in {"microtube", "microtubule"}:
        return build_microtubule_preset(side)
    if sample_norm == "paint":
        return build_paint_preset(side)
    if sample_norm == "ncp":
        return build_ncp_preset(side)
    if sample_norm in {"dynamin", "membrane"}:
        return build_full_fov_training_sample_preset(sample_norm, side)
    if sample_norm in {"tripple_color", "triple_color"}:
        return build_tripple_color_preset(side)
    raise ValueError(f"Unsupported zmap sample: {sample!r}")
