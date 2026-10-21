from pathlib import Path

import numpy as np
import os
import pytest
import subprocess
import sys
import torch

from simulation.api import SimulationDataset, SimulationSample


RIGHT_SUMMARY = Path(
    "output/neptune/physical_model_8000_split_channels/right_fdpsf/block_span_1600/summary.json"
)


def test_right_fd_asset_exposes_roi_metadata():
    from smlm_v2a.fd_conditioning.assets import load_fd_channel_asset

    asset = load_fd_channel_asset(RIGHT_SUMMARY)

    assert asset.crop_left == 600
    assert asset.crop_top == 0
    assert asset.crop_width == 600
    assert asset.crop_height == 1200
    assert asset.field_shape == (30, 15)
    assert asset.cell_width_px == 40
    assert asset.cell_height_px == 40


def test_right_fd_coord_maps_follow_absolute_roi_position():
    from smlm_v2a.fd_conditioning.assets import load_fd_channel_asset
    from smlm_v2a.fd_conditioning.coords import build_coord_channels

    asset = load_fd_channel_asset(RIGHT_SUMMARY)
    coords = build_coord_channels(asset, field_xy=torch.tensor([14.0, 29.0]), out_size=(40, 40))

    assert coords.shape == (2, 40, 40)
    assert torch.isfinite(coords).all()
    assert torch.all(coords >= 0.0)
    assert torch.all(coords <= 1.0)
    assert coords[0, 0, 0] < coords[0, 0, -1]
    assert coords[1, 0, 0] < coords[1, -1, 0]
    assert coords[0, 0, 0] == pytest.approx((560.0) / 599.0, rel=1e-5)
    assert coords[1, 0, 0] == pytest.approx((1160.0) / 1199.0, rel=1e-5)


def test_right_fd_input_channels_expand_window_by_two():
    from smlm_v2a.fd_conditioning.coords import append_coord_channels

    frames = torch.zeros((3, 40, 40), dtype=torch.float32)
    coords = torch.ones((2, 40, 40), dtype=torch.float32)

    stacked = append_coord_channels(frames, coords)

    assert stacked.shape == (5, 40, 40)
    assert torch.allclose(stacked[:3], frames)
    assert torch.allclose(stacked[3:], coords)


def test_right_fd_config_exists_and_targets_single_right_asset():
    import json

    cfg_path = Path("config/pipeline_fd_right_smlm_v2a.json")
    payload = json.loads(cfg_path.read_text(encoding="utf-8"))

    assert payload["fd_right_channel"]["enabled"] is True
    assert payload["fd_right_channel"]["summary_path"].endswith(
        "output/neptune/physical_model_8000_split_channels/right_fdpsf/block_span_1600/summary.json"
    )
    assert payload["fd_right_channel"]["crop_left"] == 600
    assert payload["fd_right_channel"]["crop_top"] == 0
    assert payload["fd_right_channel"]["crop_width"] == 600
    assert payload["fd_right_channel"]["crop_height"] == 1200
    assert payload["simulation"]["psf"]["vector"]["mean_psf"]["enabled"] is False


def _make_single_right_sample_dataset() -> SimulationDataset:
    frames = np.arange(3 * 40 * 40, dtype=np.float32).reshape(3, 40, 40)
    param_tar = np.zeros((3, 2, 4), dtype=np.float32)
    mask_tar = np.zeros((3, 2), dtype=bool)
    field_tar = np.zeros((3, 2, 2), dtype=np.float32)
    bg_tar = np.zeros((3, 40, 40), dtype=np.float32)

    param_tar[1, 0] = np.asarray([1200.0, 12.0, 21.0, 0.1], dtype=np.float32)
    mask_tar[1, 0] = True
    field_tar[:, :, 0] = 14.0
    field_tar[:, :, 1] = 29.0

    return SimulationDataset(
        frames_per_sample=3,
        num_samples=1,
        profile="train",
        samples=[
            SimulationSample(
                sample_id="right-sample-0",
                frames=frames,
                param_tar=param_tar,
                mask_tar=mask_tar,
                field_tar=field_tar,
                bg_tar=bg_tar,
            )
        ],
    )


def test_right_fd_dataset_appends_xy_channels_and_keeps_single_field_cell():
    from smlm_v2a.fd_conditioning.assets import load_fd_channel_asset
    from smlm_v2a.fd_conditioning.coords import build_coord_channels
    from smlm_v2a.training.fd_right_train_pipeline import RightFDDataset

    asset = load_fd_channel_asset(RIGHT_SUMMARY)
    dataset = RightFDDataset(
        _make_single_right_sample_dataset(),
        asset=asset,
        frame_proc=None,
        bkg_frame_proc=None,
        tar_proc=None,
        window_size=3,
        coord_size=(40, 40),
    )

    frames, detect, bkg, param, mask, field_xy = dataset[0]

    assert len(dataset) == 1
    assert frames.shape == (5, 40, 40)
    assert detect.shape == (40, 40)
    assert bkg.shape == (40, 40)
    assert param.shape == (2, 4)
    assert mask.shape == (2,)
    assert torch.equal(field_xy, torch.tensor([14.0, 29.0], dtype=torch.float32))
    assert int(detect.sum().item()) == 1
    assert torch.allclose(
        frames[3:],
        build_coord_channels(asset, field_xy=field_xy, out_size=(40, 40)),
    )


def test_right_fd_inference_helpers_crop_full_frame_and_tile_roi():
    from smlm_v2a.fd_conditioning.assets import load_fd_channel_asset
    from tools.run_fd_right_sample_inference import _build_right_roi_tiles, _crop_tiff_to_right_roi

    asset = load_fd_channel_asset(RIGHT_SUMMARY)
    full_stack = np.arange(2 * 1200 * 1200, dtype=np.uint16).reshape(2, 1200, 1200)

    cropped = _crop_tiff_to_right_roi(
        full_stack,
        crop_left=asset.crop_left,
        crop_top=asset.crop_top,
        crop_width=asset.crop_width,
        crop_height=asset.crop_height,
    )
    tiles = _build_right_roi_tiles(asset, roi_size=40)

    assert cropped.shape == (2, 1200, 600)
    assert np.array_equal(cropped, full_stack[:, 0:1200, 600:1200])
    assert len(tiles) == 30 * 15
    assert tiles[0]["tile_y0"] == 0
    assert tiles[0]["tile_x0"] == 0
    assert tiles[-1]["tile_y0"] == 29 * 40
    assert tiles[-1]["tile_x0"] == 14 * 40
    assert tiles[0]["coords"].shape == (2, 40, 40)
    assert tiles[-1]["coords"].shape == (2, 40, 40)


def test_merged_right_sample_inference_helpers_sort_inputs_and_offsets():
    from tools.run_smlm_uni_merged_right_sample_inference import (
        _build_frame_offsets,
        _resolve_logical_ome_root,
        _sort_sample_tiffs,
    )

    tiffs = [
        Path("datasets/raw_samples/sample/spool_800mW_30ms_3D_7_1_MMStack_Default_5.ome.tif"),
        Path("datasets/raw_samples/sample/spool_800mW_30ms_3D_7_1_MMStack_Default.ome.tif"),
        Path("datasets/raw_samples/sample/spool_800mW_30ms_3D_7_1_MMStack_Default_2.ome.tif"),
    ]

    sorted_tiffs = _sort_sample_tiffs(tiffs)

    assert [path.name for path in sorted_tiffs] == [
        "spool_800mW_30ms_3D_7_1_MMStack_Default.ome.tif",
        "spool_800mW_30ms_3D_7_1_MMStack_Default_2.ome.tif",
        "spool_800mW_30ms_3D_7_1_MMStack_Default_5.ome.tif",
    ]
    assert _resolve_logical_ome_root(tiffs).name == "spool_800mW_30ms_3D_7_1_MMStack_Default.ome.tif"
    assert _build_frame_offsets([8000, 8000, 8000]) == [0, 8000, 16000]


def test_merged_right_sample_inference_remaps_batched_locs_to_global_rows():
    from smlm_v2a.generic.localizations import FrameLocSet
    from tools.run_smlm_uni_merged_right_sample_inference import _rows_from_batched_locs

    locs = FrameLocSet(
        ndim=3,
        xextent=(-0.5, 39.5),
        yextent=(-0.5, 39.5),
        zextent=(-0.6, 0.6),
        id=torch.tensor([0, 1], dtype=torch.int32),
        prob=torch.tensor([0.8, 0.9], dtype=torch.float32),
        indf=torch.tensor([0, 2], dtype=torch.int32),
        xyz=torch.tensor([[1.5, 2.5, -0.1], [3.0, 4.0, 0.2]], dtype=torch.float32),
        phot=torch.tensor([1000.0, 2000.0], dtype=torch.float32),
        xyz_sig=torch.zeros((2, 3), dtype=torch.float32),
        phot_sig=torch.zeros((2,), dtype=torch.float32),
    )
    metas = [
        (100, 40, 80),
        (100, 40, 120),
        (101, 200, 240),
    ]

    rows = _rows_from_batched_locs(
        locs,
        metas=metas,
        crop_left=600,
        sample_name="sample0.ome.tif",
    )

    assert rows[0]["frame"] == 100
    assert rows[0]["x_px"] == pytest.approx(81.5)
    assert rows[0]["y_px"] == pytest.approx(42.5)
    assert rows[0]["x_px_full"] == pytest.approx(681.5)
    assert rows[1]["frame"] == 101
    assert rows[1]["x_px"] == pytest.approx(243.0)
    assert rows[1]["y_px"] == pytest.approx(204.0)
    assert rows[1]["sample_tiff"] == "sample0.ome.tif"


def test_merged_right_sample_inference_builds_64_context_tiles_over_64_grid():
    from tools.run_smlm_uni_merged_right_sample_inference import _build_context_tiles

    tiles = _build_context_tiles(crop_height=1152, crop_width=576, context_size=64, valid_size=64)

    assert len(tiles) == 18 * 9
    assert tiles[0] == {
        "patch_y0": 0,
        "patch_x0": 0,
        "keep_y0": 0,
        "keep_x0": 0,
        "valid_size": 64,
    }
    assert tiles[1] == {
        "patch_y0": 0,
        "patch_x0": 64,
        "keep_y0": 0,
        "keep_x0": 0,
        "valid_size": 64,
    }
    assert tiles[-1] == {
        "patch_y0": 1088,
        "patch_x0": 512,
        "keep_y0": 0,
        "keep_x0": 0,
        "valid_size": 64,
    }

    coverage = np.zeros((1152, 576), dtype=np.int16)
    for tile in tiles:
        y0 = tile["patch_y0"] + tile["keep_y0"]
        x0 = tile["patch_x0"] + tile["keep_x0"]
        coverage[y0 : y0 + tile["valid_size"], x0 : x0 + tile["valid_size"]] += 1

    assert np.all(coverage == 1)


def test_fd_left_overlap_inference_builds_edgecover_tiles_for_600x1200_valid96():
    from tools.run_fd_left_fullres_sample_inference import _build_edgecover_context_tiles

    tiles = _build_edgecover_context_tiles(
        crop_height=1200,
        crop_width=600,
        context_size=128,
        valid_size=96,
    )

    assert len(tiles) == 13 * 7
    assert tiles[0] == {
        "tile_index": 0,
        "patch_y0": 0,
        "patch_x0": 0,
        "keep_y0": 0,
        "keep_x0": 0,
        "keep_h": 96,
        "keep_w": 96,
        "valid_y0": 0,
        "valid_x0": 0,
    }
    assert tiles[-1] == {
        "tile_index": 90,
        "patch_y0": 1072,
        "patch_x0": 472,
        "keep_y0": 80,
        "keep_x0": 104,
        "keep_h": 48,
        "keep_w": 24,
        "valid_y0": 1152,
        "valid_x0": 576,
    }

    coverage = np.zeros((1200, 600), dtype=np.int16)
    for tile in tiles:
        y0 = tile["patch_y0"] + tile["keep_y0"]
        x0 = tile["patch_x0"] + tile["keep_x0"]
        coverage[y0 : y0 + tile["keep_h"], x0 : x0 + tile["keep_w"]] += 1

    assert np.all(coverage == 1)


def test_sample_inference_defaults_to_training_roi_size():
    from tools.diagnose_fd_sample_thresholds import _parse_args as parse_diagnostic_args
    from tools.run_smlm_uni_merged_right_sample_inference import _parse_args as parse_inference_args

    original_argv = sys.argv
    try:
        sys.argv = ["prog", "--output-dir", "out"]
        diagnostic_args = parse_diagnostic_args()
        assert diagnostic_args.roi_size == 64
        assert diagnostic_args.valid_roi_size == 64

        sys.argv = ["prog", "--checkpoint", "ckpt.pt", "--output-dir", "out"]
        inference_args = parse_inference_args()
        assert inference_args.roi_size == 64
        assert inference_args.valid_roi_size is None
    finally:
        sys.argv = original_argv


def test_merged_right_sample_inference_filters_context_locs_to_valid_region():
    from smlm_v2a.generic.localizations import FrameLocSet
    from tools.run_smlm_uni_merged_right_sample_inference import _rows_from_batched_context_locs

    locs = FrameLocSet(
        ndim=3,
        xextent=(-0.5, 63.5),
        yextent=(-0.5, 63.5),
        zextent=(-0.6, 0.6),
        id=torch.tensor([0, 1, 2], dtype=torch.int32),
        prob=torch.tensor([0.8, 0.7, 0.9], dtype=torch.float32),
        indf=torch.tensor([0, 0, 1], dtype=torch.int32),
        xyz=torch.tensor(
            [
                [12.5, 12.5, -0.1],
                [5.0, 5.0, 0.0],
                [25.0, 30.0, 0.2],
            ],
            dtype=torch.float32,
        ),
        phot=torch.tensor([1000.0, 800.0, 2000.0], dtype=torch.float32),
        xyz_sig=torch.zeros((3, 3), dtype=torch.float32),
        phot_sig=torch.zeros((3,), dtype=torch.float32),
    )
    metas = [
        {"frame_id": 100, "patch_y0": 28, "patch_x0": 48, "keep_y0": 12, "keep_x0": 12, "valid_size": 40},
        {"frame_id": 101, "patch_y0": 1136, "patch_x0": 536, "keep_y0": 24, "keep_x0": 24, "valid_size": 40},
    ]

    rows = _rows_from_batched_context_locs(
        locs,
        metas=metas,
        crop_left=600,
        sample_name="sample0.ome.tif",
    )

    assert len(rows) == 2
    assert rows[0]["frame"] == 100
    assert rows[0]["x_px"] == pytest.approx(60.5)
    assert rows[0]["y_px"] == pytest.approx(40.5)
    assert rows[0]["x_px_full"] == pytest.approx(660.5)
    assert rows[1]["frame"] == 101
    assert rows[1]["x_px"] == pytest.approx(561.0)
    assert rows[1]["y_px"] == pytest.approx(1166.0)
    assert rows[1]["sample_tiff"] == "sample0.ome.tif"


def test_fd_left_overlap_filters_context_locs_to_rectangular_edge_keep_region():
    from smlm_v2a.generic.localizations import FrameLocSet
    from tools.run_fd_left_fullres_sample_inference import _rows_from_batched_context_locs

    locs = FrameLocSet(
        ndim=3,
        xextent=(-0.5, 127.5),
        yextent=(-0.5, 127.5),
        zextent=(-0.6, 0.6),
        id=torch.tensor([0, 1, 2], dtype=torch.int32),
        prob=torch.tensor([0.8, 0.7, 0.9], dtype=torch.float32),
        indf=torch.tensor([0, 0, 0], dtype=torch.int32),
        xyz=torch.tensor(
            [
                [110.0, 90.0, -0.1],
                [120.0, 90.0, 0.0],
                [109.0, 127.0, 0.2],
            ],
            dtype=torch.float32,
        ),
        phot=torch.tensor([1000.0, 800.0, 2000.0], dtype=torch.float32),
        xyz_sig=torch.zeros((3, 3), dtype=torch.float32),
        phot_sig=torch.zeros((3,), dtype=torch.float32),
    )
    metas = [
        {
            "frame_id": 17,
            "patch_y0": 1072,
            "patch_x0": 472,
            "keep_y0": 80,
            "keep_x0": 104,
            "keep_h": 48,
            "keep_w": 24,
            "tile_index": 90,
        }
    ]

    rows = _rows_from_batched_context_locs(
        locs,
        metas=metas,
        crop_left=0,
        crop_top=0,
        sample_name="sample0.ome.tif",
    )

    assert len(rows) == 3
    assert rows[0]["frame"] == 17
    assert rows[0]["x_px"] == pytest.approx(582.0)
    assert rows[0]["y_px"] == pytest.approx(1162.0)
    assert rows[0]["tile_index"] == 90
    assert rows[1]["x_px"] == pytest.approx(592.0)
    assert rows[1]["y_px"] == pytest.approx(1162.0)
    assert rows[2]["x_px"] == pytest.approx(581.0)
    assert rows[2]["y_px"] == pytest.approx(1199.0)


def test_merged_sample_inference_supports_left_half_crop_geometry():
    from tools.run_smlm_uni_merged_right_sample_inference import _crop_to_selected_half, _resolve_crop_geometry

    block = np.arange(2 * 4 * 6, dtype=np.float32).reshape(2, 4, 6)

    crop_left, crop_width = _resolve_crop_geometry(full_width=6, half="left")
    cropped = _crop_to_selected_half(block, half="left")

    assert crop_left == 0
    assert crop_width == 3
    assert cropped.shape == (2, 4, 3)
    assert np.array_equal(cropped, block[:, :, :3])


def test_merged_sample_inference_supports_right_half_crop_geometry():
    from tools.run_smlm_uni_merged_right_sample_inference import _crop_to_selected_half, _resolve_crop_geometry

    block = np.arange(2 * 4 * 6, dtype=np.float32).reshape(2, 4, 6)

    crop_left, crop_width = _resolve_crop_geometry(full_width=6, half="right")
    cropped = _crop_to_selected_half(block, half="right")

    assert crop_left == 3
    assert crop_width == 3
    assert cropped.shape == (2, 4, 3)
    assert np.array_equal(cropped, block[:, :, 3:])


def test_merged_sample_inference_collapses_split_ome_parts_into_one_logical_movie(tmp_path):
    import tifffile

    from tools.run_smlm_uni_merged_right_sample_inference import _resolve_movie_inputs

    shapes = [(3, 4, 6), (2, 4, 6), (2, 4, 6)]
    paths = []
    for index, shape in enumerate(shapes):
        suffix = "" if index == 0 else f"_{index}"
        path = tmp_path / f"movie{suffix}.ome.tif"
        tifffile.imwrite(path, np.zeros(shape, dtype=np.uint16))
        paths.append(path)

    movies = _resolve_movie_inputs(paths)

    assert [movie["path"].name for movie in movies] == ["movie.ome.tif"]
    assert [movie["frame_count"] for movie in movies] == [3]
    assert [movie["frame_offset"] for movie in movies] == [0]
    assert [[part.name for part in movie["source_paths"]] for movie in movies] == [
        ["movie.ome.tif", "movie_1.ome.tif", "movie_2.ome.tif"]
    ]


def test_merged_sample_inference_keeps_distinct_logical_movie_roots(tmp_path):
    import tifffile

    from tools.run_smlm_uni_merged_right_sample_inference import _resolve_movie_inputs

    paths = []
    for stem in ["movie_a", "movie_a_1", "movie_b", "movie_b_1"]:
        path = tmp_path / f"{stem}.ome.tif"
        tifffile.imwrite(path, np.zeros((3, 4, 6), dtype=np.uint16))
        paths.append(path)

    movies = _resolve_movie_inputs(paths)

    assert [movie["path"].name for movie in movies] == ["movie_a.ome.tif", "movie_b.ome.tif"]
    assert [movie["frame_count"] for movie in movies] == [3, 3]
    assert [movie["frame_offset"] for movie in movies] == [0, 3]
    assert [[part.name for part in movie["source_paths"]] for movie in movies] == [
        ["movie_a.ome.tif", "movie_a_1.ome.tif"],
        ["movie_b.ome.tif", "movie_b_1.ome.tif"],
    ]


def test_merged_right_sample_inference_resolves_effective_input_scaling():
    from types import SimpleNamespace

    from tools.run_smlm_uni_merged_right_sample_inference import _resolve_effective_input_scaling

    param = SimpleNamespace(Scaling=SimpleNamespace(input_offset=495.5861846207447, input_scale=27331.918051340144))

    offset_default, scale_default = _resolve_effective_input_scaling(
        param,
        input_offset=None,
        input_scale=None,
    )
    offset_override, scale_override = _resolve_effective_input_scaling(
        param,
        input_offset=250.0,
        input_scale=5000.0,
    )
    offset_partial, scale_partial = _resolve_effective_input_scaling(
        param,
        input_offset=250.0,
        input_scale=None,
    )

    assert offset_default == pytest.approx(495.5861846207447)
    assert scale_default == pytest.approx(27331.918051340144)
    assert offset_override == pytest.approx(250.0)
    assert scale_override == pytest.approx(5000.0)
    assert offset_partial == pytest.approx(250.0)
    assert scale_partial == pytest.approx(27331.918051340144)


def test_merged_right_sample_inference_slurm_script_exposes_config_and_scaling_overrides():
    script = Path("scripts/slurm/slurm_smlm_uni_merged_right_sample_inference.sh").read_text(encoding="utf-8")

    assert 'CONFIG="${CONFIG:-config/pipeline_smlm_uni.json}"' in script
    assert '--config "${CONFIG}"' in script
    assert 'INPUT_OFFSET="${INPUT_OFFSET:-}"' in script
    assert 'INPUT_SCALE="${INPUT_SCALE:-}"' in script
    assert 'cmd+=(--input-offset "${INPUT_OFFSET}")' in script
    assert 'cmd+=(--input-scale "${INPUT_SCALE}")' in script


def test_v2a_norm_training_pipeline_resolves_strict_original_scaling_contract():
    from config import load_pipeline_config
    from simulation.config_types import ScalingInfo
    from smlm_v2a.training.train_pipeline import _build_param

    cfg = load_pipeline_config("config/pipeline_smlm_uni_tanh_photmaxnorm.json")
    scaling_info = ScalingInfo(
        photon_max=999.0,
        z_max=0.6,
        bg_max=240.0,
        photon_mean=10000.0,
        photon_std=1000.0,
        z_min_original=-0.6,
        z_max_original=0.6,
    )

    param = _build_param(
        cfg,
        scaling_info,
        output_dir="tmp",
        device="cuda:0",
        epochs=300,
        batch_size=32,
        window_size=3,
    )

    assert param.Scaling.input_offset == pytest.approx(110.0)
    assert param.Scaling.input_scale == pytest.approx(200.0)
    assert param.Scaling.phot_max == pytest.approx(18000.0)
    assert param.Scaling.z_max == pytest.approx(0.72)
    assert param.Scaling.bkg_max == pytest.approx(240.0)


def test_neptune_training_cfg_targets_right_half_physical_model_and_right_field_extent():
    import json
    from pathlib import Path

    train_cfg = json.loads(Path("neptune/config/smlm_uni_train_tanh_photmaxnorm.json").read_text(encoding="utf-8"))
    pipeline_cfg = json.loads(Path("config/pipeline_smlm_uni_tanh_photmaxnorm.json").read_text(encoding="utf-8"))

    assert train_cfg["initial_model"] == "neptune/output/2026-04-13-neptune-physical-model-right-half-torch-v2"
    assert pipeline_cfg["simulation"]["emitter"]["field_extent"] == [[600, 1200], [0, 1200]]


def test_neptune_training_cfg_switches_smlm_uni_to_standardconv():
    import json
    from pathlib import Path

    pipeline_cfg = json.loads(Path("config/pipeline_smlm_uni_tanh_photmaxnorm.json").read_text(encoding="utf-8"))

    assert pipeline_cfg["smlm_overrides"]["depthwise"] is False


def test_right_fd_launcher_aligns_to_smlm_uni_density_contract():
    script = Path("scripts/slurm/slurm_smlm_fd_right_channel.sh").read_text(encoding="utf-8")

    assert "tools/run_liteloc_density_sweep.py" in script
    assert "--densities 1" in script
    assert "--batch-size 64" in script
    assert "--base-seed 42" in script
    assert "--n-max 256" in script
    assert "--train-script smlm_v2a/training/fd_right_train_pipeline.py" in script


def test_density_sweep_launcher_resets_metrics_file_before_run(tmp_path):
    from tools.run_liteloc_density_sweep import _touch_file

    metrics_path = tmp_path / "runs" / "density_1" / "metrics.jsonl"
    metrics_path.parent.mkdir(parents=True)
    metrics_path.write_text('{"epoch": 999, "jaccard": 0.123}\n', encoding="utf-8")

    _touch_file(metrics_path)

    assert metrics_path.exists()
    assert metrics_path.read_text(encoding="utf-8") == ""


def test_smlm_mu20000_depthwise_config_uses_requested_photon_distribution():
    import json

    payload = json.loads(
        Path("config/pipeline_smlm_uni_decode_aligned_fixedscale_depthwise_mu20000_sigma1000.json").read_text(
            encoding="utf-8"
        )
    )

    assert payload["simulation"]["emitter"]["intensity_mu_sig"] == [20000, 1000]


def test_smlm_mu20000_standardconv_config_switches_depthwise_off():
    import json

    payload = json.loads(
        Path("config/pipeline_smlm_uni_decode_aligned_fixedscale_standardconv_mu20000_sigma1000.json").read_text(
            encoding="utf-8"
        )
    )

    assert payload["simulation"]["emitter"]["intensity_mu_sig"] == [20000, 1000]
    assert payload["smlm_overrides"]["depthwise"] is False


def test_mu20000_dual_conv_density_launchers_match_requested_contract():
    for script_path in [
        Path("scripts/slurm/slurm_smlm_uni_density_decode_aligned_fixedscale_depthwise_mu20000_sigma1000.sh"),
        Path("scripts/slurm/slurm_smlm_uni_density_decode_aligned_fixedscale_standardconv_mu20000_sigma1000.sh"),
    ]:
        script = script_path.read_text(encoding="utf-8")

        assert "tools/run_liteloc_density_sweep.py" in script
        assert "--densities 0.1 0.2 0.5 1 2 5" in script
        assert "--epochs 500" in script
        assert "--batch-size 64" in script
        assert "--base-seed 42" in script
        assert "--n-max 256" in script
        assert "--gpus 0 1 2" in script


def test_density_sweep_launcher_cycles_pending_densities_across_limited_gpus(tmp_path, monkeypatch):
    import json

    import tools.run_liteloc_density_sweep as sweep

    cfg = {
        "optical": {
            "pixel_size_nm_x": 101.11,
            "pixel_size_nm_y": 98.83,
        },
        "simulation": {
            "frames_per_sample": 100,
            "num_samples": 1,
            "emitter": {
                "emitter_extent": [[-0.5, 63.5], [-0.5, 63.5], [-0.6, 0.6]],
                "lifetime_avg": 1.0,
            },
            "profiles": {
                "train": {"frames_per_sample": 100, "num_samples": 1},
                "eval": {"frames_per_sample": 10, "num_samples": 1},
            },
        },
    }

    launches = []

    class FakeProc:
        _pid = 3000

        def __init__(self, cmd, stdout, stderr, env):
            self.cmd = cmd
            self.stdout = stdout
            self.stderr = stderr
            self.env = env
            self.pid = FakeProc._pid
            FakeProc._pid += 1

            out_dir = Path(cmd[cmd.index("--output-dir") + 1])
            metrics_path = out_dir / "metrics.jsonl"
            metrics_path.write_text(
                json.dumps(
                    {
                        "epoch": 499,
                        "loss": 0.1,
                        "precision": 0.9,
                        "recall": 0.8,
                        "jaccard": 0.7,
                        "rmse_lat": 1.1,
                        "rmse_ax": 2.2,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            launches.append(
                {
                    "gpu": env["CUDA_VISIBLE_DEVICES"],
                    "out_dir": str(out_dir),
                    "cmd": cmd,
                }
            )

        def wait(self):
            return 0

    monkeypatch.setattr(sweep, "load_pipeline_config", lambda _path: cfg)
    monkeypatch.setattr(sweep.subprocess, "Popen", FakeProc)
    monkeypatch.setattr(
        sweep.sys,
        "argv",
        [
            "run_liteloc_density_sweep.py",
            "--config",
            "config/ignored.json",
            "--densities",
            "0.1",
            "0.2",
            "0.5",
            "1",
            "--gpus",
            "0",
            "1",
            "--output-root",
            str(tmp_path / "output"),
            "--experiment-name",
            "mu20000_depthwise",
            "--model-name",
            "smlm_test",
            "--train-script",
            "smlm_v2a/training/train_pipeline.py",
            "--epochs",
            "500",
            "--batch-size",
            "64",
            "--run",
        ],
    )

    sweep.main()

    assert len(launches) == 4
    assert {item["gpu"] for item in launches} <= {"0", "1"}
    summary_path = tmp_path / "output" / "mu20000_depthwise" / "smlm_test" / "summary.json"
    summary_csv_path = tmp_path / "output" / "mu20000_depthwise" / "smlm_test" / "summary.csv"
    assert summary_path.is_file()
    assert summary_csv_path.is_file()

    summary_rows = json.loads(summary_path.read_text(encoding="utf-8"))
    assert [row["density"] for row in summary_rows] == [0.1, 0.2, 0.5, 1.0]
    assert all(row["exit_code"] == 0 for row in summary_rows)
    assert all((tmp_path / Path(row["run_dir"])).is_dir() or Path(row["run_dir"]).is_dir() for row in summary_rows)


def test_right_fd_training_cfg_preserves_patch_extent_and_field_span():
    from config import load_pipeline_config
    from smlm_v2a.fd_conditioning.assets import load_fd_channel_asset
    from smlm_v2a.training.fd_right_train_pipeline import _field_span_in_cells, _right_cfg

    cfg = load_pipeline_config("config/pipeline_fd_right_smlm_v2a.json")
    asset = load_fd_channel_asset(RIGHT_SUMMARY)

    cfg_i = _right_cfg(cfg, field_cell=(3, 4), profile="train", seed=42)
    emitter_extent = cfg_i["simulation"]["emitter"]["emitter_extent"]
    field_extent = cfg_i["simulation"]["emitter"]["field_extent"]

    assert emitter_extent == [[-0.5, 63.5], [-0.5, 63.5], [-0.6, 0.6]]
    span_y, span_x = _field_span_in_cells(asset, patch_size=(64, 64))
    assert span_x == pytest.approx(64.0 / 40.0, rel=1e-6)
    assert span_y == pytest.approx(64.0 / 40.0, rel=1e-6)
    assert field_extent == [
        [3.0, pytest.approx(3.0 + span_x, rel=1e-6)],
        [4.0, pytest.approx(4.0 + span_y, rel=1e-6)],
    ]


def test_right_fd_train_pipeline_starts_without_pythonpath():
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    result = subprocess.run(
        ["python", "smlm_v2a/training/fd_right_train_pipeline.py", "--help"],
        cwd=Path("."),
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
