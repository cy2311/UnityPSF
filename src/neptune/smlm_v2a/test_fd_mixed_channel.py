from pathlib import Path

import numpy as np
import pytest
import torch
import tifffile

from neptune.smlm_v2a.generic.localizations import FrameLocSet


def _loc_meta():
    return dict(
        ndim=3,
        xextent=(0.0, 600.0),
        yextent=(0.0, 1200.0),
        zextent=(-0.6, 0.6),
        nchannels=1,
        R2Ts=None,
        Ts2R=None,
        i_type=torch.int32,
        f_type=torch.float32,
    )


def _make_locset(xyz, *, prob=None, phot=None, indf=None):
    meta = _loc_meta()
    xyz_t = torch.tensor(xyz, dtype=torch.float32)
    n = xyz_t.shape[0]
    loc = FrameLocSet(**meta)
    loc.init_load(
        id=torch.arange(n, dtype=torch.int32),
        indf=torch.zeros(n, dtype=torch.int32) if indf is None else torch.tensor(indf, dtype=torch.int32),
        xyz=xyz_t,
        prob=torch.ones(n, dtype=torch.float32) if prob is None else torch.tensor(prob, dtype=torch.float32),
        phot=torch.full((n,), 1000.0, dtype=torch.float32) if phot is None else torch.tensor(phot, dtype=torch.float32),
    )
    return loc


def test_fd_channel_asset_loads_and_builds_40x40_conditioning():
    from neptune.smlm_v2a.fd_conditioning.assets import load_fd_channel_asset

    asset = load_fd_channel_asset(
        Path("output/neptune/physical_model_8000_split_channels/left_fdpsf/block_span_2000/summary.json")
    )

    cond = asset.conditioning_from_field_xy(torch.tensor([7.2, 12.8], dtype=torch.float32), out_size=(40, 40))

    assert asset.phase_maps.shape == (8, 30, 15)
    assert cond.shape == (8, 40, 40)
    assert torch.isfinite(cond).all()
    assert torch.allclose(cond[:, 0, 0], cond[:, -1, -1])


def test_append_conditioning_channels_concatenates_frame_and_condition_maps():
    from neptune.smlm_v2a.fd_conditioning.mixed import append_conditioning_channels

    frames = torch.zeros((3, 40, 40), dtype=torch.float32)
    cond = torch.ones((8, 40, 40), dtype=torch.float32)

    stacked = append_conditioning_channels(frames, cond)

    assert stacked.shape == (11, 40, 40)
    assert torch.allclose(stacked[:3], frames)
    assert torch.allclose(stacked[3:], cond)


def test_channel_batch_sampler_keeps_each_batch_channel_pure():
    from neptune.smlm_v2a.fd_conditioning.mixed import ChannelBatchSampler

    sampler = ChannelBatchSampler(
        channel_to_indices={"left": list(range(0, 10)), "right": list(range(10, 20))},
        batch_size=4,
        batches_per_epoch=6,
        seed=7,
    )

    batches = list(iter(sampler))

    assert len(batches) == 6
    for batch in batches:
        assert len(batch) == 4
        channels = {"left" if idx < 10 else "right" for idx in batch}
        assert len(channels) == 1


def test_cross_channel_dedup_prefers_higher_probability_for_same_frame():
    from neptune.smlm_v2a.postprocess.channel_dedup import deduplicate_channel_locsets

    left = _make_locset(
        [[100.0, 200.0, 0.05], [300.0, 500.0, -0.02]],
        prob=[0.95, 0.60],
        indf=[3, 3],
    )
    right = _make_locset(
        [[100.4, 199.8, 0.04], [420.0, 700.0, 0.01]],
        prob=[0.70, 0.90],
        indf=[3, 3],
    )

    merged = deduplicate_channel_locsets(left, right, lateral_tol_px=1.0, axial_tol_px=0.2)

    assert len(merged) == 3
    assert torch.any(torch.isclose(merged.prob, torch.tensor(0.95)))
    assert torch.any(torch.isclose(merged.prob, torch.tensor(0.60)))
    assert torch.any(torch.isclose(merged.prob, torch.tensor(0.90)))
    assert not torch.any(torch.isclose(merged.prob, torch.tensor(0.70)))


def test_fd_mixed_patch_emitter_average_scales_from_full_channel_area():
    from neptune.smlm_v2a.fd_conditioning.assets import load_fd_channel_asset
    from neptune.smlm_v2a.training.fd_mixed_train_pipeline import _resolve_patch_emitter_average

    asset = load_fd_channel_asset(
        Path("output/neptune/physical_model_8000_split_channels/left_fdpsf/block_span_2000/summary.json")
    )

    patch_em_avg = _resolve_patch_emitter_average(
        full_frame_em_avg=3.2,
        patch_size=(40, 40),
        asset=asset,
    )

    assert patch_em_avg == pytest.approx(3.2 * (40 * 40) / (600 * 1200), rel=1e-6)


def test_fd_mixed_epoch_seed_plan_advances_between_epochs():
    from neptune.smlm_v2a.training.fd_mixed_train_pipeline import _epoch_seed_plan

    epoch0 = _epoch_seed_plan(base_seed=42, epoch=0)
    epoch3 = _epoch_seed_plan(base_seed=42, epoch=3)

    assert epoch0 == {
        "train_left": 142,
        "train_right": 242,
        "eval_left": 342,
        "eval_right": 442,
    }
    assert epoch3 == {
        "train_left": 3142,
        "train_right": 3242,
        "eval_left": 3342,
        "eval_right": 3442,
    }


def test_inference_blocks_cover_all_window_indices_once():
    from tools.run_fd_mixed_sample_inference import _iter_inference_blocks

    blocks = list(_iter_inference_blocks(n_frames=10, block_frames=5, window_size=3))

    assert blocks == [(0, 5), (3, 8), (6, 10)]
    covered = []
    for start, stop in blocks:
        covered.extend(range(start, stop - 3 + 1))
    assert covered == list(range(8))


def test_load_tiff_block_reads_requested_frame_slice(tmp_path):
    from tools.run_fd_mixed_sample_inference import _load_tiff_block

    data = np.arange(5 * 3 * 4, dtype=np.uint16).reshape(5, 3, 4)
    path = tmp_path / "mini_stack.ome.tif"
    tifffile.imwrite(path, data)

    block = _load_tiff_block(path, start=1, stop=4)

    assert block.shape == (3, 3, 4)
    assert np.array_equal(block, data[1:4])


def test_merged_overlay_counts_follow_deduplicated_channel_labels():
    from tools.run_fd_mixed_sample_inference import _render_merged_channel_counts

    merged_rows = [
        {"x_px": 1.0, "y_px": 1.0, "channel": "left"},
        {"x_px": 2.0, "y_px": 3.0, "channel": "right"},
        {"x_px": 2.0, "y_px": 3.0, "channel": "right"},
    ]

    red_counts, green_counts = _render_merged_channel_counts(
        merged_rows,
        pixel_size_nm=10.0,
        camera_pixel_nm_x=10.0,
        camera_pixel_nm_y=10.0,
        width_px=10,
        height_px=10,
    )

    assert int(red_counts.sum()) == 1
    assert int(green_counts.sum()) == 2
