import torch
from pathlib import Path
import pytest

from neptune.smlm_v2a.evaluation.matcher import GreedyHungarianMatching
from neptune.smlm_v2a.generic.localizations import FrameLocSet
from neptune.smlm_v2a.local_deblending.dataset import crop_patch, encode_patch_targets
from neptune.smlm_v2a.local_deblending.geometry import ConflictPatch, build_conflict_patches
from neptune.smlm_v2a.local_deblending.losses import LocalSetGMMLoss
from neptune.smlm_v2a.local_deblending.merge import (
    merge_patch_predictions,
    merge_patch_predictions_with_strategy,
)
from neptune.smlm_v2a.local_deblending.model import LocalSetPredictor
from neptune.smlm_v2a.local_deblending.pipeline import (
    build_patch_examples,
    decode_patch_predictions,
    targets_to_locset,
)


def _loc_meta():
    return dict(
        ndim=3,
        xextent=(-0.5, 63.5),
        yextent=(-0.5, 63.5),
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


def _matcher():
    return GreedyHungarianMatching(
        match_dims=3,
        cam_pxszs=(100.0, 100.0, 100.0),
        dist_tolr=250.0,
        dist_tolz=500.0,
        dist_tolv=None,
    )


def test_build_conflict_patches_groups_close_proposals_and_selects_single_anchor():
    proposals = _make_locset(
        [
            [10.0, 10.0, 0.10],
            [12.0, 10.5, 0.08],
            [40.0, 40.0, 0.20],
        ],
        prob=[0.95, 0.70, 0.80],
    )

    patches = build_conflict_patches(
        proposals,
        conflict_radius_px=3.0,
        patch_size=25,
    )

    assert len(patches) == 1
    patch = patches[0]
    assert patch.anchor_index == 0
    assert patch.proposal_indices == (0, 1)
    assert patch.patch_size == 25
    assert patch.frame_index == 0
    assert patch.x0 == 0
    assert patch.y0 == 0
    assert patch.x1 == 25
    assert patch.y1 == 25


def test_crop_patch_extracts_centered_spatial_window():
    frames = torch.arange(3 * 32 * 32, dtype=torch.float32).view(3, 32, 32)

    patch = crop_patch(frames, x0=4, y0=5, x1=9, y1=10)

    assert patch.shape == (3, 5, 5)
    assert torch.equal(patch, frames[:, 5:10, 4:9])


def test_encode_patch_targets_filters_to_patch_and_localizes_coordinates():
    gt = _make_locset(
        [
            [10.0, 10.0, 0.10],
            [20.0, 20.0, 0.20],
            [50.0, 50.0, -0.10],
        ],
        phot=[1000.0, 1500.0, 1200.0],
    )
    patch = ConflictPatch(
        frame_index=0,
        anchor_index=0,
        proposal_indices=(0, 1),
        x0=8,
        y0=8,
        x1=24,
        y1=24,
        patch_size=16,
    )

    pxyz_tar, mask = encode_patch_targets(
        gt,
        patch,
        max_targets=4,
        phot_max=2000.0,
        z_max=0.6,
    )

    assert torch.equal(mask, torch.tensor([True, True, False, False]))
    assert torch.allclose(pxyz_tar[0], torch.tensor([0.5, 2.0, 2.0, 0.10 / 0.6], dtype=torch.float32))
    assert torch.allclose(pxyz_tar[1], torch.tensor([0.75, 12.0, 12.0, 0.20 / 0.6], dtype=torch.float32))
    assert torch.allclose(pxyz_tar[2:], torch.zeros((2, 4), dtype=torch.float32))


def test_local_set_predictor_outputs_slot_tensors_with_valid_ranges():
    model = LocalSetPredictor(in_channels=4, hidden_channels=16, num_slots=4, patch_size=25)
    x = torch.randn(2, 4, 25, 25)

    out = model(x)

    assert out["prob"].shape == (2, 4)
    assert out["mu"].shape == (2, 4, 4)
    assert out["sig"].shape == (2, 4, 4)
    assert torch.all((out["prob"] >= 0.0) & (out["prob"] <= 1.0))
    assert torch.all(out["sig"] > 0.0)
    assert torch.all((out["mu"][..., 1:3] >= 0.0) & (out["mu"][..., 1:3] <= 25.0))
    assert torch.all((out["mu"][..., 3] >= -1.0) & (out["mu"][..., 3] <= 1.0))


def test_local_set_gmm_loss_is_finite_and_prefers_better_alignment():
    loss_fn = LocalSetGMMLoss()
    target = torch.tensor([[[0.5, 10.0, 11.0, 0.2]]], dtype=torch.float32)
    mask = torch.tensor([[True]], dtype=torch.bool)

    aligned = {
        "prob": torch.tensor([[0.9, 0.1]], dtype=torch.float32),
        "mu": torch.tensor([[[0.5, 10.0, 11.0, 0.2], [0.2, 3.0, 3.0, -0.1]]], dtype=torch.float32),
        "sig": torch.full((1, 2, 4), 0.1, dtype=torch.float32),
    }
    offset = {
        "prob": torch.tensor([[0.9, 0.1]], dtype=torch.float32),
        "mu": torch.tensor([[[0.5, 2.0, 2.0, -0.2], [0.2, 3.0, 3.0, -0.1]]], dtype=torch.float32),
        "sig": torch.full((1, 2, 4), 0.1, dtype=torch.float32),
    }

    loss_aligned = loss_fn(aligned, target, mask)
    loss_offset = loss_fn(offset, target, mask)

    assert torch.isfinite(loss_aligned).all()
    assert torch.isfinite(loss_offset).all()
    assert loss_aligned.mean().item() < loss_offset.mean().item()


def test_merge_patch_predictions_replaces_global_proposals_inside_patch():
    global_locs = _make_locset(
        [
            [10.0, 10.0, 0.10],
            [12.0, 10.5, 0.08],
            [40.0, 40.0, 0.20],
        ],
        prob=[0.95, 0.70, 0.80],
    )
    patch = ConflictPatch(
        frame_index=0,
        anchor_index=0,
        proposal_indices=(0, 1),
        x0=0,
        y0=0,
        x1=25,
        y1=25,
        patch_size=25,
    )
    local_pred = _make_locset(
        [
            [10.2, 10.1, 0.11],
            [13.0, 10.8, 0.09],
        ],
        prob=[0.92, 0.60],
    )

    merged = merge_patch_predictions(global_locs, [(patch, local_pred)], matcher=_matcher())

    assert len(merged) == 3
    assert torch.allclose(merged.xyz[-1], torch.tensor([40.0, 40.0, 0.20]))
    assert torch.any(torch.isclose(merged.xyz[:, 0], torch.tensor(10.2)))
    assert torch.any(torch.isclose(merged.xyz[:, 0], torch.tensor(13.0)))


def test_merge_patch_predictions_avoids_matcher_forward_slow_path(monkeypatch):
    empty = FrameLocSet(**_loc_meta())
    patch_a = ConflictPatch(
        frame_index=0,
        anchor_index=0,
        proposal_indices=(0, 1),
        x0=0,
        y0=0,
        x1=25,
        y1=25,
        patch_size=25,
    )
    patch_b = ConflictPatch(
        frame_index=0,
        anchor_index=2,
        proposal_indices=(2, 3),
        x0=8,
        y0=8,
        x1=33,
        y1=33,
        patch_size=25,
    )
    local_a = _make_locset(
        [
            [10.0, 10.0, 0.10],
            [30.0, 30.0, 0.15],
        ],
        prob=[0.95, 0.80],
    )
    local_b = _make_locset(
        [
            [10.1, 10.0, 0.11],
            [45.0, 45.0, 0.20],
        ],
        prob=[0.90, 0.70],
    )
    matcher = _matcher()

    def _slow_path(*_args, **_kwargs):
        raise AssertionError("matcher.forward slow path should not be used for local patch dedupe")

    monkeypatch.setattr(matcher, "forward", _slow_path)

    merged = merge_patch_predictions(
        empty,
        [(patch_a, local_a), (patch_b, local_b)],
        matcher=matcher,
    )

    assert len(merged) == 3
    assert torch.allclose(merged.xyz[0], torch.tensor([10.0, 10.0, 0.10]))
    assert torch.allclose(merged.xyz[1], torch.tensor([30.0, 30.0, 0.15]))
    assert torch.allclose(merged.xyz[2], torch.tensor([45.0, 45.0, 0.20]))


def test_merge_strategy_keep_global_adds_only_unmatched_local_predictions():
    global_locs = _make_locset(
        [
            [10.0, 10.0, 0.10],
            [12.0, 10.5, 0.08],
            [40.0, 40.0, 0.20],
        ],
        prob=[0.95, 0.70, 0.80],
    )
    patch = ConflictPatch(
        frame_index=0,
        anchor_index=0,
        proposal_indices=(0, 1),
        x0=0,
        y0=0,
        x1=25,
        y1=25,
        patch_size=25,
    )
    local_pred = _make_locset(
        [
            [10.1, 10.0, 0.10],
            [15.0, 12.0, 0.09],
        ],
        prob=[0.90, 0.85],
    )

    merged = merge_patch_predictions_with_strategy(
        global_locs,
        [(patch, local_pred)],
        matcher=_matcher(),
        strategy="keep_global_add_local_unmatched",
    )

    assert len(merged) == 3
    assert torch.allclose(merged.xyz[0], torch.tensor([10.0, 10.0, 0.10]))
    assert torch.allclose(merged.xyz[1], torch.tensor([40.0, 40.0, 0.20]))
    assert torch.allclose(merged.xyz[2], torch.tensor([15.0, 12.0, 0.09]))


def test_merge_strategy_keep_high_conf_global_replaces_only_low_conflict_predictions():
    global_locs = _make_locset(
        [
            [10.0, 10.0, 0.10],
            [12.0, 10.5, 0.08],
            [40.0, 40.0, 0.20],
        ],
        prob=[0.95, 0.45, 0.80],
    )
    patch = ConflictPatch(
        frame_index=0,
        anchor_index=0,
        proposal_indices=(0, 1),
        x0=0,
        y0=0,
        x1=25,
        y1=25,
        patch_size=25,
    )
    local_pred = _make_locset(
        [
            [10.1, 10.0, 0.10],
            [13.5, 10.8, 0.09],
        ],
        prob=[0.92, 0.88],
    )

    merged = merge_patch_predictions_with_strategy(
        global_locs,
        [(patch, local_pred)],
        matcher=_matcher(),
        strategy="keep_high_conf_global_add_local_unmatched",
        global_prob_threshold=0.8,
    )

    assert len(merged) == 3
    assert torch.allclose(merged.xyz[0], torch.tensor([10.0, 10.0, 0.10]))
    assert torch.allclose(merged.xyz[1], torch.tensor([40.0, 40.0, 0.20]))
    assert torch.allclose(merged.xyz[2], torch.tensor([13.5, 10.8, 0.09]))


def test_merge_strategy_keep_anchor_preserves_anchor_and_fills_new_local_predictions():
    global_locs = _make_locset(
        [
            [10.0, 10.0, 0.10],
            [12.0, 10.5, 0.08],
            [40.0, 40.0, 0.20],
        ],
        prob=[0.95, 0.70, 0.80],
    )
    patch = ConflictPatch(
        frame_index=0,
        anchor_index=0,
        proposal_indices=(0, 1),
        x0=0,
        y0=0,
        x1=25,
        y1=25,
        patch_size=25,
    )
    local_pred = _make_locset(
        [
            [10.1, 10.0, 0.10],
            [13.0, 10.8, 0.09],
        ],
        prob=[0.92, 0.88],
    )

    merged = merge_patch_predictions_with_strategy(
        global_locs,
        [(patch, local_pred)],
        matcher=_matcher(),
        strategy="keep_anchor_global_local_fill",
    )

    assert len(merged) == 3
    assert torch.allclose(merged.xyz[0], torch.tensor([10.0, 10.0, 0.10]))
    assert torch.allclose(merged.xyz[1], torch.tensor([40.0, 40.0, 0.20]))
    assert torch.allclose(merged.xyz[2], torch.tensor([13.0, 10.8, 0.09]))


def test_targets_to_locset_reconstructs_absolute_frame_targets():
    pxyz_tar = torch.tensor(
        [
            [
                [0.50, 10.0, 11.0, 0.20],
                [0.00, 0.0, 0.0, 0.00],
                [0.00, 0.0, 0.0, 0.00],
            ],
            [
                [0.75, 20.0, 21.0, -0.50],
                [0.25, 22.0, 23.0, 0.50],
                [0.00, 0.0, 0.0, 0.00],
            ],
        ],
        dtype=torch.float32,
    )
    mask = torch.tensor(
        [
            [True, False, False],
            [True, True, False],
        ],
        dtype=torch.bool,
    )

    locs = targets_to_locset(
        pxyz_tar,
        mask,
        loc_meta=FrameLocSet(**_loc_meta()),
        phot_max=2000.0,
        z_max=0.6,
        frame_offset=5,
    )

    assert len(locs) == 3
    assert torch.equal(locs.indf, torch.tensor([5, 6, 6], dtype=torch.int32))
    assert torch.allclose(locs.phot, torch.tensor([1000.0, 1500.0, 500.0]))
    assert torch.allclose(
        locs.xyz,
        torch.tensor(
            [
                [10.0, 11.0, 0.12],
                [20.0, 21.0, -0.30],
                [22.0, 23.0, 0.30],
            ],
            dtype=torch.float32,
        ),
        atol=1e-6,
    )


def test_decode_patch_predictions_projects_slots_back_to_global_frame():
    patch = ConflictPatch(
        frame_index=7,
        anchor_index=0,
        proposal_indices=(0, 1),
        x0=8,
        y0=10,
        x1=24,
        y1=26,
        patch_size=16,
    )
    output = {
        "prob": torch.tensor([[0.80, 0.20]], dtype=torch.float32),
        "mu": torch.tensor(
            [
                [
                    [0.50, 2.0, 3.0, 0.50],
                    [0.25, 10.0, 11.0, -0.50],
                ]
            ],
            dtype=torch.float32,
        ),
        "sig": torch.full((1, 2, 4), 0.1, dtype=torch.float32),
    }

    decoded = decode_patch_predictions(
        output,
        patches=[patch],
        loc_meta=FrameLocSet(**_loc_meta()),
        phot_max=2000.0,
        z_max=0.6,
        prob_threshold=0.5,
    )

    assert len(decoded) == 1
    pred = decoded[0]
    assert len(pred) == 1
    assert torch.equal(pred.indf, torch.tensor([7], dtype=torch.int32))
    assert torch.allclose(pred.prob, torch.tensor([0.80]))
    assert torch.allclose(pred.phot, torch.tensor([1000.0]))
    assert torch.allclose(pred.xyz, torch.tensor([[10.0, 13.0, 0.30]], dtype=torch.float32), atol=1e-6)


def test_build_patch_examples_concatenates_raw_patch_and_anchor_channel():
    frames = torch.arange(2 * 3 * 16 * 16, dtype=torch.float32).view(2, 3, 16, 16)
    anchor_render = torch.zeros((2, 16, 16), dtype=torch.float32)
    anchor_render[0, 4:8, 5:9] = 1.0

    gt = _make_locset(
        [
            [6.0, 5.0, 0.10],
            [8.0, 7.0, 0.20],
        ],
        phot=[1000.0, 1500.0],
        indf=[0, 0],
    )
    patch = ConflictPatch(
        frame_index=0,
        anchor_index=0,
        proposal_indices=(0, 1),
        x0=4,
        y0=3,
        x1=12,
        y1=11,
        patch_size=8,
    )

    examples = build_patch_examples(
        frames,
        anchor_render,
        gt_locs=gt,
        patches=[patch],
        max_targets=4,
        phot_max=2000.0,
        z_max=0.6,
    )

    assert len(examples) == 1
    example = examples[0]
    assert example.patch == patch
    assert example.x.shape == (4, 8, 8)
    assert torch.equal(example.x[:3], frames[0, :, 3:11, 4:12])
    assert torch.equal(example.x[3], anchor_render[0, 3:11, 4:12])
    assert torch.equal(example.mask, torch.tensor([True, True, False, False]))


def test_runner_resolves_local_deblending_config(tmp_path: Path):
    from neptune.smlm_v2a.training.train_local_deblender import resolve_local_deblending_config

    ckpt = tmp_path / "global_model_2.pt"
    ckpt.write_bytes(b"stub")
    cfg = {
        "local_deblending": {
            "global_checkpoint": str(ckpt),
            "patch_size": 25,
            "conflict_radius_px": 3.0,
            "num_slots": 4,
            "max_targets": 4,
        }
    }

    parsed = resolve_local_deblending_config(cfg, repo_root=tmp_path)

    assert parsed.global_checkpoint == ckpt
    assert parsed.patch_size == 25
    assert parsed.conflict_radius_px == 3.0
    assert parsed.num_slots == 4
    assert parsed.max_targets == 4
