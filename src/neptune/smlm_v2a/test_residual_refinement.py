import torch

from neptune.smlm_v2a.evaluation.matcher import GreedyHungarianMatching
from neptune.smlm_v2a.generic.localizations import FrameLocSet
from neptune.smlm_v2a.residual_refinement.loop import ResidualRefinementRuntime
from neptune.smlm_v2a.residual_refinement.state import (
    build_next_round_targets,
    filter_by_probability,
    matched_gt_and_remaining,
    merge_round_predictions,
    subtract_center_render,
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


def _make_locset(xyz, prob=None, phot=None):
    meta = _loc_meta()
    xyz_t = torch.tensor(xyz, dtype=torch.float32)
    n = xyz_t.shape[0]
    loc = FrameLocSet(**meta)
    loc.init_load(
        id=torch.arange(n, dtype=torch.int32),
        indf=torch.zeros(n, dtype=torch.int32),
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


def test_matched_gt_and_remaining_uses_tp_match_with_probability_filter():
    gt = _make_locset(
        [
            [10.0, 10.0, 0.10],
            [20.0, 20.0, 0.20],
        ]
    )
    pred = _make_locset(
        [
            [10.1, 10.0, 0.10],
            [20.1, 20.0, 0.20],
        ],
        prob=[0.95, 0.60],
    )

    accepted = filter_by_probability(pred, threshold=0.8)
    matched_gt, remaining_gt = matched_gt_and_remaining(accepted, gt, _matcher())

    assert len(accepted) == 1
    assert len(matched_gt) == 1
    assert torch.allclose(matched_gt.xyz[0], gt.xyz[0])
    assert len(remaining_gt) == 1
    assert torch.allclose(remaining_gt.xyz[0], gt.xyz[1])


def test_merge_round_predictions_discards_later_duplicates():
    round1 = _make_locset(
        [
            [12.0, 12.0, 0.05],
        ],
        prob=[0.95],
    )
    round2 = _make_locset(
        [
            [12.1, 12.0, 0.05],
            [40.0, 40.0, 0.10],
        ],
        prob=[0.91, 0.92],
    )

    merged = merge_round_predictions([round1, round2], _matcher())

    assert len(merged) == 2
    assert torch.allclose(merged.xyz[0], round1.xyz[0])
    assert torch.allclose(merged.xyz[1], round2.xyz[1])


def test_merge_round_predictions_keeps_empty_meta_when_all_rounds_empty():
    empty1 = FrameLocSet(**_loc_meta())
    empty2 = FrameLocSet(**_loc_meta())

    merged = merge_round_predictions([empty1, empty2], _matcher())

    assert len(merged) == 0
    assert merged.meta_isclose(empty1)


def test_subtract_center_render_only_changes_middle_channel():
    window = torch.ones((2, 3, 4, 4), dtype=torch.float32)
    render = torch.full((2, 4, 4), 0.25, dtype=torch.float32)

    updated = subtract_center_render(window, render)

    assert torch.allclose(updated[:, 0], window[:, 0])
    assert torch.allclose(updated[:, 2], window[:, 2])
    assert torch.allclose(updated[:, 1], window[:, 1] - render)


def test_targets_to_locset_restores_photon_and_z_scaling():
    pxyz_tar = torch.tensor(
        [
            [
                [0.5, 10.0, 12.0, 0.25],
                [0.0, 0.0, 0.0, 0.0],
            ]
        ],
        dtype=torch.float32,
    )
    mask = torch.tensor([[True, False]], dtype=torch.bool)

    locs = targets_to_locset(
        pxyz_tar,
        mask,
        loc_meta=FrameLocSet(**_loc_meta()),
        phot_max=2000.0,
        z_max=0.8,
    )

    assert len(locs) == 1
    assert torch.allclose(locs.phot, torch.tensor([1000.0]))
    assert torch.allclose(locs.xyz[0], torch.tensor([10.0, 12.0, 0.2]))


def test_build_next_round_targets_masks_matched_gt_slots():
    pxyz_tar = torch.tensor(
        [
            [
                [0.5, 10.0, 10.0, 0.125],
                [0.4, 20.0, 20.0, 0.250],
                [0.0, 0.0, 0.0, 0.0],
            ]
        ],
        dtype=torch.float32,
    )
    mask = torch.tensor([[True, True, False]], dtype=torch.bool)
    accepted = _make_locset([[10.1, 10.0, 0.10]], prob=[0.95], phot=[1000.0])

    detect_next, pxyz_next, mask_next, matched_gt = build_next_round_targets(
        accepted,
        pxyz_tar,
        mask,
        loc_meta=FrameLocSet(**_loc_meta()),
        matcher=_matcher(),
        phot_max=2000.0,
        z_max=0.8,
        height=32,
        width=32,
    )

    assert len(matched_gt) == 1
    assert torch.allclose(matched_gt.xyz[0], torch.tensor([10.0, 10.0, 0.1]))
    assert torch.equal(mask_next, torch.tensor([[False, True, False]]))
    assert torch.allclose(pxyz_next[0, 1], pxyz_tar[0, 1])
    assert torch.allclose(pxyz_next[0, 0], torch.zeros(4))
    assert detect_next.sum().item() == 1.0
    assert detect_next[0, 20, 20].item() == 1.0


class _StubModel:
    def __init__(self):
        self.calls = 0

    def __call__(self, x):
        self.calls += 1
        return torch.zeros((x.shape[0], 10, x.shape[-2], x.shape[-1]), dtype=x.dtype, device=x.device)


class _StubCriterion:
    def __init__(self):
        self.calls = 0

    def forward(self, output, detect_tar, pxyz_tar, mask_tar, bkg_tar):
        self.calls += 1
        return torch.full((output.shape[0], 2), float(self.calls), dtype=output.dtype, device=output.device)


class _StubPostProcessor:
    def __init__(self, rounds):
        self.rounds = list(rounds)
        self.calls = 0

    def forward(self, y_out):
        locs = self.rounds[self.calls]
        self.calls += 1
        return locs


class _StubRenderer:
    def __init__(self):
        self.render_counts = []

    def render(self, locs, *, batch_size, height, width, device):
        self.render_counts.append(len(locs))
        return torch.zeros((batch_size, height, width), dtype=torch.float32, device=device)


def test_residual_runtime_sums_round_losses_and_realigns_predictions():
    accepted_round1 = _make_locset([[10.0, 10.0, 0.1]], prob=[0.95], phot=[1000.0])
    empty = FrameLocSet(**_loc_meta())
    runtime = ResidualRefinementRuntime(
        rounds=3,
        acceptance_threshold=0.8,
        training_render_source="matched_gt",
        inference_render_source="prediction",
        post_processor=_StubPostProcessor([accepted_round1, empty, empty]),
        matcher=_matcher(),
        renderer=_StubRenderer(),
        loc_meta=FrameLocSet(**_loc_meta()),
        phot_max=2000.0,
        z_max=0.8,
    )
    model = _StubModel()
    criterion = _StubCriterion()
    x = torch.ones((1, 3, 32, 32), dtype=torch.float32)
    detect_tar = torch.zeros((1, 32, 32), dtype=torch.float32)
    detect_tar[0, 10, 10] = 1.0
    pxyz_tar = torch.tensor([[[0.5, 10.0, 10.0, 0.125]]], dtype=torch.float32)
    mask_tar = torch.tensor([[True]], dtype=torch.bool)
    bkg_tar = torch.zeros((1, 32, 32), dtype=torch.float32)

    loss_total, round_predictions = runtime.forward_rounds(
        model,
        criterion,
        x,
        detect_tar,
        pxyz_tar,
        mask_tar,
        bkg_tar,
        render_source="matched_gt",
        frame_offset=5,
        collect_predictions=True,
    )

    assert model.calls == 3
    assert criterion.calls == 3
    assert torch.allclose(loss_total, torch.full((1, 2), 6.0))
    assert len(round_predictions) == 3
    assert len(round_predictions[0]) == 1
    assert round_predictions[0].indf[0].item() == 5
    assert runtime.renderer.render_counts == [1, 0]
