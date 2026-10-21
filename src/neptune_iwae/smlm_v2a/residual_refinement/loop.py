from __future__ import annotations

import time
from dataclasses import dataclass

import torch
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..data.feature_factory import SpatialIntegration
from ..evaluation.matcher import GreedyHungarianMatching
from ..generic.localizations import FrameLocSet
from ..models.smlm_model import DoubleUNet, SingleUNet
from ..training.losses import GMMLoss
from ..training.progress import MetricMeter
from ..utils import hardware
from .render import MeanPSFCenterRenderer
from .state import (
    build_next_round_targets,
    filter_by_probability,
    merge_round_predictions,
    subtract_center_render,
)


@dataclass(frozen=True)
class ResidualRefinementRuntime:
    rounds: int
    acceptance_threshold: float
    training_render_source: str
    inference_render_source: str
    post_processor: SpatialIntegration
    matcher: GreedyHungarianMatching
    renderer: MeanPSFCenterRenderer
    loc_meta: FrameLocSet
    phot_max: float
    z_max: float

    def _step_state(
        self,
        accepted: FrameLocSet,
        x: torch.Tensor,
        pxyz_tar: torch.Tensor,
        mask_tar: torch.BoolTensor,
        *,
        render_source: str,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.BoolTensor]:
        detect_next, pxyz_next, mask_next, matched_gt = build_next_round_targets(
            accepted,
            pxyz_tar,
            mask_tar,
            loc_meta=self.loc_meta,
            matcher=self.matcher,
            phot_max=self.phot_max,
            z_max=self.z_max,
            height=x.shape[-2],
            width=x.shape[-1],
        )
        if render_source == "matched_gt":
            render_locs = matched_gt
        elif render_source == "prediction":
            render_locs = accepted
        else:
            raise ValueError(f"Unsupported residual render source: {render_source!r}")
        center_render = self.renderer.render(
            render_locs,
            batch_size=int(x.shape[0]),
            height=int(x.shape[-2]),
            width=int(x.shape[-1]),
            device=x.device,
        )
        x_next = subtract_center_render(x, center_render)
        return x_next, detect_next, pxyz_next, mask_next

    def forward_rounds(
        self,
        model: SingleUNet | DoubleUNet,
        criterion: GMMLoss,
        x: torch.Tensor,
        detect_tar: torch.Tensor,
        pxyz_tar: torch.Tensor,
        mask_tar: torch.BoolTensor,
        bkg_tar: torch.Tensor,
        *,
        render_source: str,
        frame_offset: int = 0,
        collect_predictions: bool = False,
    ) -> tuple[torch.Tensor, list[FrameLocSet]]:
        state_x = x
        state_detect = detect_tar
        state_pxyz = pxyz_tar
        state_mask = mask_tar

        round_losses: list[torch.Tensor] = []
        round_predictions: list[FrameLocSet] = []

        for round_ix in range(int(self.rounds)):
            y_out = model(state_x)
            round_losses.append(criterion.forward(y_out, state_detect, state_pxyz, state_mask, bkg_tar))

            loc_out = self.post_processor.forward(y_out.detach())
            accepted = filter_by_probability(loc_out, threshold=self.acceptance_threshold)
            if collect_predictions:
                round_predictions.append(accepted.realign_frame(frame_offset))

            if round_ix + 1 >= int(self.rounds):
                continue

            state_x, state_detect, state_pxyz, state_mask = self._step_state(
                accepted,
                state_x,
                state_pxyz,
                state_mask,
                render_source=render_source,
            )

        loss_total = torch.stack(round_losses, dim=0).sum(dim=0)
        return loss_total, round_predictions


def train_forward_residual(
    model: SingleUNet | DoubleUNet,
    optimizer: Optimizer,
    loss: GMMLoss,
    dataloader: DataLoader,
    grad_mod: bool,
    epoch: int,
    device: torch.device | str,
    runtime: ResidualRefinementRuntime,
) -> torch.Tensor:
    loss_cmp_epoch = []
    loss_vals_log = MetricMeter()
    model.train()

    tqdm_enum = tqdm(dataloader, total=len(dataloader), smoothing=0.0)
    t0 = time.time()

    for x, detect_tar, bkg_tar, pxyz_tar, mask_tar, _frame_loc in tqdm_enum:
        t_data = time.time() - t0
        x, detect_tar, bkg_tar, pxyz_tar, mask_tar = hardware.to_device(
            [x, detect_tar, bkg_tar, pxyz_tar, mask_tar],
            device,
        )

        optimizer.zero_grad()
        state_x = x
        state_detect = detect_tar
        state_pxyz = pxyz_tar
        state_mask = mask_tar
        loss_total = None

        for round_ix in range(int(runtime.rounds)):
            y_out = model(state_x)
            round_loss = loss.forward(y_out, state_detect, state_pxyz, state_mask, bkg_tar)
            round_loss.sum(-1).mean().backward()
            loss_total = round_loss.detach() if loss_total is None else loss_total + round_loss.detach()

            if round_ix + 1 >= int(runtime.rounds):
                continue

            loc_out = runtime.post_processor.forward(y_out.detach())
            accepted = filter_by_probability(loc_out, threshold=runtime.acceptance_threshold)
            state_x, state_detect, state_pxyz, state_mask = runtime._step_state(
                accepted,
                state_x,
                state_pxyz,
                state_mask,
                render_source=runtime.training_render_source,
            )

        if grad_mod:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.03, norm_type=2)
        optimizer.step()

        t_batch = time.time() - t0
        loss_mean = loss_total.sum(-1).mean()
        loss_vals_log.update(loss_mean)
        tqdm_enum.set_description(
            f"(TrainRR) E: {epoch} - t: {t_batch:.2f} - t_dat: {t_data:.2f} - L: {loss_vals_log.mean:.3f}"
        )

        loss_cmp_epoch.append(loss_total.mean(dim=0).cpu())
        t0 = time.time()

    return torch.stack(loss_cmp_epoch, dim=0)


def test_forward_residual(
    model: SingleUNet | DoubleUNet,
    loss: GMMLoss,
    dataloader: DataLoader,
    epoch: int,
    device: torch.device | str,
    runtime: ResidualRefinementRuntime,
) -> tuple[torch.Tensor, FrameLocSet]:
    loss_cmp_epoch = []
    loss_vals_log = MetricMeter()
    loc_parts: list[FrameLocSet] = []

    model.eval()
    tqdm_enum = tqdm(dataloader, total=len(dataloader), smoothing=0.0)
    t0 = time.time()
    frame_offset = 0

    with torch.no_grad():
        for x, detect_tar, bkg_tar, pxyz_tar, mask_tar, _frame_loc in tqdm_enum:
            x, detect_tar, bkg_tar, pxyz_tar, mask_tar = hardware.to_device(
                [x, detect_tar, bkg_tar, pxyz_tar, mask_tar],
                device,
            )
            loss_val, round_predictions = runtime.forward_rounds(
                model,
                loss,
                x,
                detect_tar,
                pxyz_tar,
                mask_tar,
                bkg_tar,
                render_source=runtime.inference_render_source,
                frame_offset=frame_offset,
                collect_predictions=True,
            )

            merged = merge_round_predictions(round_predictions, runtime.matcher)
            loc_parts.append(merged)

            t_batch = time.time() - t0
            loss_vals_log.update(loss_val.sum(-1).mean())
            tqdm_enum.set_description(f"( TestRR) E: {epoch} - t: {t_batch:.2f} - L: {loss_vals_log.mean:.3f}")

            loss_cmp_epoch.append(loss_val.mean(dim=0).detach().cpu())
            frame_offset += int(x.shape[0])

    loss_cmp_epoch_t = torch.stack(loss_cmp_epoch, dim=0)
    loc_out = FrameLocSet(**runtime.loc_meta.meta) if not loc_parts else FrameLocSet.loc_cat(loc_parts)
    return loss_cmp_epoch_t, loc_out
