from __future__ import annotations

import pytest
import torch

from unity_psf.localization.smlm_output import SMLMOutputChannels
from unity_psf.localization.training_adapter import LocalizationTrainBatch
from unity_psf.training.localizer_eval import make_legacy_localization_eval_loss
from unity_psf.training.loop import TrainingBatch
from unity_psf.localization.legacy_decode import decode_legacy_targets
from unity_psf.localization.smlm_targets import SMLMTargetContract


class _FixedGMMModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        batch, _, height, width = images.shape
        output = torch.zeros((batch, SMLMOutputChannels.count, height, width), device=images.device)
        output[:, SMLMOutputChannels.p, 1, 1] = 1.0
        output[:, SMLMOutputChannels.photons_mu, 1, 1] = 0.5
        output[:, SMLMOutputChannels.z_mu, 1, 1] = 0.25
        output[:, SMLMOutputChannels.pxyz_sigma] = 0.1
        return output + self.anchor * 0.0


def test_gmm_eval_uses_online_physical_scales_in_nm() -> None:
    model = _FixedGMMModel()
    loc_batch = LocalizationTrainBatch(
        model_input=torch.zeros((1, 3, 3, 3)),
        detect_tar=torch.zeros((1, 3, 3)),
        bkg_tar=torch.zeros((1, 3, 3)),
        pxyz_tar=torch.tensor([[[1.5, 1.5, 0.3, 10_000.0]]]),
        mask_tar=torch.tensor([[True]]),
        metadata={},
    )

    def base_loss(current_model: torch.nn.Module, batch: TrainingBatch) -> torch.Tensor:
        del batch
        return current_model.anchor * 0.0

    eval_loss = make_legacy_localization_eval_loss(
        base_loss,
        {
            "online_generation": {
                "z_scale": 0.6,
                "photon_scale": 20_000.0,
                "pxyz_target_order": "v03",
                "pixel_size_nm_x": 100.0,
                "pixel_size_nm_y": 100.0,
            },
            "eval": {"match_dims": 3, "dist_tol_xy_nm": 250.0, "dist_tol_z_nm": 500.0},
        },
    )

    eval_loss(model, TrainingBatch(inputs=loc_batch, targets=loc_batch.detect_tar))
    metrics = eval_loss.aggregate_eval_metrics()

    assert metrics["rmse_ax"] == pytest.approx(150.0)
    assert metrics["photon_relative_error"] == pytest.approx(0.0)


def test_legacy_and_v03_targets_decode_to_same_canonical_physical_coordinates() -> None:
    contract = SMLMTargetContract(
        target_order="legacy_iwae",
        photon_scale=20_000.0,
        z_scale=0.6,
        z_scale_nm=600.0,
        z_unit="um",
        z_activation="tanh",
    )
    legacy = torch.tensor([[[0.5, 1.5, 1.5, 0.25]]])
    v03 = torch.tensor([[[1.5, 1.5, 0.15, 10_000.0]]])
    mask = torch.tensor([[True]])

    legacy_decoded = decode_legacy_targets(legacy, mask, contract=contract)
    v03_decoded = decode_legacy_targets(
        v03,
        mask,
        contract=contract.with_target_order("v03"),
    )

    torch.testing.assert_close(legacy_decoded.xyz_px_nm, v03_decoded.xyz_px_nm)
    torch.testing.assert_close(legacy_decoded.photons, v03_decoded.photons)
