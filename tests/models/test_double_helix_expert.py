from __future__ import annotations

from pathlib import Path

import torch

from unity_psf.contracts import load_checkpoint, save_checkpoint
from unity_psf.models.psf_moe.experts.double_helix import DoubleHelixDirectXYZLoss, DoubleHelixExpert


def _expert() -> DoubleHelixExpert:
    torch.manual_seed(31)
    return DoubleHelixExpert(feature_channels=8)


def test_double_helix_expert_has_direct_xyz_and_lobe_auxiliary_outputs() -> None:
    expert = _expert()
    output = expert(torch.randn(2, 8, 8, 8))

    assert output.detection_logits.shape == (2, 8, 8)
    assert output.xy_offset.shape == (2, 2, 8, 8)
    assert output.z.shape == (2, 8, 8)
    assert output.photons.shape == (2, 8, 8)
    assert set(output.auxiliary) == {"lobe_angle", "lobe_separation"}
    assert expert.checkpoint_metadata().expert_type.value == "double_helix"


def test_double_helix_expert_optimizer_and_checkpoint_reload(tmp_path: Path) -> None:
    expert = _expert()
    optimizer = torch.optim.AdamW(expert.parameters(), lr=1e-3)
    features = torch.randn(2, 8, 8, 8)
    before = {name: value.detach().clone() for name, value in expert.state_dict().items()}

    optimizer.zero_grad(set_to_none=True)
    output = expert(features)
    loss = output.detection_logits.square().mean() + output.z.square().mean() + output.photons.mean()
    loss.backward()
    optimizer.step()
    assert any(not torch.equal(value, before[name]) for name, value in expert.state_dict().items())

    checkpoint_path = tmp_path / "dh_instance.pt"
    save_checkpoint(checkpoint_path, expert.state_dict(), metadata=expert.checkpoint_metadata())
    restored = _expert()
    restored.load_state_dict(load_checkpoint(checkpoint_path)["model_state_dict"])
    expert.eval()
    restored.eval()
    with torch.no_grad():
        expected = expert(features)
        actual = restored(features)
    assert torch.equal(expected.detection_logits, actual.detection_logits)
    assert torch.equal(expected.xy_offset, actual.xy_offset)
    assert torch.equal(expected.z, actual.z)
    assert torch.equal(expected.photons, actual.photons)
    for name in expected.auxiliary:
        assert torch.equal(expected.auxiliary[name], actual.auxiliary[name])


def test_double_helix_direct_xyz_loss_consumes_dense_direct_and_auxiliary_targets() -> None:
    expert = _expert()
    output = expert(torch.randn(2, 8, 8, 8))
    targets = {
        "detection": torch.zeros(2, 8, 8),
        "xy_offset": torch.zeros(2, 2, 8, 8),
        "z": torch.zeros(2, 8, 8),
        "photons": torch.ones(2, 8, 8),
        "lobe_angle": torch.zeros(2, 1, 8, 8),
        "lobe_separation": torch.ones(2, 1, 8, 8),
    }
    loss = DoubleHelixDirectXYZLoss()
    value = loss(output, targets)
    assert value.ndim == 0
    assert torch.isfinite(value)
