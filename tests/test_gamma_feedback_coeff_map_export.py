from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from neptune_v03.optics.nat_field import build_named_nat_config, full_roi_coeff_stack_torch
from neptune_v03.training.run_high_fidelity import _export_gamma_feedback_coeff_maps


def test_gamma_feedback_export_matches_base_map_shape_for_cropped_domains(tmp_path: Path) -> None:
    nat_config = build_named_nat_config("order1_13", img_size_x=600, img_size_y=1200)
    base_maps = torch.zeros((len(nat_config.aberrations), 400, 400), dtype=torch.float32)
    objective = SimpleNamespace(
        device=torch.device("cpu"),
        nat_config=nat_config,
        base_maps_by_domain={"left": base_maps},
        config=SimpleNamespace(preserve_full_base_modes=False),
    )
    layout = SimpleNamespace(artifacts_dir=tmp_path)
    gamma = torch.full((len(nat_config.gammas),), 0.25, dtype=torch.float32)

    entries = _export_gamma_feedback_coeff_maps(
        objective,
        gamma=gamma,
        layout=layout,
        epoch=30,
        global_step=12510,
        artifact_policy="compact_latest",
    )

    assert entries and entries[0][0] == "left"
    payload = np.load(entries[0][1])
    assert payload["zernike_maps_nm"].shape == (len(nat_config.aberrations), 400, 400)


def test_gamma_feedback_export_persists_cumulative_gamma_state(tmp_path: Path) -> None:
    nat_config = build_named_nat_config("order1_13", img_size_x=6, img_size_y=5)
    base_maps = torch.full((len(nat_config.aberrations), 5, 6), 3.0, dtype=torch.float32)
    objective = SimpleNamespace(
        device=torch.device("cpu"),
        nat_config=nat_config,
        base_maps_by_domain={"left": base_maps},
        config=SimpleNamespace(preserve_full_base_modes=False),
    )
    layout = SimpleNamespace(artifacts_dir=tmp_path)
    gamma_before = torch.linspace(0.1, 1.3, len(nat_config.gammas), dtype=torch.float32)
    gamma_after = gamma_before + torch.linspace(0.01, 0.13, len(nat_config.gammas), dtype=torch.float32)

    entries = _export_gamma_feedback_coeff_maps(
        objective,
        gamma=gamma_after,
        gamma_before=gamma_before,
        layout=layout,
        epoch=300,
        global_step=125100,
        artifact_policy="compact_latest",
    )

    with np.load(entries[0][1], allow_pickle=False) as payload:
        expected = base_maps + full_roi_coeff_stack_torch(gamma_after, nat_config).maps_nm
        np.testing.assert_allclose(payload["zernike_maps_nm"], expected.numpy(), rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(payload["gamma_values_nm"], gamma_after.numpy(), rtol=0.0, atol=0.0)
        assert payload["physical_state_semantics"].item() == "base_plus_cumulative_gamma"
