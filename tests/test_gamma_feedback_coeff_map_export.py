from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from neptune_v03.optics.nat_field import build_named_nat_config
from neptune_v03.training.run_high_fidelity import _export_gamma_feedback_coeff_maps


def test_gamma_feedback_export_matches_base_map_shape_for_cropped_domains(tmp_path: Path) -> None:
    nat_config = build_named_nat_config("order1_13", img_size_x=600, img_size_y=1200)
    base_maps = torch.zeros((len(nat_config.aberrations), 400, 400), dtype=torch.float32)
    objective = SimpleNamespace(
        device=torch.device("cpu"),
        nat_config=nat_config,
        base_maps_by_domain={"left": base_maps},
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
