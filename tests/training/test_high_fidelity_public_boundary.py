from __future__ import annotations

import unity_psf.training.run_high_fidelity as entrypoint


def test_high_fidelity_entrypoint_exposes_only_public_lifecycle_api() -> None:
    assert entrypoint.__all__ == ["main", "parse_args", "resume_epoch_training_config"]
    leaked = {
        name
        for name in vars(entrypoint)
        if name.startswith("_") and not name.startswith("__")
    }
    assert leaked == set()
