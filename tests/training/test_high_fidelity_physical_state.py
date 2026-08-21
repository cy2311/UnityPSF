from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from unity_psf.training.high_fidelity.physical_state import (
    _physical_checkpoint_extra_fn,
    _read_current_physical_state,
    _runtime_dual_domain_coeff_maps,
    _write_current_physical_state,
)


def test_legacy_physical_state_write_preserves_payload_and_manifest_hash(tmp_path: Path) -> None:
    layout = SimpleNamespace(metadata_dir=tmp_path / "metadata")
    layout.metadata_dir.mkdir(parents=True)
    manifest_path = layout.metadata_dir / "run_manifest.json"
    manifest_path.write_text('{"run_id": "example"}\n', encoding="utf-8")

    state_path = _write_current_physical_state(
        layout,
        coeff_maps=(
            {"name": "left", "coeff_maps_npz": "/tmp/left.npz"},
            {"name": "right", "coeff_maps_npz": "/tmp/right.npz"},
        ),
        source="initial",
        epoch=None,
        global_step=None,
        condition_store_version=None,
    )

    expected = {
        "source": "initial",
        "epoch": None,
        "global_step": None,
        "condition_store_version": None,
        "coeff_maps": [
            {"name": "left", "coeff_maps_npz": "/tmp/left.npz"},
            {"name": "right", "coeff_maps_npz": "/tmp/right.npz"},
        ],
    }
    expected_hash = hashlib.sha256(
        json.dumps(expected, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    assert Path(state_path) == layout.metadata_dir / "current_physical_state.json"
    assert _read_current_physical_state(layout) == expected
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["run_id"] == "example"
    assert manifest["current_physical_state"] == expected
    assert manifest["initial_physical_state_hash"] == expected_hash
    assert manifest["latest_physical_state_hash"] == expected_hash

    _write_current_physical_state(
        layout,
        coeff_maps=expected["coeff_maps"],
        source="gamma_feedback",
        epoch=2,
        global_step=8,
        condition_store_version=3,
    )
    updated_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert updated_manifest["initial_physical_state_hash"] == expected_hash
    assert updated_manifest["latest_physical_state_hash"] != expected_hash


def test_physical_state_helpers_preserve_resolution_and_missing_state_behavior(tmp_path: Path) -> None:
    layout = SimpleNamespace(metadata_dir=tmp_path / "metadata")
    runtime_config = {
        "batch_provider": {
            "params": {
                "dual_domain_coeff_maps": [
                    {"name": "left", "alternating_coeff_maps_npz": "left.npz"},
                    {"path": "right.npz"},
                    {"name": "ignored"},
                    "invalid",
                ]
            }
        }
    }

    assert _runtime_dual_domain_coeff_maps(runtime_config) == (
        {"name": "left", "coeff_maps_npz": "left.npz"},
        {"name": "domain1", "coeff_maps_npz": "right.npz"},
    )
    assert _read_current_physical_state(layout) is None
    assert _physical_checkpoint_extra_fn(layout)() == {}
