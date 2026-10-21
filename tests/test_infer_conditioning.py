from pathlib import Path

import pytest

from scripts.infer.run_3371_full8000_infer_filter_recon import (
    condition_feature_dim,
    physical_pairing_metadata,
    rcc_fallback_allowed,
    resolve_coeff_map,
    runtime_pixel_size_nm,
)


def test_condition_feature_dim_validates_domain_layout() -> None:
    assert condition_feature_dim(condition_dim=10, domain_count=2, domain_index=1) == 8

    for kwargs in (
        {"condition_dim": 10, "domain_count": 0, "domain_index": 0},
        {"condition_dim": 10, "domain_count": 2, "domain_index": 2},
        {"condition_dim": 2, "domain_count": 2, "domain_index": 0},
    ):
        try:
            condition_feature_dim(**kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid condition layout was accepted: {kwargs}")


def test_runtime_pixel_size_comes_from_training_contract() -> None:
    runtime = {
        "batch_provider": {
            "params": {
                "pixel_size_nm_x": 207.0,
                "pixel_size_nm_y": 207.0,
            }
        }
    }

    assert runtime_pixel_size_nm(runtime) == (207.0, 207.0)


def test_single_domain_double_helix_map_resolves_to_left() -> None:
    coeff_map = Path("double_helix.npz")

    assert resolve_coeff_map({"double_helix": coeff_map}, side="left", domain_count=1) == coeff_map

    with pytest.raises(KeyError):
        resolve_coeff_map({"double_helix": coeff_map}, side="right", domain_count=1)


def test_physical_pairing_rejects_checkpoint_from_an_earlier_endpoint(tmp_path) -> None:
    coeff_map = tmp_path / "coeff_maps.npz"
    coeff_map.write_bytes(b"final physical map")
    current_state = {
        "condition_store_version": 28,
        "epoch": 300,
        "global_step": 125100,
        "coeff_maps": [{"name": "double_helix", "coeff_maps_npz": str(coeff_map)}],
    }
    checkpoint = {
        "epoch": 249,
        "global_step": 103833,
        "physical_state": {
            "condition_store_version": 22,
            "epoch": 240,
            "global_step": 100080,
            "coeff_maps": [{"name": "double_helix", "coeff_maps_npz": str(coeff_map)}],
        },
    }

    with pytest.raises(ValueError, match="does not match the current physical endpoint"):
        physical_pairing_metadata(checkpoint, current_state)


def test_physical_pairing_accepts_final_network_and_final_endpoint(tmp_path) -> None:
    coeff_map = tmp_path / "coeff_maps.npz"
    coeff_map.write_bytes(b"final physical map")
    current_state = {
        "condition_store_version": 28,
        "epoch": 300,
        "global_step": 125100,
        "coeff_maps": [{"name": "double_helix", "coeff_maps_npz": str(coeff_map)}],
    }
    checkpoint = {
        "epoch": 300,
        "global_step": 125100,
        "physical_state": {
            "condition_store_version": 27,
            "epoch": 290,
            "global_step": 120930,
            "coeff_maps": [{"name": "double_helix", "coeff_maps_npz": str(coeff_map)}],
        },
    }

    pairing = physical_pairing_metadata(checkpoint, current_state)

    assert pairing["contract"] == "final_network_plus_final_physical_state_endpoint"
    assert pairing["checkpoint_epoch"] == 300
    assert pairing["checkpoint_embedded_physical_epoch"] == 290
    assert pairing["evaluated_physical_epoch"] == 300
    assert pairing["coeff_maps"]["double_helix"]["sha256"] == (
        "c8dbb7506bfa8efd89aa3da1dd7f2bc9fe8946fc88de895e8daea432d622b1ff"
    )


def test_rcc_fallback_is_only_allowed_for_empty_pairwise_output(tmp_path) -> None:
    pairwise = tmp_path / "rcc_pairwise_shifts.csv"
    corrected = tmp_path / "predictions_degrid_rcc_corrected.h5"

    pairwise.write_text("")
    assert rcc_fallback_allowed(returncode=1, pairwise_path=pairwise, corrected_path=corrected)

    pairwise.write_text("frame_start,frame_end\n0,1\n")
    assert not rcc_fallback_allowed(returncode=1, pairwise_path=pairwise, corrected_path=corrected)
    assert not rcc_fallback_allowed(returncode=0, pairwise_path=pairwise, corrected_path=corrected)
