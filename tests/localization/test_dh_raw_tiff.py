from __future__ import annotations

import json

import numpy as np
import pytest
import tifffile
import torch

from unity_psf.localization.dh_raw_tiff import DHRawTiffBatchProviderConfig, build_dh_raw_tiff_batch_provider


def test_physical_update_vector_lut_samples_keep_triplets_and_targets_aligned(tmp_path) -> None:
    raw_path = tmp_path / "raw.tif"
    tifffile.imwrite(raw_path, np.zeros((3, 4, 4), dtype=np.float32))
    coefficient_map = tmp_path / "coeff_maps.npz"
    np.savez(coefficient_map, maps=np.zeros((1,), dtype=np.float32))
    state_path = tmp_path / "physical_state.json"
    state_path.write_text(json.dumps({"source": "gamma_feedback", "coeff_maps": [{"coeff_maps_npz": str(coefficient_map)}]}))
    frames = np.stack((np.full((3, 4, 4), 10.0, dtype=np.float32), np.full((3, 4, 4), 20.0, dtype=np.float32)))
    target_path = tmp_path / "batch.npz"
    np.savez(
        target_path,
        frames_adu=frames,
        emitter_xy_z_um_photons=np.array([[[1.0, 1.0, 0.0, 100.0]], [[2.0, 2.0, 0.0, 200.0]]], dtype=np.float32),
        emitter_mask=np.ones((2, 1), dtype=bool),
        dh_lobe_targets=np.zeros((2, 2, 3), dtype=np.float32),
        dh_lobe_mask=np.zeros((2, 2), dtype=bool),
    )
    metadata_path = tmp_path / "batch.metadata.json"
    metadata_path.write_text(json.dumps({"source": "dh_vector_lut_simulation_from_physical_update", "simulation_backend": "lut", "psf_type": "vector", "coefficient_map": str(coefficient_map)}))

    provider = build_dh_raw_tiff_batch_provider(
        DHRawTiffBatchProviderConfig(
            raw_tiff_path=raw_path,
            target_npz_path=target_path,
            frames_npz_path=target_path,
            physical_state_path=state_path,
            simulation_metadata_path=metadata_path,
            batch_size=2,
            steps_per_epoch=1,
            seed=0,
        )
    )
    batch = next(iter(provider(1)))

    assert batch.inputs.shape == (2, 3, 4, 4)
    assert batch.metadata["source"] == "dh_vector_lut_simulation_from_physical_update"
    assert set(batch.inputs[:, 0, 0, 0].tolist()).issubset({10.0, 20.0})


def test_materialized_formal_batch_rejects_reusing_samples_within_an_epoch(tmp_path) -> None:
    raw_path = tmp_path / "raw.tif"
    tifffile.imwrite(raw_path, np.zeros((3, 4, 4), dtype=np.float32))
    coefficient_map = tmp_path / "coeff_maps.npz"
    np.savez(coefficient_map, maps=np.zeros((1,), dtype=np.float32))
    state_path = tmp_path / "physical_state.json"
    state_path.write_text(json.dumps({"source": "gamma_feedback", "coeff_maps": [{"coeff_maps_npz": str(coefficient_map)}]}))
    target_path = tmp_path / "batch.npz"
    np.savez(
        target_path,
        frames_adu=np.zeros((2, 3, 4, 4), dtype=np.float32),
        emitter_xy_z_um_photons=np.zeros((2, 1, 4), dtype=np.float32),
        emitter_mask=np.zeros((2, 1), dtype=bool),
        dh_lobe_targets=np.zeros((2, 2, 3), dtype=np.float32),
        dh_lobe_mask=np.zeros((2, 2), dtype=bool),
    )
    metadata_path = tmp_path / "batch.metadata.json"
    metadata_path.write_text(json.dumps({"source": "dh_vector_lut_simulation_from_physical_update", "simulation_backend": "lut", "psf_type": "vector", "coefficient_map": str(coefficient_map)}))

    with pytest.raises(ValueError, match="materialized DH batch"):
        build_dh_raw_tiff_batch_provider(
            DHRawTiffBatchProviderConfig(
                raw_tiff_path=raw_path,
                target_npz_path=target_path,
                frames_npz_path=target_path,
                physical_state_path=state_path,
                simulation_metadata_path=metadata_path,
                batch_size=2,
                steps_per_epoch=2,
            )
        )


def test_materialized_epoch_visits_each_required_sample_once(tmp_path) -> None:
    raw_path = tmp_path / "raw.tif"
    tifffile.imwrite(raw_path, np.zeros((3, 4, 4), dtype=np.float32))
    coefficient_map = tmp_path / "coeff_maps.npz"
    np.savez(coefficient_map, maps=np.zeros((1,), dtype=np.float32))
    state_path = tmp_path / "physical_state.json"
    state_path.write_text(json.dumps({"source": "gamma_feedback", "coeff_maps": [{"coeff_maps_npz": str(coefficient_map)}]}))
    target_path = tmp_path / "batch.npz"
    frames = np.arange(4, dtype=np.float32)[:, None, None, None] * np.ones((1, 3, 4, 4), dtype=np.float32)
    np.savez(
        target_path,
        frames_adu=frames,
        emitter_xy_z_um_photons=np.zeros((4, 1, 4), dtype=np.float32),
        emitter_mask=np.zeros((4, 1), dtype=bool),
        dh_lobe_targets=np.zeros((4, 2, 3), dtype=np.float32),
        dh_lobe_mask=np.zeros((4, 2), dtype=bool),
    )
    metadata_path = tmp_path / "batch.metadata.json"
    metadata_path.write_text(json.dumps({"source": "dh_vector_lut_simulation_from_physical_update", "simulation_backend": "lut", "psf_type": "vector", "coefficient_map": str(coefficient_map)}))

    provider = build_dh_raw_tiff_batch_provider(
        DHRawTiffBatchProviderConfig(
            raw_tiff_path=raw_path,
            target_npz_path=target_path,
            frames_npz_path=target_path,
            physical_state_path=state_path,
            simulation_metadata_path=metadata_path,
            batch_size=2,
            steps_per_epoch=2,
            seed=7,
        )
    )

    seen = torch.cat([batch.inputs[:, 0, 0, 0] for batch in provider(1)]).tolist()
    assert sorted(seen) == [0.0, 1.0, 2.0, 3.0]
