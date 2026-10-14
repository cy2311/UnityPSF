from __future__ import annotations

import os
import importlib.util
from pathlib import Path

import numpy as np
import torch
import yaml

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).parents[2] / ".local" / "cache" / "matplotlib"))

_UNION_MODULE_PATH = Path(__file__).parents[2] / "scripts" / "infer" / "render_union_raw_ratio_bicolor.py"
_UNION_MODULE_SPEC = importlib.util.spec_from_file_location("unity_baseline_union", _UNION_MODULE_PATH)
if _UNION_MODULE_SPEC is None or _UNION_MODULE_SPEC.loader is None:
    raise ImportError(f"cannot load baseline union module from {_UNION_MODULE_PATH}")
_UNION_MODULE = importlib.util.module_from_spec(_UNION_MODULE_SPEC)
_UNION_MODULE_SPEC.loader.exec_module(_UNION_MODULE)
union_frame = _UNION_MODULE.union_frame
from unity_psf.data.normalization import CameraCalibration, TrainNormalization, adu_to_photons, normalize_train_input
from unity_psf.infer_recon.filter.filter import FilterConfig, compute_locprec_xy_nm, filter_rows
from unity_psf.localization.smlm_targets import SMLMTargetConvention, absolute_pxyz_to_local_targets


FIXTURE_PATH = Path(__file__).parents[1] / "fixtures" / "astigmatism_baseline.yaml"


def _fixture() -> dict[str, object]:
    payload = yaml.safe_load(FIXTURE_PATH.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def test_camera_photon_and_training_normalization_are_stable() -> None:
    fixture = _fixture()
    camera_spec = fixture["camera"]
    normalization_spec = fixture["normalization"]
    assert isinstance(camera_spec, dict)
    assert isinstance(normalization_spec, dict)

    frames_adu = np.asarray([100.0, 101.0, 110.0], dtype=np.float32)
    photons = adu_to_photons(
        frames_adu,
        CameraCalibration(
            baseline_adu=float(camera_spec["baseline_adu"]),
            e_per_adu=float(camera_spec["e_per_adu"]),
            qe=float(camera_spec["qe"]),
            em_gain=float(camera_spec["em_gain"]),
            spurious_charge=float(camera_spec["spurious_charge"]),
        ),
    )
    assert np.allclose(photons, np.asarray([0.0, 1.5, 19.5], dtype=np.float32))

    normalized = normalize_train_input(
        photons,
        TrainNormalization(
            input_offset=float(normalization_spec["input_offset"]),
            input_scale=float(normalization_spec["input_scale"]),
            photon_scale=float(normalization_spec["photon_scale"]),
        ),
    )
    assert np.allclose(normalized, (photons - 1.0) / 20.0)


def test_global_targets_keep_xy_order_and_apply_z_photon_scales() -> None:
    fixture = _fixture()
    target_spec = fixture["target"]
    assert isinstance(target_spec, dict)

    global_target = torch.tensor([[12.0, 20.0, 120.0, 400.0]], dtype=torch.float32)
    local_target = absolute_pxyz_to_local_targets(
        global_target,
        x=torch.tensor([10.0]),
        y=torch.tensor([18.0]),
        convention=SMLMTargetConvention(
            photon_scale=float(target_spec["photon_scale"]),
            z_scale=float(target_spec["z_scale"]),
            z_activation=str(target_spec["z_activation"]),
        ),
    )
    assert torch.allclose(local_target, torch.tensor([[2.0, 2.0, 0.2, 2.0]]))


def test_filter_stages_preserve_quality_and_z_boundaries() -> None:
    fixture = _fixture()
    filter_spec = fixture["filter"]
    assert isinstance(filter_spec, dict)
    rows = [
        {"frame": 0, "x_px": 10.0, "y_px": 20.0, "z": 50.0, "prob": 0.95, "photon": 100.0, "x_sig": 1.0, "y_sig": 1.0},
        {"frame": 1, "x_px": 11.0, "y_px": 20.0, "z": 50.0, "prob": 0.20, "photon": 100.0, "x_sig": 1.0, "y_sig": 1.0},
        {"frame": 2, "x_px": 12.0, "y_px": 20.0, "z": 100.0, "prob": 0.95, "photon": 100.0, "x_sig": 1.0, "y_sig": 1.0},
    ]
    filtered, summary = filter_rows(
        rows,
        FilterConfig(
            prob_min=float(filter_spec["probability_min"]),
            emitter_z_min_nm=float(filter_spec["z_min_nm"]),
            emitter_z_max_nm=float(filter_spec["z_max_nm"]),
            locprec_xy_nm_max=float(filter_spec["locprec_xy_nm_max"]),
        ),
        camera_pixel_nm_x=float(filter_spec["camera_pixel_nm_x"]),
        camera_pixel_nm_y=float(filter_spec["camera_pixel_nm_y"]),
    )
    assert len(filtered) == 1
    assert filtered[0]["frame"] == 0
    assert np.isclose(float(filtered[0]["locprec_xy_nm"]), compute_locprec_xy_nm(
        x_sig_px=1.0,
        y_sig_px=1.0,
        camera_pixel_nm_x=100.0,
        camera_pixel_nm_y=200.0,
    ))
    assert summary["total_in"] == 3
    assert summary["after_prob"] == 2
    assert summary["after_emitter_z"] == 1
    assert summary["after_locprec_xy_nm"] == 1
    assert summary["total_out"] == 1


def test_union_aligns_left_coordinates_and_prefers_right_duplicate() -> None:
    fixture = _fixture()
    union_spec = fixture["union"]
    assert isinstance(union_spec, dict)
    left = {
        "x_px": np.asarray([5.0, 20.0], dtype=np.float32),
        "y_px": np.asarray([7.0, 20.0], dtype=np.float32),
        "z": np.asarray([10.0, 20.0], dtype=np.float32),
        "photon": np.asarray([100.0, 200.0], dtype=np.float32),
        "prob": np.asarray([0.8, 0.7], dtype=np.float32),
        "x_sig": np.asarray([1.0, 1.0], dtype=np.float32),
        "y_sig": np.asarray([1.0, 1.0], dtype=np.float32),
    }
    right = {
        "x_px": np.asarray([105.4, 40.0], dtype=np.float32),
        "y_px": np.asarray([7.3, 40.0], dtype=np.float32),
        "z": np.asarray([11.0, 30.0], dtype=np.float32),
        "photon": np.asarray([110.0, 300.0], dtype=np.float32),
        "prob": np.asarray([0.8, 0.6], dtype=np.float32),
        "x_sig": np.asarray([1.1, 1.2], dtype=np.float32),
        "y_sig": np.asarray([1.1, 1.2], dtype=np.float32),
    }
    union = union_frame(
        left,
        right,
        np.asarray([0, 1], dtype=np.int64),
        np.asarray([0, 1], dtype=np.int64),
        max_dist_px=float(union_spec["max_dist_px"]),
        left_to_right_dx_px=float(union_spec["left_to_right_dx_px"]),
        left_to_right_dy_px=float(union_spec["left_to_right_dy_px"]),
    )

    assert union["x_px"].shape == (3,)
    duplicate = np.flatnonzero(np.isclose(union["x_px"], 105.4))[0]
    assert union["source"][duplicate] == 1
    assert union["source_index"][duplicate] == 0
    assert union["matched_left_index"][duplicate] == -1
    assert np.any(np.isclose(union["x_px"], 120.0))
    assert np.any(np.isclose(union["x_px"], 40.0))
