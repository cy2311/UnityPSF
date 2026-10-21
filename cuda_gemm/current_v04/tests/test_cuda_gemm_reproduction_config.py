import importlib.util
from pathlib import Path

import torch
import yaml

from neptune_v03.localization.online import OnlineBatchProviderConfig, _apply_cuda_gemm_camera_readout


_PREPARE_CONFIG_PATH = Path(__file__).parents[1] / "scripts/train/prepare_microtube_dual2_5819_config.py"
_PREPARE_CONFIG_SPEC = importlib.util.spec_from_file_location("prepare_microtube_dual2_5819_config", _PREPARE_CONFIG_PATH)
assert _PREPARE_CONFIG_SPEC is not None and _PREPARE_CONFIG_SPEC.loader is not None
_PREPARE_CONFIG_MODULE = importlib.util.module_from_spec(_PREPARE_CONFIG_SPEC)
_PREPARE_CONFIG_SPEC.loader.exec_module(_PREPARE_CONFIG_MODULE)
prepare_config = _PREPARE_CONFIG_MODULE.prepare_config


def test_reproduction_config_switches_lut_to_cuda_gemm_and_preserves_psf_size(tmp_path: Path) -> None:
    base_config = tmp_path / "base.yaml"
    raw_tiff = tmp_path / "raw.tif"
    zmap_root = tmp_path / "zmaps"
    output = tmp_path / "resolved.yaml"
    raw_tiff.touch()
    for side in ("left", "right"):
        export_dir = zmap_root / side / "export_nat_zmap"
        export_dir.mkdir(parents=True)
        (export_dir / "preferred_full_roi_zernike_maps_nm.npz").touch()

    base_config.write_text(
        yaml.safe_dump(
            {
                "phase_retrieval": {"input": {"tiff_path": "old.tif"}},
                "train": {
                    "online_generation": {
                        "simulation_backend": "lut",
                        "vector_batch_size": 1024,
                        "dual_domain_coeff_maps": [
                            {"name": "left", "coeff_maps_npz": "old-left.npz"},
                            {"name": "right", "coeff_maps_npz": "old-right.npz"},
                        ],
                        "lut_simulation": {"psf_size": 25, "z_steps": 61},
                    }
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    prepare_config(base_config=base_config, raw_tiff=raw_tiff, zmap_root=zmap_root, output=output)

    resolved = yaml.safe_load(output.read_text(encoding="utf-8"))
    online = resolved["train"]["online_generation"]
    assert online["simulation_backend"] == "cuda_gemm"
    assert online["vector_psf_size"] == 25
    assert online["vector_batch_size"] == 1024
    assert "lut_simulation" not in online
    assert [Path(item["coeff_maps_npz"]).name for item in online["dual_domain_coeff_maps"]] == [
        "preferred_full_roi_zernike_maps_nm.npz",
        "preferred_full_roi_zernike_maps_nm.npz",
    ]
    launcher = Path(__file__).parents[1] / "scripts/train/reproduce_microtube_dual2_v04_from0.sbatch"
    launcher_text = launcher.read_text(encoding="utf-8")
    assert "results/microtube_dual2_v04_from0_cuda_gemm" in launcher_text
    assert "results/microtube_dual2_v04_from0/logs" not in launcher_text


def test_cuda_gemm_camera_readout_preserves_lut_adu_baseline() -> None:
    config = OnlineBatchProviderConfig(
        batch_size=1,
        camera_qe=0.9,
        camera_spurious_charge=0.0,
        camera_baseline=398.6,
        camera_e_per_adu=1.020784562122306,
    )

    raw_adu = _apply_cuda_gemm_camera_readout(config, torch.zeros((2, 4, 4)), seed=11)

    assert torch.equal(raw_adu, torch.full((2, 4, 4), 398.6))
