from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from unity_psf.cli.multichannel import main


def _write_config(path: Path) -> None:
    path.write_text(
        yaml.safe_dump(
            {
                "schema_version": "0.4",
                "train": {
                    "channel_layout": {
                        "frame_size": [8, 16],
                        "channels": [
                            {"id": "left", "crop": [0, 0, 8, 8], "anchor_profile": "anchor"},
                            {"id": "right", "crop": [8, 0, 8, 8], "anchor_profile": "anchor"},
                        ],
                    },
                    "expert": {"name": "astigmatism", "instance_id": "main", "channel_id": "main"},
                    "model": {"name": "astigmatism_expert"},
                    "online_generation": {"channels": 3},
                    "real_tiff_wake": {"tiff_path": "data/raw.tif"},
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def test_plan_cli_materializes_independent_channel_configs(tmp_path: Path, monkeypatch, capsys) -> None:
    config_path = tmp_path / "base.yaml"
    _write_config(config_path)
    run_root = tmp_path / "runs"
    monkeypatch.setattr(
        "sys.argv",
        [
            "unity-psf-train-multichannel",
            "--config",
            str(config_path),
            "--run-root",
            str(run_root),
            "--expert-type",
            "astigmatism",
            "--run-name",
            "astig-experiment",
            "--mode",
            "plan",
            "--channel-seed",
            "left=3",
            "--channel-seed",
            "right=4",
        ],
    )

    assert main() == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["expert_type"] == "astigmatism"
    assert [item["channel_id"] for item in payload["channels"]] == ["left", "right"]
    assert [item["seed"] for item in payload["channels"]] == [3, 4]

    left_config = run_root / "astig-experiment" / "channels" / "left" / "config.yaml"
    right_config = run_root / "astig-experiment" / "channels" / "right" / "config.yaml"
    assert left_config.is_file()
    assert right_config.is_file()
    left = yaml.safe_load(left_config.read_text(encoding="utf-8"))
    right = yaml.safe_load(right_config.read_text(encoding="utf-8"))
    assert [item["id"] for item in left["train"]["channel_layout"]["channels"]] == ["left"]
    assert [item["id"] for item in right["train"]["channel_layout"]["channels"]] == ["right"]
    assert left["train"]["expert"]["instance_id"] == "left"
    assert right["train"]["expert"]["channel_id"] == "right"
    assert left["train"]["real_tiff_wake"]["tiff_path"] == str(tmp_path / "data" / "raw.tif")


def test_slurm_cli_writes_one_script_per_channel(tmp_path: Path, monkeypatch, capsys) -> None:
    config_path = tmp_path / "base.yaml"
    _write_config(config_path)
    script_root = tmp_path / "slurm"
    monkeypatch.setattr(
        "sys.argv",
        [
            "unity-psf-train-multichannel",
            "--config",
            str(config_path),
            "--run-root",
            str(tmp_path / "runs"),
            "--expert-type",
            "double_helix",
            "--mode",
            "slurm",
            "--slurm-script-root",
            str(script_root),
        ],
    )

    assert main() == 0
    payload = json.loads(capsys.readouterr().out)
    assert [item["channel_id"] for item in payload["jobs"]] == ["left", "right"]
    assert (script_root / "left.sh").is_file()
    assert (script_root / "right.sh").is_file()
    assert "--run-name left" in (script_root / "left.sh").read_text(encoding="utf-8")
    assert "--run-name right" in (script_root / "right.sh").read_text(encoding="utf-8")


def test_channel_configs_filter_all_physical_inputs(tmp_path: Path, monkeypatch, capsys) -> None:
    config_path = tmp_path / "base.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": "0.4",
                "train": {
                    "channel_layout": {
                        "frame_size": [8, 16],
                        "channels": [
                            {"id": "left", "crop": [0, 0, 8, 8]},
                            {"id": "right", "crop": [8, 0, 8, 8]},
                        ]
                    },
                    "expert": {"name": "astigmatism"},
                    "online_generation": {
                        "dual_domain_coeff_maps": [
                            {"name": "left", "coeff_maps_npz": "maps/left.npz"},
                            {"name": "right", "coeff_maps_npz": "maps/right.npz"},
                        ],
                        "lut_simulation": {
                            "dual_domain_zmaps": [
                                {"name": "left", "zmap_npz": "maps/left-z.npz"},
                                {"name": "right", "zmap_npz": "maps/right-z.npz"},
                            ]
                        },
                    },
                    "real_tiff_wake": {
                        "tiff_path": "data/raw.tif",
                        "domains": [
                            {"name": "left", "crop_left": 0, "crop_top": 0, "crop_width": 8, "crop_height": 8},
                            {"name": "right", "crop_left": 8, "crop_top": 0, "crop_width": 8, "crop_height": 8},
                        ],
                    },
                    "roi_bank_gamma": {
                        "base_coeff_maps": [
                            {"name": "left", "coeff_maps_npz": "maps/left.npz"},
                            {"name": "right", "coeff_maps_npz": "maps/right.npz"},
                        ],
                        "roi_bank_source": {
                            "mode": "auto_build",
                            "raw_path": "data/raw.tif",
                            "domains": [
                                {"name": "left", "crop_left": 0, "crop_top": 0, "crop_width": 8, "crop_height": 8},
                                {"name": "right", "crop_left": 8, "crop_top": 0, "crop_width": 8, "crop_height": 8},
                            ],
                        },
                        "auto_build_domains": [
                            {"name": "left", "crop_left": 0, "crop_top": 0, "crop_width": 8, "crop_height": 8},
                            {"name": "right", "crop_left": 8, "crop_top": 0, "crop_width": 8, "crop_height": 8},
                        ],
                    },
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "unity-psf-train-multichannel",
            "--config",
            str(config_path),
            "--run-root",
            str(tmp_path / "runs"),
            "--expert-type",
            "astigmatism",
            "--mode",
            "plan",
        ],
    )

    assert main() == 0
    capsys.readouterr()
    for channel_id, other_id in (("left", "right"), ("right", "left")):
        channel_path = tmp_path / "runs" / "astigmatism" / "channels" / channel_id / "config.yaml"
        channel = yaml.safe_load(channel_path.read_text(encoding="utf-8"))
        train = channel["train"]
        assert [item["name"] for item in train["real_tiff_wake"]["domains"]] == [channel_id]
        selected_domain = train["real_tiff_wake"]["domains"][0]
        assert selected_domain["crop_left"] == (0 if channel_id == "left" else 8)
        assert [item["name"] for item in train["online_generation"]["dual_domain_coeff_maps"]] == [channel_id]
        assert train["online_generation"]["dual_domain_coeff_maps"][0]["coeff_maps_npz"] == str(
            tmp_path / "maps" / f"{channel_id}.npz"
        )
        assert [item["name"] for item in train["online_generation"]["lut_simulation"]["dual_domain_zmaps"]] == [channel_id]
        assert train["online_generation"]["lut_simulation"]["dual_domain_zmaps"][0]["zmap_npz"] == str(
            tmp_path / "maps" / f"{channel_id}-z.npz"
        )
        assert [item["name"] for item in train["roi_bank_gamma"]["base_coeff_maps"]] == [channel_id]
        assert [item["name"] for item in train["roi_bank_gamma"]["roi_bank_source"]["domains"]] == [channel_id]
        assert [item["name"] for item in train["roi_bank_gamma"]["auto_build_domains"]] == [channel_id]
        assert all(
            item["name"] != other_id
            for entries in (
                train["real_tiff_wake"]["domains"],
                train["online_generation"]["dual_domain_coeff_maps"],
                train["online_generation"]["lut_simulation"]["dual_domain_zmaps"],
                train["roi_bank_gamma"]["base_coeff_maps"],
                train["roi_bank_gamma"]["roi_bank_source"]["domains"],
                train["roi_bank_gamma"]["auto_build_domains"],
            )
            for item in entries
        )


def test_channel_config_rejects_ambiguous_physical_domains(tmp_path: Path) -> None:
    config = {
        "train": {
            "channel_layout": {
                "frame_size": [8, 16],
                "channels": [{"id": "left", "crop": [0, 0, 8, 8]}],
            },
            "expert": {"name": "astigmatism"},
            "real_tiff_wake": {
                "domains": [
                    {"name": "calibration_a", "crop_left": 0, "crop_top": 0, "crop_width": 8, "crop_height": 8},
                    {"name": "calibration_b", "crop_left": 8, "crop_top": 0, "crop_width": 8, "crop_height": 8},
                ]
            },
        }
    }
    source = tmp_path / "base.yaml"
    source.write_text(yaml.safe_dump(config), encoding="utf-8")
    from unity_psf.cli.multichannel import _channel_config
    from unity_psf.training.multichannel import ChannelRunSpec
    from unity_psf.contracts.modality import PSFModality

    spec = ChannelRunSpec(
        expert_type=PSFModality.ASTIGMATISM,
        instance_id="left",
        channel_id="left",
        seed=0,
        run_root=tmp_path / "runs",
        run_name="left",
        crop=(0, 0, 8, 8),
    )
    with pytest.raises(ValueError, match="no unique entry for channel='left'"):
        _channel_config(config, PSFModality.ASTIGMATISM, spec, base_dir=source.parent)
