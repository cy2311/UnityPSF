from __future__ import annotations

import ast
from pathlib import Path

import pytest

from unity_psf.localization import scale_resolver
from unity_psf.localization.runtime import config, loss_config, model_config, provider_config, simulator_range_config
from unity_psf.localization.runtime_config import (
    build_localization_runtime_config as legacy_build,
    resolve_localization_model_config as legacy_resolve_model,
)


@pytest.mark.parametrize("expert_type", ["astigmatism", "emitter_2d", "double_helix"])
def test_runtime_build_resolves_scales_once(monkeypatch, expert_type: str) -> None:
    calls = []
    target_calls = []
    resolve_scales = scale_resolver.resolve_physical_scales
    resolve_target = scale_resolver.resolve_target_contract

    def counted_resolver(train_cfg):
        scales = resolve_scales(train_cfg)
        calls.append(scales)
        return scales

    def counted_target_resolver(train_cfg, **kwargs):
        target = resolve_target(train_cfg, **kwargs)
        target_calls.append(target)
        return target

    for module in (config, loss_config, provider_config, simulator_range_config, scale_resolver):
        monkeypatch.setattr(module, "resolve_physical_scales", counted_resolver)
    monkeypatch.setattr(config, "resolve_target_contract", counted_target_resolver)

    runtime = config.build_localization_runtime_config(
        {
            "train": {
                "expert": {"expert_type": expert_type, "channel_id": "main"},
                "channel_layout": {"channels": ["main"]},
                "online_generation": {"channels": 3, "pxyz_target_order": "v03"},
                "scaling": {"photon_max": 2000.0, "z_max": 500.0},
            }
        }
    )
    assert len(calls) == 1
    assert len(target_calls) == 1
    params = runtime["batch_provider"]["params"]
    assert params["photon_scale"] == calls[0].photon_scale
    assert params["z_scale"] == calls[0].z_scale
    assert runtime["resolved_contract"]["physical_scales"] == calls[0].as_dict()
    assert runtime["resolved_contract"]["target_contract"]["target_order"] == "legacy_iwae"


def test_runtime_config_legacy_exports_use_canonical_resolvers() -> None:
    assert legacy_build is config.build_localization_runtime_config
    assert legacy_resolve_model is model_config.resolve_localization_model_config
    assert config._loss_config is loss_config.resolve_loss_config
    assert config._microtube_tiff_provider_config is provider_config.microtube_tiff_provider_config


def test_runtime_configuration_size_limits() -> None:
    runtime_dir = Path(config.__file__).parent
    assert len((runtime_dir / "config.py").read_text().splitlines()) <= 180
    for name in (
        "config", "model_config", "loss_config", "modality_config",
        "conditioning_config", "provider_config", "contract_config", "simulator_range_config",
    ):
        source = (runtime_dir / f"{name}.py").read_text()
        assert len(source.splitlines()) <= 300, name
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                assert node.end_lineno - node.lineno + 1 <= 300, node.name
            if isinstance(node, ast.ImportFrom):
                assert all(not imported.name.startswith("_") for imported in node.names), name
