"""Static contracts for the deprecated DH namespace and archived workflows."""

from __future__ import annotations

import importlib
import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
ARCHIVE_ROOT = PROJECT_ROOT / "scripts" / "archive"
FORMAL_ROOT = SRC_ROOT / "unity_psf"

WRAPPER_TARGETS = {
    "calibration": "unity_psf.optics.psf.double_helix.calibration",
    "dataset": "unity_psf.optics.psf.double_helix.dataset",
    "field_fit": "unity_psf.optics.psf.double_helix.field_fit",
    "field_gamma": "unity_psf.optics.psf.double_helix.field_gamma",
    "gamma_field": "unity_psf.optics.psf.double_helix.gamma_field",
    "lg_calibration": "unity_psf.optics.psf.double_helix.lg_calibration",
    "lg_carrier": "unity_psf.optics.psf.double_helix.lg_carrier",
    "local_fit": "unity_psf.optics.psf.double_helix.local_fit",
    "localization": "unity_psf.optics.psf.double_helix.localization",
    "lut": "unity_psf.optics.psf.double_helix.lut",
    "physical_update": "unity_psf.optics.psf.double_helix.physical_update",
    "pixel_pupil_calibration": "unity_psf.optics.psf.double_helix.pixel_pupil_calibration",
    "shared_carrier_field": "unity_psf.optics.psf.double_helix.shared_carrier_field",
    "vector_model": "unity_psf.optics.psf.double_helix.vector_model",
}

MODULE_PATTERN = re.compile(r"\bdouble_helix(?:\.legacy)?\.[A-Za-z0-9_]+")


def test_compatibility_wrappers_export_canonical_objects() -> None:
    for wrapper_name, target_name in WRAPPER_TARGETS.items():
        wrapper = importlib.import_module(f"double_helix.{wrapper_name}")
        target = importlib.import_module(target_name)
        assert wrapper.__all__, f"{wrapper_name} must keep an explicit public boundary"
        for name in wrapper.__all__:
            assert hasattr(target, name), f"{wrapper_name}.{name} has no canonical target"
            assert getattr(wrapper, name) is getattr(target, name)


def test_cli_compatibility_entries_keep_only_the_main_contract() -> None:
    calibration = importlib.import_module("double_helix.run_calibration")
    evaluation = importlib.import_module("double_helix.run_evaluation")
    assert calibration.__all__ == ["main"]
    assert evaluation.__all__ == ["main"]
    assert calibration.main is importlib.import_module("unity_psf.cli.double_helix_calibration").main
    assert evaluation.main is importlib.import_module("unity_psf.cli.double_helix_evaluation").main


def test_archive_module_references_resolve_to_existing_entrypoints() -> None:
    references = set()
    for path in ARCHIVE_ROOT.rglob("*"):
        if path.is_file() and path.suffix in {".sh", ".sbatch", ".py"}:
            references.update(MODULE_PATTERN.findall(path.read_text()))

    for reference in sorted(references):
        module_path = SRC_ROOT.joinpath(*reference.split("."))
        assert module_path.with_suffix(".py").is_file(), reference


def test_formal_unity_source_does_not_import_legacy_namespace() -> None:
    imports = []
    for path in FORMAL_ROOT.rglob("*.py"):
        imports.extend(MODULE_PATTERN.findall(path.read_text()))
    assert not [item for item in imports if item.startswith("double_helix.legacy.")]
