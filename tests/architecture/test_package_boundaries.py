from __future__ import annotations

import ast
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = PROJECT_ROOT / "src" / "unity_psf"


def _import_roots(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.add(node.module.split(".", 1)[0])
    return roots


def test_unity_package_does_not_import_legacy_package_roots() -> None:
    violations: list[str] = []
    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        legacy_roots = _import_roots(path) & {"double_helix"}
        if legacy_roots:
            violations.append(f"{path.relative_to(PROJECT_ROOT)}: {sorted(legacy_roots)}")
    assert violations == []


def test_neptune_console_scripts_are_not_published() -> None:
    pyproject = (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert "neptune-v04-" not in pyproject


def test_tests_do_not_import_sibling_project_modules() -> None:
    violations = []
    for path in sorted((PROJECT_ROOT / "tests").rglob("*.py")):
        if _import_roots(path) & {"Normalization", "neptune_iwae", "neptune_v03", "neptune_v04"}:
            violations.append(str(path.relative_to(PROJECT_ROOT)))
    assert violations == []


def test_cuda_extension_sources_are_declared_as_package_data() -> None:
    pyproject = (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert '"unity_psf.localization.cuda_extension"' in pyproject
    assert '"*.cpp", "*.cu", "*.py"' in pyproject
    assert (PACKAGE_ROOT / "localization/cuda_extension/bindings.cpp").is_file()
    assert (PACKAGE_ROOT / "localization/cuda_extension/kernels.cu").is_file()
