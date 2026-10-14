from __future__ import annotations

import ast
from pathlib import Path


PROJECT_ROOT = Path(__file__).parents[2]
HIGH_FIDELITY = PROJECT_ROOT / "src" / "unity_psf" / "training" / "high_fidelity"


def test_high_fidelity_engine_is_only_a_compatibility_facade() -> None:
    engine = HIGH_FIDELITY / "engine.py"
    assert sum(1 for _ in engine.open(encoding="utf-8")) <= 200
    tree = ast.parse(engine.read_text(encoding="utf-8"))
    public_functions = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and not node.name.startswith("_")
    ]
    assert {node.name for node in public_functions} <= {"main", "parse_args", "resume_epoch_training_config"}


def test_high_fidelity_workflows_are_present_and_runner_does_not_import_engine() -> None:
    required = {
        "runner.py",
        "peak_bootstrap.py",
        "roi_workflow.py",
        "posterior_workflow.py",
        "gamma_workflow.py",
        "heldout_monitor.py",
        "physical_state.py",
        "diagnostics.py",
    }
    assert required <= {path.name for path in HIGH_FIDELITY.glob("*.py")}
    runner = ast.parse((HIGH_FIDELITY / "runner.py").read_text(encoding="utf-8"))
    assert not any(
        isinstance(node, ast.ImportFrom) and node.module == "unity_psf.training.high_fidelity.engine"
        for node in ast.walk(runner)
    )
