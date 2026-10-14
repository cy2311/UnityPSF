from __future__ import annotations

import ast
from pathlib import Path

import pytest


SOURCE = Path(__file__).parents[2] / "src" / "unity_psf"
TRAINING = SOURCE / "training"


@pytest.mark.parametrize("filename", ["loop.py", "modality_joint.py"])
def test_training_facades_contain_exports_not_implementations(filename: str) -> None:
    path = TRAINING / filename
    source = path.read_text(encoding="utf-8")
    assert len(source.splitlines()) <= 150
    assert not any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        for node in ast.parse(source).body
    )


def test_internal_code_does_not_depend_on_training_facades() -> None:
    forbidden = {"unity_psf.training.loop", "unity_psf.training.modality_joint"}
    violations = []
    for path in SOURCE.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module in forbidden:
                violations.append(f"{path.relative_to(SOURCE)}:{node.lineno}")
            if isinstance(node, ast.ImportFrom) and path == TRAINING / "__init__.py":
                if node.module in {"loop", "modality_joint"}:
                    violations.append(f"{path.relative_to(SOURCE)}:{node.lineno}")
    assert not violations, violations


@pytest.mark.parametrize(
    "filename",
    [
        "loop_types.py", "epoch_loop.py", "training_run.py", "evaluation.py", "resume.py",
        "checkpoint_io.py", "modality_types.py", "modality_epoch.py", "modality_evaluation.py",
        "modality_checkpoint.py", "modality_physical_state.py",
    ],
)
def test_training_owners_stay_within_single_workflow_budget(filename: str) -> None:
    path = TRAINING / filename
    assert len(path.read_text(encoding="utf-8").splitlines()) <= 350
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            assert not node.module.startswith("unity_psf.cli")
            assert not node.module.startswith("unity_psf.training.entrypoints")


def test_generic_compatibility_exports_are_the_canonical_objects() -> None:
    from unity_psf.training import loop
    from unity_psf.training.epoch_loop import train_one_epoch
    from unity_psf.training.evaluation import evaluate
    from unity_psf.training.loop_types import TrainingBatch, TrainingConfig
    from unity_psf.training.resume import resume_training_checkpoint
    from unity_psf.training.training_run import train_epochs

    assert loop.train_one_epoch is train_one_epoch
    assert loop.train_epochs is train_epochs
    assert loop.evaluate is evaluate
    assert loop.TrainingBatch is TrainingBatch
    assert loop.TrainingConfig is TrainingConfig
    assert loop.resume_training_checkpoint is resume_training_checkpoint


def test_split_owners_do_not_import_cross_module_private_interfaces() -> None:
    owners = [
        *TRAINING.glob("modality_*.py"),
        *TRAINING.glob("epoch_loop.py"),
        *TRAINING.glob("training_run.py"),
        *TRAINING.glob("resume.py"),
        *(SOURCE / "localization" / "data").glob("online_*.py"),
        *(TRAINING / "entrypoints").glob("expert_parallel_*.py"),
    ]
    for path in owners:
        if path.name == "modality_joint.py":
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.ImportFrom):
                assert all(not alias.name.startswith("_") for alias in node.names), (
                    path, node.lineno,
                )


def test_internal_online_consumers_import_owners_not_facade() -> None:
    for path in SOURCE.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.ImportFrom):
                assert node.module != "unity_psf.localization.data.online", (path, node.lineno)


@pytest.mark.parametrize(
    "filename",
    ["expert_parallel_cli.py", "expert_parallel_worker.py", "expert_parallel_coordination.py",
     "expert_parallel_artifacts.py", "expert_parallel_release.py"],
)
def test_expert_parallel_owners_stay_bounded(filename: str) -> None:
    path = TRAINING / "entrypoints" / filename
    assert len(path.read_text(encoding="utf-8").splitlines()) <= 350


def test_expert_parallel_entrypoint_is_only_exports_and_launch() -> None:
    path = TRAINING / "entrypoints" / "train_modality_expert_parallel.py"
    source = path.read_text(encoding="utf-8")
    assert len(source.splitlines()) <= 180
    assert not any(isinstance(node, (ast.FunctionDef, ast.ClassDef)) for node in ast.parse(source).body)
