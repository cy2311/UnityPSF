from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def quality_runner(tmp_path: Path):
    project = tmp_path / "standalone project"
    scripts = project / "scripts"
    scripts.mkdir(parents=True)
    shutil.copyfile(PROJECT_ROOT / "scripts/quality_check.sh", scripts / "quality_check.sh")
    interpreter = tmp_path / "chosen python"
    interpreter.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "from pathlib import Path\n"
        "arguments = sys.argv[1:]\n"
        "if arguments[0] == '-c':\n"
        "    sys.exit(1 if arguments[1] == 'import ' + os.environ.get('MISSING_TOOL', '') else 0)\n"
        "with Path(os.environ['COMMAND_LOG']).open('a') as handle:\n"
        "    handle.write(json.dumps({'arguments': arguments, 'cwd': os.getcwd(), "
        "'mypy_cache': os.environ.get('MYPY_CACHE_DIR'), "
        "'bytecode': os.environ.get('PYTHONDONTWRITEBYTECODE')}) + '\\n')\n"
        "sys.exit(4 if arguments[1] == os.environ.get('FAIL_TOOL') else 0)\n",
        encoding="utf-8",
    )
    interpreter.chmod(0o755)
    command_log = tmp_path / "commands.jsonl"
    environment = {
        **os.environ,
        "PYTHON": str(interpreter),
        "QUALITY_TOOLS_DIR": str(tmp_path / "no-extra-tools"),
        "COMMAND_LOG": str(command_log),
    }
    environment.pop("MISSING_TOOL", None)
    environment.pop("FAIL_TOOL", None)

    def run(**overrides: str):
        result = subprocess.run(
            ["bash", str(scripts / "quality_check.sh")],
            cwd=tmp_path,
            env={**environment, **overrides},
            capture_output=True,
            text=True,
            timeout=30,
        )
        commands = [json.loads(line) for line in command_log.read_text().splitlines()] if command_log.exists() else []
        return result, commands, project

    return run


def test_quality_check_uses_selected_interpreter_and_project_local_caches(quality_runner) -> None:
    result, commands, project = quality_runner()

    assert result.returncode == 0, result.stderr
    assert [command["arguments"][1] for command in commands] == ["ruff", "mypy", "pytest"]
    assert all(command["arguments"][0] == "-m" for command in commands)
    assert all(command["cwd"] == str(project) for command in commands)
    assert all(command["mypy_cache"] == str(project / ".local/cache/mypy") for command in commands)
    assert all(command["bytecode"] == "1" for command in commands)
    assert commands[0]["arguments"][-2:] == ["--cache-dir", ".local/cache/ruff"]
    assert commands[-1]["arguments"][-2:] == ["-o", "cache_dir=.local/cache/pytest"]


def test_quality_check_reports_missing_tool_before_running_checks(quality_runner) -> None:
    result, commands, _project = quality_runner(MISSING_TOOL="mypy")

    assert result.returncode == 2
    assert "missing quality tool: mypy" in result.stderr
    assert "dev extra" in result.stderr
    assert commands == []


@pytest.mark.parametrize("failed_tool,expected_tools", [("ruff", ["ruff"]), ("mypy", ["ruff", "mypy"]), ("pytest", ["ruff", "mypy", "pytest"])])
def test_quality_check_stops_at_first_failure(quality_runner, failed_tool: str, expected_tools: list[str]) -> None:
    result, commands, _project = quality_runner(FAIL_TOOL=failed_tool)

    assert result.returncode == 4
    assert [command["arguments"][1] for command in commands] == expected_tools
