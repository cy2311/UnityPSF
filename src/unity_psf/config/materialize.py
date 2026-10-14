from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from .load import materialize_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Materialize a UnityPSF v0.3 config from base plus overrides.")
    parser.add_argument("--base", required=True, help="Base YAML config.")
    parser.add_argument("--override", action="append", default=[], help="Override YAML config. Repeatable.")
    parser.add_argument(
        "--set",
        dest="set_values",
        action="append",
        default=[],
        metavar="PATH=VALUE",
        help="Set a scalar value after YAML overrides, using dotted paths. Repeatable.",
    )
    parser.add_argument("--output", required=True, help="Output path for the resolved config.")
    parser.add_argument("--format", choices=("yaml", "json"), default="yaml", help="Output format.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    resolved = materialize_config(args.base, *args.override)
    for assignment in args.set_values:
        _apply_assignment(resolved, assignment)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.format == "json":
        output.write_text(json.dumps(resolved, indent=2) + "\n", encoding="utf-8")
    else:
        output.write_text(yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8")
    return 0


def _apply_assignment(config: dict[str, Any], assignment: str) -> None:
    path, separator, raw_value = assignment.partition("=")
    if not separator or not path or not raw_value:
        raise ValueError(f"expected PATH=VALUE assignment, got {assignment!r}")
    keys = path.split(".")
    target: dict[str, Any] = config
    for key in keys[:-1]:
        value = target.get(key)
        if value is None:
            value = {}
            target[key] = value
        if not isinstance(value, dict):
            raise TypeError(f"cannot descend through non-mapping config key: {key}")
        target = value
    target[keys[-1]] = yaml.safe_load(raw_value)


if __name__ == "__main__":
    raise SystemExit(main())
