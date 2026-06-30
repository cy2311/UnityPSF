from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from .load import materialize_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Materialize a Neptune v0.3 config from base plus overrides.")
    parser.add_argument("--base", required=True, help="Base YAML config.")
    parser.add_argument("--override", action="append", default=[], help="Override YAML config. Repeatable.")
    parser.add_argument("--output", required=True, help="Output path for the resolved config.")
    parser.add_argument("--format", choices=("yaml", "json"), default="yaml", help="Output format.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    resolved = materialize_config(args.base, *args.override)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.format == "json":
        output.write_text(json.dumps(resolved, indent=2) + "\n", encoding="utf-8")
    else:
        output.write_text(yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
