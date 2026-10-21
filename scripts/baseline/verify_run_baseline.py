from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RUN = Path("results/microtube_dual2_v04_from0_cuda_gemm/runs/5832")
MANIFEST = Path("docs/baselines/microtube_dual2_v04_from0_cuda_gemm_run_5832.sha256")


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def expected_entries() -> dict[str, str]:
    run_root = ROOT / RUN
    return {
        path.relative_to(ROOT).as_posix(): digest(path)
        for path in sorted(run_root.rglob("*"))
        if path.is_file()
    }


def load_manifest() -> dict[str, str]:
    entries: dict[str, str] = {}
    for line_number, line in enumerate((ROOT / MANIFEST).read_text().splitlines(), 1):
        if not line.strip():
            continue
        checksum, relative_path = line.split("  ", 1)
        entries[relative_path] = checksum
    return entries


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify immutable run 5832 artifacts.")
    parser.add_argument("--write", action="store_true", help="Create the initial baseline manifest.")
    args = parser.parse_args()

    actual = expected_entries()
    manifest_path = ROOT / MANIFEST
    if args.write:
        if manifest_path.exists():
            parser.error(f"refusing to overwrite existing baseline: {MANIFEST}")
        manifest_path.write_text(
            "".join(f"{checksum}  {path}\n" for path, checksum in sorted(actual.items()))
        )
        print(f"Wrote {len(actual)} checksums to {MANIFEST}")
        return 0

    if not manifest_path.is_file():
        print(f"Missing baseline manifest: {MANIFEST}", file=sys.stderr)
        return 2

    expected = load_manifest()
    changed = sorted(path for path in expected.keys() & actual.keys() if expected[path] != actual[path])
    missing = sorted(expected.keys() - actual.keys())
    added = sorted(actual.keys() - expected.keys())
    if changed or missing or added:
        for label, paths in (("CHANGED", changed), ("MISSING", missing), ("ADDED", added)):
            for path in paths:
                print(f"{label}: {path}")
        return 1
    print(f"Baseline verified: {len(expected)} files match run 5832.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
