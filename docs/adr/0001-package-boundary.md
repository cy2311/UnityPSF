# ADR 0001: Installable Package Boundary

- Status: Accepted
- Date: 2026-08-02
- Scope: Neptune v0.4 package distribution

## Context

The main package uses a `src/neptune_v04` layout, but the reusable
double-helix calibration code is still located at the project root in
`double_helix/`. The existing SLURM jobs invoke it with commands such as
`python -m double_helix.run_calibration`. With `setuptools` restricted to the
`src/` directory, those commands work only when the repository root happens to
be on `sys.path`; they do not represent an independently installable package.

## Decision

For the first migration slice, `pyproject.toml` discovers both package roots
and includes `neptune_v04*` and `double_helix*` in the distribution. The
existing `double_helix` module path remains stable so current jobs do not need
to be rewritten in the same change.

The physical implementation will not be copied or duplicated. A later slice
will move reusable double-helix code into the Neptune namespace and leave
thin compatibility entry points only while all SLURM consumers are migrated.

## Consequences

- Editable installs and built wheels contain both Python packages.
- `python -m double_helix...` remains backward compatible for current jobs.
- The source tree still has a temporary two-root layout; this is intentional
  migration debt, not the final architecture.
- The next migration slice must add package-level calibration entry points,
  migrate consumers incrementally, and then remove the compatibility path.

## Verification

- Editable install succeeds with `pip install -e neptune_v0.4`.
- A built wheel contains 81 `neptune_v04` Python files and 34 `double_helix`
  Python files.
- A wheel installed into an isolated temporary directory imports both
  `neptune_v04` and `double_helix.run_fd_zmap` from outside the repository.
