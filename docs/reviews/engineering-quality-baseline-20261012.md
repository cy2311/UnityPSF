# Engineering quality baseline — 2026-10-12

## Scope and priority

The first priority is an executable quality baseline, not another structural
refactor or a larger training run. The earlier comparison report recorded a
passing test suite but could not run Ruff. Investigation also found that one
test imported `Normalization` from a sibling checkout, while the built wheel
omitted the CUDA renderer's `bindings.cpp` and `kernels.cu`.

This change is confined to `unity_v0.3`. It does not change training defaults,
checkpoint semantics, calibration inputs, or model behavior.

## Changes

- The gate uses the selected Python interpreter for Ruff, mypy and pytest,
  checks tool availability first, fails immediately on a failing stage, and
  keeps caches under `.local/cache/` without writing source-tree bytecode.
- The gain contract test now uses a fixed synthetic background reference,
  rather than importing the Neptune implementation from a sibling checkout.
  An architecture test rejects the known sibling package imports.
- The wheel includes the CUDA renderer's C++ and CUDA source files. A source
  boundary test guards the package-data declaration, and a real wheel was
  built and inspected to verify the files are present.
- Regression tests cover interpreter selection, invocation from outside the
  project, paths containing spaces, missing tools and failure propagation.

## Reproduce

```bash
cd unity_v0.3
python -m pip install -e ".[dev]"
bash scripts/quality_check.sh
python -m pip wheel --no-deps --no-build-isolation \
  --wheel-dir .local/tmp/quality-wheel-check .
```

The wheel build requires the declared setuptools and wheel build tools in the
chosen environment when `--no-build-isolation` is used. The gate permits an
explicit `PYTHON` interpreter and project-local `QUALITY_TOOLS_DIR` rather
than installing or upgrading dependencies implicitly.

## Verification receipt

Executed in this checkout on 2026-10-12 with Python 3.11, Ruff 0.6.9,
mypy 1.8.0 and pytest 7.4.0:

| Check | Result |
| --- | --- |
| `bash scripts/quality_check.sh` | Exit 0; Ruff clean, mypy clean in six modules, **292 passed, 1 skipped** in 70.56 seconds |
| Targeted gate, package-boundary and gain tests | **12 passed** |
| Wheel build and ZIP content assertions | Exit 0; `bindings.cpp` and `kernels.cu` both present |
| Gain contract with sibling `Normalization` imports blocked | Passed |

The skip is the existing CUDA-GEMM simulation test: this session has no CUDA
runtime. No GPU training was launched. Ruff was downloaded into
`.local/quality-tools` without modifying the global Python installation.
The wheel build directory was moved into `.local/tmp/quality-wheel-build`;
the wheel itself remains under `.local/tmp/quality-wheel-check`.

## Remaining priorities

1. **Standalone package inputs:** move default training templates into
   installable resources and test `unity-psf-train plan` outside the checkout.
   CUDA source inclusion alone does not make the CLI independently usable.
2. **Actual enforcement:** connect the gate at the canonical Git repository
   root. The current project directory is untracked in the parent repository;
   its nested workflow is not active. This task does not modify that parent
   repository or create commits.
3. **Reproducible environments:** record compatible runtime and quality-tool
   constraints. The current baseline is one installed environment, not proof
   that every unconstrained dependency combination works.
4. **GPU and scientific acceptance:** run the existing CUDA/SLURM smoke and
   retain the receipt, then measure held-out localization accuracy and
   resource consumption. CPU tests and wheel inspection do not prove GPU
   compilation, training throughput or scientific quality.

## Workspace boundary

Downloaded Ruff tools and wheel inspection artifacts stay under `.local/`.
Existing root caches, source-tree cached artifacts and the parent's unrelated
dirty files are not part of this fix and are not removed. The workspace is
not a clean release checkout.
