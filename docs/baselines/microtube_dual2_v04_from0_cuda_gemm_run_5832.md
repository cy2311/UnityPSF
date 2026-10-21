# Run 5832 Baseline

This records `results/microtube_dual2_v04_from0_cuda_gemm/runs/5832` as the
reference training process and output for subsequent project-wide optimization.

## Recorded state

- Pipeline status: `completed`.
- Training stage: `high_fidelity_localization`, epoch 300, global step 125100.
- Seed: `20260711`; resolved settings and training metadata are retained in the
  run directory.
- The baseline covers every regular file in the run directory, including
  checkpoints, metrics, resolved configuration, generated artifacts, and
  diagnostics. The adjacent SHA-256 manifest records their exact bytes.

## Verification

From the repository root, run:

```bash
python scripts/baseline/verify_run_baseline.py
```

The verifier reports changed, missing, or added files and exits nonzero on any
difference. Do not regenerate the manifest after optimization; doing so would
replace the reference. The manifest captures the run outputs, not a complete
source-code/environment snapshot, so matching hashes establish artifact
integrity rather than bitwise reproducibility of a future training rerun.

## Workspace boundary

Future optimization may change source, configuration, and scripts, but must not
modify, rename, or delete the run 5832 directory or its checksum manifest.
Store new runs under a distinct run ID and compare their relevant output
contracts with this baseline. Existing unrelated working-tree changes are
outside this freeze and have not been reverted or incorporated.
