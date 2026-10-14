# ADR 0005: One resolved training contract

- Status: Accepted
- Date: 2026-10-11
- Scope: Canonical UnityPSF training in `unity_v0.3`

## Context

The selection CLI, joint YAML, modality configs, runtime builder, resume
checks, and checkpoint provenance each interpreted part of the training setup.
That allowed the displayed plan and the executed runtime to drift, especially
when a resolved YAML was edited after planning.

## Decision

`unity_psf.contracts.training.ResolvedTrainingContract` is the sole resolved
semantic contract for a formal run. It owns the selected modality and channel
instances, seeds, step budgets, epochs, resolved configs, external input
records, execution mode, and required rank count.

The selection CLI writes `training_contract.json` and derives `joint.yaml` from
the same object. The executor must load that contract before constructing a
runtime, verify the joint instance identity and every resolved YAML against it,
and use the contract hash as the resume and checkpoint identity. Release
checkpoints declare the contract schema and hash in provenance.

All formal run artifacts use `resolved_training_contract_hash` as their single
contract identity. No separate training signature is calculated or serialized.

## Consequences

- A changed route, seed, budget, optimizer, anchor, DH PCA input, or resolved
  modality config produces a different contract hash.
- Resume and release publication fail before training when the contract,
  joint YAML, or resolved config files disagree.
- The old v1 checkpoint reader remains read-only; new formal releases require
  the v2 joint checkpoint and canonical contract provenance.
- New training options must be added to the contract builder and its tests
  before they can be consumed by the runtime.

## Verification

The contract tests cover stable serialization, semantic hash changes, joint
identity, and rejection of modified resolved YAML. The full test suite passes
with `285 passed, 1 skipped`.
