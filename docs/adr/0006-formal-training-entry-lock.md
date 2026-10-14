# ADR 0006: Lock the formal training entrypoint

- Status: Accepted
- Date: 2026-10-12
- Scope: Formal UnityPSF v0.3 training

## Decision

`unity-psf-train` is the only public entrypoint allowed to create a formal
UnityPSF release checkpoint. It materializes the resolved training contract and
passes its hash to the internal Expert Parallel executor. The executor rejects
direct launches or launches carrying a different contract hash.

Formal SLURM scripts call the CLI. The former fixed three-modality raw-TIFF
launcher was removed because it accepted a second joint YAML contract.

DH physical-update, high-fidelity, and localization commands remain auxiliary
workflows. They may produce their own intermediate artifacts, but they cannot
publish the canonical v2 UnityPSF release checkpoint.

## Consequences

- A formal run always has one resolved contract and one contract hash.
- Adding a formal launcher requires a CLI route and an entrypoint-lock test.
- Internal training modules remain importable for implementation reuse and
  tests, but direct process execution fails before training starts.
