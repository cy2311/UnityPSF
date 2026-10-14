# High-fidelity training structure

## Decision

The high-fidelity training path is organized by workflow responsibility rather
than by the historical `engine.py` implementation file:

- `runner.py` owns CLI arguments, run lifecycle, resume, evaluation, and stage status.
- `peak_bootstrap.py` owns optional peak/z-map initialization.
- `roi_workflow.py`, `roi_bank_source.py`, `roi_bank_builder.py`, and `roi_split.py` own ROI source resolution and bank construction.
- `posterior_workflow.py` and `posterior_context.py` own posterior sampling and projection contexts.
- `gamma_workflow.py`, `gamma_projection.py`, and `heldout_monitor.py` own Gamma updates and held-out monitoring.
- `physical_state.py` owns physical artifacts and checkpoint metadata.
- `diagnostics.py` owns report and artifact generation.

`engine.py` is now a compatibility facade. New code must import the owning
workflow module directly; the facade exists only for established integrations.

## Guardrails

`tests/architecture/test_high_fidelity_boundaries.py` enforces that the facade
stays below 200 lines, exposes no training implementation functions, and that
the canonical runner does not import it.

## Verification

The refactor preserves the existing checkpoint, resume, Gamma, ROI, physical
state, and localization behavior. Full verification on October 12, 2026:

```text
343 passed, 1 skipped
compileall passed
```

No CUDA or multi-GPU runtime verification was performed in this refactor.
