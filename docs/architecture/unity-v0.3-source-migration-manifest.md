# UnityPSF v0.3 source migration manifest

- Date: 2026-09-24
- Source snapshot: current working tree at `unity/src`, `unity/tests`, and `unity/scripts`, plus the PPG/PCA backend at `ppg/`
- Destination: `unity_v0.3/src`, `unity_v0.3/tests`, and `unity_v0.3/scripts`
- Runtime/cache exclusions: `__pycache__/`, `*.pyc`, `*.pyo`
- Source files copied: 184
- Test files copied: 45
- Script files copied: 73
- PPG backend modules copied: 22
- PPG backend tests copied: 7

## Included

- `unity/src/unity_psf`: active UnityPSF package
- `unity/src/double_helix`: reusable DH calibration and physics package
- `unity/tests`: package, contract, model, physics, and training tests
- `unity/scripts`: reusable train, infer, and diagnostic entrypoints
- `unity_v0.3/src/ppg`: canonical PPG/PCA physical backend with 36-parameter Legendre map updates
- `unity_v0.3/tests/ppg`: PPG backend contract and physical-update tests
- `unity_v0.3/configs/calibration/ppg`: small, versioned PPG map/catalog fixtures used by tests and default config

## Explicitly excluded

- `unity/.local`, `unity/logs`, `unity/output`, `unity/datasets`, and caches
- Python bytecode and pytest/matplotlib caches
- the old `unity/README.md` and `unity/configs`: v0.3 owns its own README and configs
- external vendor repositories and generated checkpoints
- the original `ppg/` working tree: retained as provenance/reference, not an import source for v0.3

## Ownership after migration

`unity_v0.3` is the only editable mainline for new UnityPSF work. The source
tree at `unity` remains untouched by this migration and is retained as a
reference for provenance and comparison. Divergence between the two trees must
be resolved through an explicit migration change, never through an implicit
 `PYTHONPATH` order.

## Path and script boundary

Active scripts under `scripts/train`, `scripts/infer`, and `scripts/ppg` resolve
the project root from their own location and write runtime files only under
`logs/`, `output/`, or `.local/`. Historical archive and replay scripts were
removed during the final v0.3 cleanup and are no longer part of this tree.

Config-owned artifacts use `project://...` references. The runtime resolver
maps those references to the nearest v0.3 root containing `pyproject.toml`,
which keeps active configs independent of the old `unity` checkout path.
