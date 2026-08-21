# Unity v0.2 Test Fixture Audit

Status: Phase 6 complete on 2026-08-21

This audit checks test setup duplication without introducing a universal
fixture factory. A helper is a valid extraction candidate only when it has at
least two real consumers, the same semantics and defaults, no hidden contract
assertions, and a measurable reduction in repeated setup.

## Result

No fixture was extracted in this phase. The current repeated-looking helpers
were reviewed and intentionally kept local because their setup encodes
different contracts or boundary conditions.

| Candidate | Consumers | Comparison | Decision |
| --- | --- | --- | --- |
| `_provider_config` in `tests/localization/test_online_provider_contract.py` and the inline configs in `test_conditioning_profile.py` | Online provider route tests; conditioning/seed/profile tests | Same provider type, but different purpose: route contract uses a minimal deterministic batch, while conditioning tests vary seed, epoch, cached-window, and profile semantics | Keep local; extracting a shared default would hide which fields each contract needs |
| `_coeff_map` in `tests/training/test_channel_physical_context.py` and `test_multichannel_physical_isolation.py` | Single-channel physical context tests; left/right isolation and restore tests | Same six-mode NPZ shape and mode order; one helper creates parent directories, the other relies on `tmp_path`; call sites encode different channel and state ownership | Keep local; a shared helper would add a test-support API for only two near-identical call sites and erase the directory precondition difference |
| `_config` helpers in runtime tests | `test_astigmatism_runtime.py`, config snapshot tests, and CLI/config tests | Each loads a different schema or fixture path and intentionally preserves legacy/formal differences | Keep local; merging would create a schema-selecting factory |
| `_model`, `_batch`, `_loss` helpers in training/model tests | Several modality and checkpoint tests | Similar names but different model classes, output contracts, target shapes, and optimization semantics | Keep local; these are characterization fixtures, not shared behavior |
| `tmp_path` file setup | Many tests | Paths are intentionally isolated per test and contain different artifact schemas | Keep local; no shared lifecycle exists |

There are no `pytest.fixture` definitions in the test tree. Existing helpers
are module-local functions, so they do not create a hidden global fixture
dependency.

## Acceptance checks

- At least two real consumers were required before considering extraction.
- Provider, modality, channel, and checkpoint contracts were compared field by
  field rather than by helper name.
- No assertion was moved behind a parametrized or universal factory.
- No fixture module, `conftest.py`, or test-wide facade was added.

The correct Phase 6 outcome is therefore zero extracted fixtures and an audit
record explaining why. Future extraction should happen only when a third
consumer appears with identical semantics, or when the same helper can be
moved without adding defaults or branching on modality/channel.
