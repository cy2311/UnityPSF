# Compatibility Surfaces

The supported public namespace is `unity_psf`. New code must import from the
canonical modules listed below.

## Retained shims

| Legacy import | Canonical import | Reason |
| --- | --- | --- |
| `unity_psf.localization.runtime_config` | `unity_psf.localization.runtime.config` | Existing external training scripts may still import the old module path. |
| `unity_psf.localization.online` / `unity_psf.localization.data.online` | `unity_psf.localization.data.online_types` / `online_provider` | Existing evaluation and data preparation scripts may still import the old module paths. |
| `unity_psf.training.loop` | `loop_types`, `epoch_loop`, `training_run`, `evaluation`, `resume` | Re-exports preserve generic training integrations; internal code uses implementation owners. |
| `unity_psf.training.modality_joint` | `modality_types`, `modality_epoch`, `modality_evaluation`, `modality_checkpoint` | Re-exports preserve modality integrations without a second implementation. |
| `InstanceRouter` in `unity_psf.models.psf_moe.router` | `ModalityRouter` for new models | Required to read legacy v1 joint checkpoints; it is never used for new checkpoints. |
| `legacy_decode.py` | current localization evaluation contract | PPG diagnostics and LiteLoc-compatible evaluation still consume this decoder. |
| `double_helix` package | UnityPSF DH calibration and physical update code | This is active calibration functionality, not a namespace alias. |

The two localization shim modules contain exports only. Internal code and tests
use the canonical paths so new dependencies on the shims do not accumulate.
`localization.runtime.config` retains only build orchestration and compatibility
exports; model, loss, conditioning, provider and contract resolution have
separate implementation owners.

## Training entrypoints

`unity-psf-train` (`unity_psf.cli.train_selection`) is the only formal training
entrypoint. It launches the internal modality Expert Parallel executor and
publishes `unity_psf.joint_checkpoint.v2` release files.

The internal module `unity_psf.training.entrypoints.train_modality_expert_parallel`
is not a public entrypoint. It refuses to start unless launched by
`unity-psf-train` with the matching resolved contract hash. SLURM training
scripts must call the CLI and must not invoke the internal module directly.

`unity-psf-train-high-fidelity`, `unity-psf-train-localization`, and
`ppg.train_dh_physical` are auxiliary workflows. They do not publish the
canonical UnityPSF release checkpoint and are outside the formal modality
selection contract.

The former per-channel and joint training orchestrators were removed after the
canonical migration. Historical jobs must be converted to `unity-psf-train`;
there is no second training implementation to keep compatible.
The old v1 checkpoint loader remains read-only; canonical release validation
requires v2 plus `training_entrypoint` provenance.

## Removed

`neptune_v04` was a one-file namespace alias with no repository consumers, no
checkpoint references, and no active launcher usage. It is no longer included
in the v0.3 package or wheel. Historical documents may still mention the old
baseline name; those references describe provenance and are not import
contracts.
