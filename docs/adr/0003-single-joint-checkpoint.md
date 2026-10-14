# ADR 0003: Single Joint Checkpoint and One UnityPSF Model Identity

- Status: Accepted
- Date: 2026-08-04
- Scope: UnityPSF release artifact, model loading, and multimodal training

## Context

The original post-task-10 plan treated UnityPSF as a directory bundle whose
manifest referenced separate prototype, channel-instance, and calibration
files. That structure preserved scientific isolation, but it made the public
artifact look like several models assembled by an external dispatcher.

The research claim is stronger and clearer: UnityPSF is one multimodal PSF
foundation model. A user should load one checkpoint, obtain one model object,
and call one localization API for every supported PSF family. Modality and
channel experts remain necessary, but they are internal conditional modules of
that model rather than separate user-facing products.

## Decision

The official release unit is one physical file named
`unitypsf_joint.ckpt`. Loading it creates one top-level `UnityPSF` model.

The joint checkpoint contains versioned nested state for:

- the UnityPSF model and input/output contracts;
- the modality and channel router;
- the expert registry and every supported expert instance;
- channel-local FiLM, peak zmap, gamma, and physical state;
- calibration payloads required for inference;
- provenance, supported modalities, code version, and integrity hashes;
- optional optimizer, scheduler, scaler, RNG, epoch, and global-step state for
  a training-resume checkpoint.

Raw TIFF files and large source datasets are not embedded. Their immutable
manifest and content hashes are recorded as provenance.

Every saved snapshot represents one complete UnityPSF model identity. Training
may keep `latest`, milestone, and release snapshots over time, but it must not
publish one checkpoint per expert as the final interface. Existing per-channel
checkpoints from tasks 1-10 remain valid migration inputs and legacy recovery
artifacts.

The checkpoint declares `supported_modalities`. The first formal release may
contain `emitter_2d` and `astigmatism` only. It must not create an empty or fake
Double Helix expert. A later upgrade adds `double_helix` while preserving the
same schema and top-level model API.

Loading may be eager or lazy. Lazy loading may materialize only the routed
expert on a GPU, but this is an execution optimization and does not change the
single checkpoint or single model identity.

## First Milestone

The first formal UnityPSF milestone is dual modality plus multichannel:

- real Origami data provides the initial `Emitter2DExpert(channel=main)` path;
- Astigmatism provides independent `left` and `right` instances;
- all three trained instances are stored in one `unitypsf_joint.ckpt`;
- one `UnityPSF` object routes both modalities and all declared channels;
- one run report exposes per-modality and per-channel results.

The generic channel contract also supports Emitter2D left/right instances when
paired 2D channel data is available. The first milestone does not pretend that
unpaired Origami TIFF groups are a paired left/right acquisition.

## Consequences

- Task 11 changes from directory-bundle assembly to joint-checkpoint assembly.
- Task 12 must introduce the top-level `UnityPSF` model and one loading API.
- Distributed expert training must gather state at a synchronization barrier;
  rank 0 validates and atomically commits the joint checkpoint.
- Checkpoint completeness and size require explicit tests.
- Scientific isolation remains unchanged: channel instances do not share
  parameters, optimizer state, peak zmap, gamma, or calibration state.
- The release artifact is easier to version, move, cite, and reproduce.

## Rejected Alternatives

### Directory bundle as the only release artifact

Rejected because users would still manage a manifest plus multiple model
files, weakening the one-model contract and making partial copies easy.

### Shared dense backbone for all PSFs

Rejected for the first production architecture because it would erase the
already validated complete-expert boundary and introduce a new scientific
assumption before multimodal parity is established.

### Empty Double Helix slot before data exists

Rejected because an untrained expert would make the checkpoint appear to
support a modality that has not passed scientific validation.

## Verification

- A single file round-trips on a different path without external model files.
- One top-level model loads Emitter2D main and Astigmatism left/right.
- Modifying any nested expert or calibration payload fails integrity checks.
- Lazy and eager loading return contract-equivalent outputs.
- A two-modality checkpoint refuses a Double Helix route with an explicit
  unsupported-modality error.
