"""Channel physical-state snapshots and calibration provenance."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any

from unity_psf.training.modality_types import ModalityChannelStream, ModalityTrainingRuntime


def physical_provenance_registry(
    runtimes: Mapping[str, "ModalityTrainingRuntime"],
) -> dict[str, dict[str, Any]]:
    return {
        str(modality): {
            str(channel_id): copy.deepcopy(dict(stream.calibration).get("physical_provenance"))
            for channel_id, stream in runtime.channels.items()
            if dict(stream.calibration).get("physical_provenance") is not None
        }
        for modality, runtime in runtimes.items()
    }


def snapshot_channel_physical_state(
    stream: ModalityChannelStream,
) -> dict[str, Any]:
    state = (
        stream.physical_state
        if stream.snapshot_physical_state is None
        else stream.snapshot_physical_state()
    )
    if not isinstance(state, Mapping):
        raise TypeError("channel physical state snapshot must be a mapping")
    snapshot = copy.deepcopy(dict(state))
    if isinstance(stream.physical_state, dict):
        stream.physical_state.clear()
        stream.physical_state.update(copy.deepcopy(snapshot))
    return snapshot
