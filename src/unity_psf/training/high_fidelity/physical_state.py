from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from unity_psf.training.channel_context import (
    ChannelTrainingContext,
    atomic_write_json,
    sha256_file,
)


def _write_initial_physical_state(
    layout,
    runtime_config: Mapping[str, Any],
    *,
    physical_context: ChannelTrainingContext | None = None,
) -> str | None:
    if physical_context is not None:
        return physical_context.write_physical_state(source="initial")
    entries = _runtime_dual_domain_coeff_maps(runtime_config)
    if not entries:
        return None
    return _write_current_physical_state(
        layout,
        coeff_maps=entries,
        source="initial",
        epoch=None,
        global_step=None,
        condition_store_version=None,
    )


def _runtime_dual_domain_coeff_maps(runtime_config: Mapping[str, Any]) -> tuple[dict[str, str], ...]:
    batch_provider = runtime_config.get("batch_provider")
    if not isinstance(batch_provider, Mapping):
        return ()
    params = batch_provider.get("params")
    if not isinstance(params, Mapping):
        return ()
    entries = params.get("dual_domain_coeff_maps")
    if not isinstance(entries, (tuple, list)):
        return ()
    output = []
    for idx, item in enumerate(entries):
        if not isinstance(item, Mapping):
            continue
        path = item.get("coeff_maps_npz") or item.get("alternating_coeff_maps_npz") or item.get("path")
        if path is None:
            continue
        output.append({"name": str(item.get("name", f"domain{idx}")), "coeff_maps_npz": str(path)})
    return tuple(output)


def _write_current_physical_state(
    layout,
    *,
    coeff_maps,
    source: str,
    epoch: int | None,
    global_step: int | None,
    condition_store_version: int | None,
) -> str:
    entries = tuple({"name": str(item["name"]), "coeff_maps_npz": str(item["coeff_maps_npz"])} for item in coeff_maps)
    payload = {
        "source": str(source),
        "epoch": None if epoch is None else int(epoch),
        "global_step": None if global_step is None else int(global_step),
        "condition_store_version": None if condition_store_version is None else int(condition_store_version),
        "coeff_maps": list(entries),
    }
    path = layout.metadata_dir / "current_physical_state.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(path, payload)
    state_hash = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    manifest_path = layout.metadata_dir / "run_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["current_physical_state"] = payload
        manifest["current_physical_coeff_maps"] = list(entries)
        manifest.setdefault("initial_physical_state_hash", state_hash)
        manifest["latest_physical_state_hash"] = state_hash
        atomic_write_json(manifest_path, manifest)
    return str(path)


def _physical_checkpoint_extra_fn(
    layout,
    *,
    physical_context: ChannelTrainingContext | None = None,
):
    def checkpoint_extra() -> dict[str, object]:
        state = _read_current_physical_state(
            layout,
            path=None if physical_context is None else physical_context.physical_state_path,
        )
        if state is None:
            return {}
        physical_coeff_maps = state.get("coeff_maps", [])
        if physical_context is not None:
            physical_context.validate_physical_state_identity(state)
            if not isinstance(physical_coeff_maps, list):
                raise ValueError("physical state coeff_maps must be a list")
            if len(physical_coeff_maps) > 1:
                raise ValueError("physical state must contain exactly one coefficient map per channel")
            for item in physical_coeff_maps:
                if not isinstance(item, Mapping):
                    raise ValueError("physical state coeff_maps entries must be mappings")
                if str(item.get("name", "")) != physical_context.channel.channel_id:
                    raise ValueError("physical state coefficient map channel_id does not match the context")
                path_value = item.get("coeff_maps_npz")
                if path_value is None or not Path(str(path_value)).is_file():
                    raise FileNotFoundError(path_value)
                expected_hash = item.get("sha256")
                if str(state.get("schema_version", "")) == "unitypsf.channel_physical_state.v1" and expected_hash is None:
                    raise ValueError("physical state coefficient map hash is required")
                if expected_hash is not None and str(expected_hash) != sha256_file(str(path_value)):
                    raise ValueError(f"physical artifact hash mismatch: {path_value}")
            peak_path_value = state.get("peak_zmap_path")
            if peak_path_value is not None:
                peak_path = Path(str(peak_path_value)).resolve()
                if not peak_path.is_file():
                    raise FileNotFoundError(peak_path)
                expected_peak_hash = state.get("peak_zmap_sha256")
                if str(state.get("schema_version", "")) == "unitypsf.channel_physical_state.v1" and expected_peak_hash is None:
                    raise ValueError("physical state peak zmap hash is required")
                if expected_peak_hash is not None and str(expected_peak_hash) != sha256_file(peak_path):
                    raise ValueError(f"peak zmap artifact hash mismatch: {peak_path}")
        return {
            "physical_state": state,
            "physical_coeff_maps": physical_coeff_maps,
            **(
                {"physical_state_hash": physical_context.latest_physical_state_hash}
                if physical_context is not None and physical_context.latest_physical_state_hash is not None
                else {}
            ),
            **(
                {"initial_physical_state_hash": physical_context.initial_physical_state_hash}
                if physical_context is not None and physical_context.initial_physical_state_hash is not None
                else {}
            ),
        }

    return checkpoint_extra


def _read_current_physical_state(layout, *, path: Path | None = None) -> dict[str, object] | None:
    path = path or layout.metadata_dir / "current_physical_state.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))
