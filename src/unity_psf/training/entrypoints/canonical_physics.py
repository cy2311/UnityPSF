"""Physical initialization provenance and formal runtime audits."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from unity_psf.contracts import ChannelLayout, JointExpertKey, PSFModality
from unity_psf.optics import resolve_astigmatism_anchor_profile, resolve_circular_psf_anchor_profile
from unity_psf.optics.empirical_psf import load_empirical_focal_psf
from unity_psf.training import ChannelTrainingContext
from unity_psf.training.channel_context import sha256_file


FORMAL_DENSITY_UM2 = 0.5
FORMAL_BACKGROUND_PHOTONS = 110.0
FORMAL_PHOTON_MEAN = 20000.0
FORMAL_PHOTON_SIGMA = 1000.0
FORMAL_PHOTON_RANGE = (0.0, 31000.0)
FORMAL_RECENTER_MODE = "fd_deeploc_exact_recenter"


def channel_metadata(
    runtime_config: Mapping[str, Any],
    *,
    physical_context: ChannelTrainingContext,
    config_path: Path,
    key: str,
    data_seed: int,
    experiment_metadata: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    parsed = JointExpertKey.parse(key)
    layout = ChannelLayout.from_value(runtime_config["channel_layout"])
    channel = layout[parsed.channel_id]
    physical_state = json.loads(
        physical_context.physical_state_path.read_text(encoding="utf-8")
    )
    calibration = {"calibration_ref": channel.calibration_ref}
    provenance = {
        "training_instance": key,
        "config": str(config_path),
        "data_seed": int(data_seed),
    }
    provider = runtime_config.get("batch_provider", {})
    provider_params = provider.get("params", {}) if isinstance(provider, Mapping) else {}
    sampling = sampling_recenter_metadata(provider_params)
    provenance.update(sampling)
    calibration.update(
        {
            name: sampling[name]
            for name in (
                "camera_baseline_adu",
                "camera_qe",
                "camera_e_per_adu",
                "train_background_adu",
                "recenter_mode",
            )
        }
    )
    if isinstance(provider_params, Mapping) and provider_params.get("psf_type") == "empirical_focal":
        empirical = load_empirical_focal_psf(
            str(provider_params["empirical_psf_path"]),
            channel_id=str(provider_params["empirical_psf_channel"]),
            focus_index=int(provider_params["empirical_psf_focus_index"]),
        )
        calibration.update(empirical.metadata())
    if isinstance(provider_params, Mapping) and provider_params.get("psf_type") == "vector":
        entries = provider_params.get("dual_domain_coeff_maps", ())
        if isinstance(entries, (list, tuple)) and len(entries) == 1 and isinstance(entries[0], Mapping):
            path_value = entries[0].get("coeff_maps_npz")
            if path_value is not None:
                path = Path(str(path_value)).resolve()
                calibration.update(
                    {
                        "psf_type": "vector",
                        "coefficient_map": str(path),
                        "coefficient_map_sha256": sha256_file(path),
                    }
                )
    if isinstance(provider_params, Mapping) and provider_params.get("psf_type") == "ppg_direct":
        path_value = provider_params.get("ppg_map_path")
        if not path_value:
            raise ValueError(f"PPG map path is missing for {key}")
        path = Path(str(path_value)).resolve()
        manifest = path.with_suffix(".json")
        if not path.is_file() or not manifest.is_file():
            raise ValueError(f"PPG map or beads manifest is missing for {key}: {path}")
        calibration.update(
            {
                "psf_type": "ppg_direct",
                "ppg_map_path": str(path),
                "ppg_map_sha256": sha256_file(path),
                "ppg_map_schema": "ppg.pca_map.v1",
                "beads_manifest_path": str(manifest),
                "beads_manifest_sha256": sha256_file(manifest),
            }
        )
    if runtime_config.get("physical_initialization") is not None:
        calibration["physical_initialization"] = runtime_config["physical_initialization"]
    calibration["physical_provenance"] = physical_provenance(
        parsed.modality,
        provider_params,
        calibration,
    )
    if experiment_metadata is not None:
        for name in ("real_sample_root", "real_sample_acquisition", "real_sample_frames"):
            if name in experiment_metadata:
                provenance[name] = experiment_metadata[name]
    return physical_state, calibration, provenance


def physical_provenance(
    modality: PSFModality,
    provider_params: Mapping[str, Any],
    calibration: Mapping[str, Any],
) -> dict[str, Any]:
    """Return a stable, machine-readable description of the physical start."""

    fixed_anchor_nm = provider_params.get("fixed_astig_anchor_nm")
    fixed_anchor = {
        "enabled": fixed_anchor_nm is not None and float(fixed_anchor_nm) != 0.0,
        "value_nm": None if fixed_anchor_nm is None else float(fixed_anchor_nm),
    }
    entries = provider_params.get("dual_domain_coeff_maps", ())
    if isinstance(entries, (list, tuple)) and entries:
        path = calibration.get("coefficient_map")
        sha256 = calibration.get("coefficient_map_sha256")
        if path is None or sha256 is None:
            raise ValueError("coefficient map provenance requires path and sha256")
        return {
            "schema": "unitypsf.physical_provenance.v1",
            "initialization": "coefficient_map",
            "anchor": None,
            "coefficient_map": {"path": str(path), "sha256": str(sha256), "mode": "absolute"},
            "fixed_anchor": fixed_anchor,
            "inputs": [{"role": "coefficient_map", "path": str(path), "sha256": str(sha256)}],
        }
    if modality is PSFModality.DOUBLE_HELIX:
        path = calibration.get("ppg_map_path")
        sha256 = calibration.get("ppg_map_sha256")
        manifest = calibration.get("beads_manifest_path")
        manifest_sha256 = calibration.get("beads_manifest_sha256")
        if None in (path, sha256, manifest, manifest_sha256):
            raise ValueError("PPG provenance requires map and beads manifest hashes")
        return {
            "schema": "unitypsf.physical_provenance.v1",
            "initialization": "beads_pca",
            "anchor": None,
            "coefficient_map": None,
            "fixed_anchor": {"enabled": False, "value_nm": None},
            "pca": {
                "map_path": str(path),
                "map_sha256": str(sha256),
                "beads_manifest_path": str(manifest),
                "beads_manifest_sha256": str(manifest_sha256),
                "method": "pca",
            },
            "inputs": [
                {"role": "ppg_pca_map", "path": str(path), "sha256": str(sha256)},
                {"role": "beads_manifest", "path": str(manifest), "sha256": str(manifest_sha256)},
            ],
        }
    profile_name = str(provider_params.get("conditioning_profile", ""))
    if modality is PSFModality.EMITTER_2D:
        profile = resolve_circular_psf_anchor_profile(profile_name)
        anchor = {
            "type": "circular_psf",
            "profile": profile.name,
            "radius_nm": float(provider_params.get("circular_psf_radius_nm", profile.radius_nm)),
            "radius_definition": profile.radius_definition,
        }
    else:
        profile = resolve_astigmatism_anchor_profile(profile_name)
        anchor = {
            "type": "astigmatism",
            "profile": profile.name,
            "wavelength_nm": float(profile.wavelength_nm),
            "anchor_nm": float(profile.anchor_nm),
        }
    return {
        "schema": "unitypsf.physical_provenance.v1",
        "initialization": "anchor_only",
        "anchor": anchor,
        "coefficient_map": None,
        "fixed_anchor": fixed_anchor,
        "inputs": [],
    }


def sampling_recenter_metadata(provider_params: Mapping[str, Any]) -> dict[str, Any]:
    density = provider_params.get("emitter_density_um2")
    background = tuple(float(value) for value in provider_params.get("background_range", ()))
    photon_range = tuple(float(value) for value in provider_params.get("photon_range", ()))
    photon_mean = provider_params.get("photon_mean")
    photon_sigma = provider_params.get("photon_sigma")
    camera_baseline = float(provider_params.get("camera_baseline", 0.0))
    camera_qe = float(provider_params.get("camera_qe", 0.0))
    camera_e_per_adu = float(provider_params.get("camera_e_per_adu", 0.0))
    width = int(provider_params.get("width", 0))
    height = int(provider_params.get("height", 0))
    pixel_size_nm_x = float(provider_params.get("pixel_size_nm_x", 0.0))
    pixel_size_nm_y = float(provider_params.get("pixel_size_nm_y", 0.0))
    density_value = None if density is None else float(density)
    target_active = None
    if density_value is not None:
        target_active = (
            density_value
            * width
            * pixel_size_nm_x
            / 1000.0
            * height
            * pixel_size_nm_y
            / 1000.0
        )
    train_background_photons = (
        background[0]
        if len(background) == 2 and background[0] == background[1]
        else None
    )
    train_background_adu = None
    if train_background_photons is not None and camera_e_per_adu > 0.0:
        train_background_adu = camera_baseline + train_background_photons * camera_qe / camera_e_per_adu
    return {
        "emitter_density_um2": density_value,
        "target_active_emitters_per_frame": target_active,
        "background_range_photons": list(background),
        "photon_mean": None if photon_mean is None else float(photon_mean),
        "photon_sigma": None if photon_sigma is None else float(photon_sigma),
        "photon_range": list(photon_range),
        "camera_baseline_adu": camera_baseline,
        "camera_qe": camera_qe,
        "camera_e_per_adu": camera_e_per_adu,
        "train_background_photons": train_background_photons,
        "train_background_adu": train_background_adu,
        "recenter_mode": FORMAL_RECENTER_MODE,
    }


def audit_formal_runtime_contracts(
    modality: PSFModality | str,
    runtime_configs: Mapping[str, Mapping[str, Any]],
    *,
    require_complete: bool = True,
) -> dict[str, dict[str, Any]]:
    parsed_modality = PSFModality.parse(modality)
    allowed_channels = {
        PSFModality.DOUBLE_HELIX: {"main"},
        PSFModality.EMITTER_2D: {"left", "right"},
        PSFModality.ASTIGMATISM: {"left", "right"},
    }[parsed_modality]
    selected_channels = set(runtime_configs)
    if not selected_channels or not selected_channels.issubset(allowed_channels):
        channel_label = "main" if parsed_modality is PSFModality.DOUBLE_HELIX else "left and/or right"
        raise ValueError(f"formal {parsed_modality.value} training requires {channel_label} channels")

    physical_hashes: dict[str, str] = {}
    evidence: dict[str, dict[str, Any]] = {}
    for channel_id, runtime_config in runtime_configs.items():
        provider = runtime_config.get("batch_provider", {})
        provider_params = provider.get("params", {}) if isinstance(provider, Mapping) else {}
        if parsed_modality is PSFModality.DOUBLE_HELIX:
            if provider.get("name") != "online_train_batch":
                raise ValueError("formal double_helix training requires the online GMM provider")
            direct_ppg = (
                provider_params.get("simulation_backend") == "ppg_direct"
                and provider_params.get("psf_type") == "ppg_direct"
            )
            vector_cuda_gemm = (
                provider_params.get("simulation_backend") == "cuda_gemm"
                and provider_params.get("psf_type") == "vector"
            )
            if not (direct_ppg or vector_cuda_gemm):
                raise ValueError(
                    "formal double_helix training requires vector/CUDA-GEMM or PPG direct simulation"
                )
            if int(provider_params.get("vector_psf_size", -1)) != 21:
                raise ValueError("formal double_helix training requires vector_psf_size=21")
            if vector_cuda_gemm and provider_params.get("pupil_carrier_complex") is None:
                raise ValueError("formal double_helix training requires the calibrated pupil carrier")
            loss = runtime_config.get("loss", {})
            loss_params = loss.get("params", {}) if isinstance(loss, Mapping) else {}
            if not isinstance(loss, Mapping) or loss.get("name") != "active_smlm_gmm_loss":
                raise ValueError("formal double_helix training requires GMM loss")
            target_order = loss_params.get("target_order") if isinstance(loss_params, Mapping) else None
            if target_order not in {"v03", "legacy_iwae"}:
                raise ValueError("formal double_helix GMM target order must be v03 or legacy_iwae")
            if target_order == "legacy_iwae" and (
                loss_params.get("gmm_backend") != "mixture_same_family"
                or int(loss_params.get("gmm_target_chunk", -1)) != 4
                or int(loss_params.get("gmm_component_chunk", -1)) != 64
            ):
                raise ValueError(
                    "formal double_helix legacy GMM contract requires mixture_same_family with chunks 4/64"
                )
            evidence[channel_id] = {
                "provider": str(provider["name"]),
                "target_source": "ppg_gmm" if direct_ppg else "physical_update_vector_cuda_gemm_gmm",
                "simulation_backend": str(provider_params.get("simulation_backend")),
                "psf_type": str(provider_params.get("psf_type")),
                "loss": "active_smlm_gmm_loss",
                "target_order": target_order,
            }
            continue

        optimizer = runtime_config.get("optimizer", {})
        if str(optimizer.get("name", "")).lower() != "adamw":
            raise ValueError(f"formal {parsed_modality.value}:{channel_id} training requires AdamW")
        optimizer_params = optimizer.get("params", {})
        if not isinstance(optimizer_params, Mapping) or float(optimizer_params.get("weight_decay", -1.0)) != 0.1:
            raise ValueError(f"formal {parsed_modality.value}:{channel_id} AdamW requires weight_decay=0.1")
        if float(optimizer_params.get("lr", -1.0)) != 0.0006:
            raise ValueError(f"formal {parsed_modality.value}:{channel_id} AdamW requires lr=0.0006")

        loss = runtime_config.get("loss", {})
        loss_params = loss.get("params", {}) if isinstance(loss, Mapping) else {}
        if not isinstance(loss, Mapping) or loss.get("name") != "active_smlm_gmm_loss":
            raise ValueError(f"formal {parsed_modality.value}:{channel_id} training requires GMM loss")
        if not isinstance(loss_params, Mapping) or loss_params.get("target_order") != "legacy_iwae":
            raise ValueError(f"formal {parsed_modality.value}:{channel_id} GMM target order must be legacy_iwae")
        if loss_params.get("gmm_backend") != "mixture_same_family":
            raise ValueError(f"formal {parsed_modality.value}:{channel_id} GMM backend must be mixture_same_family")
        if int(loss_params.get("gmm_target_chunk", -1)) != 4 or int(loss_params.get("gmm_component_chunk", -1)) != 64:
            raise ValueError(f"formal {parsed_modality.value}:{channel_id} GMM chunks must be 4/64")

        training = runtime_config.get("resolved_contract", {}).get("training_runtime", {})
        amp = training.get("amp", {}) if isinstance(training, Mapping) else {}
        grad_clip = training.get("grad_clip", {}) if isinstance(training, Mapping) else {}
        scheduler = training.get("scheduler", {}) if isinstance(training, Mapping) else {}
        if not isinstance(amp, Mapping) or amp.get("active") is not True or amp.get("dtype") != "float16":
            raise ValueError(f"formal {parsed_modality.value}:{channel_id} training requires float16 AMP")
        if not isinstance(grad_clip, Mapping) or float(grad_clip.get("configured_norm", -1.0)) != 0.03:
            raise ValueError(f"formal {parsed_modality.value}:{channel_id} training requires grad_clip_norm=0.03")
        scheduler_params = scheduler.get("params", {}) if isinstance(scheduler, Mapping) else {}
        if (
            not isinstance(scheduler, Mapping)
            or scheduler.get("name") != "StepLR"
            or scheduler.get("step_unit") != "epoch"
            or not isinstance(scheduler_params, Mapping)
            or int(scheduler_params.get("step_size", -1)) != 10
            or float(scheduler_params.get("gamma", -1.0)) != 0.9
        ):
            raise ValueError(
                f"formal {parsed_modality.value}:{channel_id} requires StepLR(step_size=10, gamma=0.9) per epoch"
            )

        evidence[channel_id] = {
            "optimizer": "AdamW",
            "learning_rate": 0.0006,
            "weight_decay": 0.1,
            "scheduler": "StepLR",
            "scheduler_step_size": 10,
            "scheduler_gamma": 0.9,
            "scheduler_step_unit": "epoch",
            "loss": "active_smlm_gmm_loss",
            "target_order": "legacy_iwae",
            "gmm_backend": "mixture_same_family",
            "gmm_target_chunk": 4,
            "gmm_component_chunk": 64,
            "amp_dtype": "float16",
            "grad_clip_norm": 0.03,
        }
        if not isinstance(provider_params, Mapping):
            raise ValueError(f"formal {parsed_modality.value}:{channel_id} provider params are invalid")
        sampling = sampling_recenter_metadata(provider_params)
        if sampling["emitter_density_um2"] != FORMAL_DENSITY_UM2:
            raise ValueError(f"formal {parsed_modality.value}:{channel_id} requires emitter density=0.5 emitters/um2")
        if tuple(sampling["background_range_photons"]) != (FORMAL_BACKGROUND_PHOTONS, FORMAL_BACKGROUND_PHOTONS):
            raise ValueError(f"formal {parsed_modality.value}:{channel_id} requires fixed background=110 photons/pixel")
        if (
            sampling["photon_mean"] != FORMAL_PHOTON_MEAN
            or sampling["photon_sigma"] != FORMAL_PHOTON_SIGMA
            or tuple(sampling["photon_range"]) != FORMAL_PHOTON_RANGE
        ):
            raise ValueError(
                f"formal {parsed_modality.value}:{channel_id} requires photon mean/sigma=20000/1000 and range=0..31000"
            )
        evidence[channel_id].update(sampling)

        if parsed_modality in {PSFModality.EMITTER_2D, PSFModality.ASTIGMATISM}:
            if provider_params.get("psf_type") != "vector" or provider_params.get("simulation_backend") != "cuda_gemm":
                raise ValueError(f"formal {parsed_modality.value}:{channel_id} requires vector CUDA-GEMM simulation")
            entries = provider_params.get("dual_domain_coeff_maps", ())
            initialization = runtime_config.get("physical_initialization", "coefficient_map")
            if initialization == "anchor_only":
                expected_profile = "circular_2d_50nm" if parsed_modality is PSFModality.EMITTER_2D else "astigmatism_660nm"
                if entries or provider_params.get("conditioning_profile") != expected_profile:
                    raise ValueError(
                        f"formal {parsed_modality.value}:{channel_id} anchor_only requires empty coefficient maps and conditioning_profile={expected_profile}"
                    )
                if parsed_modality is PSFModality.EMITTER_2D and float(provider_params.get("circular_psf_radius_nm", 0.0)) != 50.0:
                    raise ValueError(f"formal emitter_2d:{channel_id} anchor_only requires circular_psf_radius_nm=50.0")
                evidence[channel_id]["physical_initialization"] = "anchor_only"
            elif initialization == "coefficient_map":
                if not isinstance(entries, (list, tuple)) or len(entries) != 1:
                    raise ValueError(f"formal {parsed_modality.value}:{channel_id} requires one coefficient map")
                entry = entries[0]
                path_value = entry.get("coeff_maps_npz") if isinstance(entry, Mapping) else None
                if path_value is None:
                    raise ValueError(f"formal {parsed_modality.value}:{channel_id} coefficient map path is missing")
                path = Path(str(path_value)).resolve()
                if not path.is_file():
                    raise ValueError(f"formal {parsed_modality.value}:{channel_id} coefficient map does not exist: {path}")
                physical_hashes[channel_id] = sha256_file(path)
                evidence[channel_id].update(
                    {
                        "physical_initialization": "coefficient_map",
                        "coefficient_map": str(path),
                        "coefficient_map_sha256": physical_hashes[channel_id],
                    }
                )
            else:
                raise ValueError(f"unknown physical_initialization: {initialization}")
            evidence[channel_id].update(
                {
                    "psf_type": "vector",
                    "simulation_backend": "cuda_gemm",
                    "z_range_um": list(provider_params.get("z_range", ())),
                }
            )
        if parsed_modality is PSFModality.EMITTER_2D:
            if tuple(float(value) for value in provider_params.get("z_range", ())) != (-0.1, 0.1):
                raise ValueError(f"formal emitter_2d:{channel_id} requires nuisance z range -0.1..0.1 um")
            model_params = runtime_config.get("model", {}).get("params", {})
            disabled = model_params.get("disabled_attr", ()) if isinstance(model_params, Mapping) else ()
            if 3 not in tuple(int(value) for value in disabled) or int(loss_params.get("disable_attr", -1)) != 3:
                raise ValueError(f"formal emitter_2d:{channel_id} must disable z head and z loss")

    if require_complete and len(physical_hashes) > 1 and parsed_modality in {
        PSFModality.EMITTER_2D,
        PSFModality.ASTIGMATISM,
    } and len(set(physical_hashes.values())) != len(physical_hashes):
        raise ValueError(f"formal {parsed_modality.value} left/right coefficient maps must be physically distinct")
    return evidence


_channel_metadata = channel_metadata
_physical_provenance = physical_provenance
_sampling_recenter_metadata = sampling_recenter_metadata
_audit_formal_runtime_contracts = audit_formal_runtime_contracts
