from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json
import shutil


@dataclass(frozen=True)
class StandardZMapPaths:
    root_dir: Path
    alternating_coeff_maps_npz: Path
    provisional_scalar_zmap_npz: Path
    diagnostics_summary_json: Path
    export_summary_json: Path
    metadata_json: Path


def _standard_paths(root_dir: Path) -> StandardZMapPaths:
    return StandardZMapPaths(
        root_dir=root_dir,
        alternating_coeff_maps_npz=root_dir / "alternating_full_roi_zernike_maps_nm.npz",
        provisional_scalar_zmap_npz=root_dir / "provisional_non_astig_rms_nm.npz",
        diagnostics_summary_json=root_dir / "real_nat_diagnostics_summary.json",
        export_summary_json=root_dir / "export_nat_zmap_summary.json",
        metadata_json=root_dir / "standard_metadata.json",
    )


def _copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def _load_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def promote_standard_zmap(
    *,
    side: str,
    diagnostics_dir: str | Path,
    export_dir: str | Path,
    target_root: str | Path,
    harvest_dir: str | Path | None = None,
    notes: str | None = None,
    method: str = "neptune_zmap_main_local_vector_nat",
    version_tag: str = "microtube_spatial_balanced_v2",
) -> StandardZMapPaths:
    diagnostics_dir = Path(diagnostics_dir)
    export_dir = Path(export_dir)
    target_root = Path(target_root)
    target_root.mkdir(parents=True, exist_ok=True)
    paths = _standard_paths(target_root)

    _copy_if_exists(export_dir / "alternating_full_roi_zernike_maps_nm.npz", paths.alternating_coeff_maps_npz)
    _copy_if_exists(export_dir / "provisional_non_astig_rms_nm.npz", paths.provisional_scalar_zmap_npz)
    _copy_if_exists(diagnostics_dir / "real_nat_diagnostics_summary.json", paths.diagnostics_summary_json)
    _copy_if_exists(export_dir / "export_nat_zmap_summary.json", paths.export_summary_json)

    diag_summary = _load_json_if_exists(paths.diagnostics_summary_json)
    export_summary = _load_json_if_exists(paths.export_summary_json)
    metadata = {
        "side": str(side),
        "method": str(method),
        "target_root": str(target_root),
        "harvest_dir": None if harvest_dir is None else str(Path(harvest_dir)),
        "diagnostics_dir": str(diagnostics_dir),
        "export_dir": str(export_dir),
        "artifacts": {
            "alternating_coeff_maps_npz": str(paths.alternating_coeff_maps_npz),
            "provisional_scalar_zmap_npz": str(paths.provisional_scalar_zmap_npz),
            "diagnostics_summary_json": str(paths.diagnostics_summary_json),
            "export_summary_json": str(paths.export_summary_json),
        },
        "diagnostics_metrics": diag_summary.get("alternating_metrics", {}),
        "comparison_metrics": diag_summary.get("comparison_metrics", {}),
        "export_config": export_summary.get("config", {}),
        "notes": notes or "",
        "version_tag": str(version_tag),
    }
    paths.metadata_json.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return paths
