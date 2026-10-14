from __future__ import annotations

from typing import Any, Mapping

from unity_psf.roi_library import ROIBank, ROIRecord, load_roi_bank


def resolve_roi_gamma_banks(
    bank: ROIBank,
    gamma_cfg: Mapping[str, Any],
) -> tuple[ROIBank, ROIBank | None, str, str | None]:
    if "heldout_roi_library_path" in gamma_cfg:
        return bank, load_roi_bank(str(gamma_cfg["heldout_roi_library_path"])), "fixed_samples", "configured_hdf5"
    heldout_count = auto_heldout_count(gamma_cfg, total_count=len(bank.records))
    if heldout_count <= 0:
        if auto_heldout_requested(gamma_cfg):
            return bank, None, "disabled_insufficient_records", "auto_from_roi_library_insufficient_records"
        return bank, None, "not_configured", None
    records_by_domain: dict[str, list[ROIRecord]] = {}
    for record in bank.records:
        records_by_domain.setdefault(str(record.domain_name), []).append(record)
    selected_records: list[ROIRecord] = []
    heldout_records: list[ROIRecord] = []
    if len(records_by_domain) <= 1:
        split_at = len(bank.records) - heldout_count
        selected_records = list(bank.records[:split_at])
        heldout_records = list(bank.records[split_at:])
    else:
        for domain_records in records_by_domain.values():
            domain_heldout_count = auto_heldout_count(gamma_cfg, total_count=len(domain_records))
            if domain_heldout_count <= 0:
                selected_records.extend(domain_records)
                continue
            split_at = len(domain_records) - domain_heldout_count
            selected_records.extend(domain_records[:split_at])
            heldout_records.extend(domain_records[split_at:])
    if not selected_records:
        raise ValueError("auto held-out split must leave at least one selected ROI")
    if not heldout_records:
        return bank, None, "disabled_insufficient_records", "auto_from_roi_library_insufficient_records"
    selected_bank = ROIBank(
        records=tuple(selected_records),
        config=bank.config,
        metadata={**bank.metadata, "heldout_split": "selected"},
        empty_grid_cell_ids=bank.empty_grid_cell_ids,
        format_version=bank.format_version,
    )
    heldout_bank = ROIBank(
        records=tuple(heldout_records),
        config=bank.config,
        metadata={**bank.metadata, "heldout_split": "heldout"},
        empty_grid_cell_ids=bank.empty_grid_cell_ids,
        format_version=bank.format_version,
    )
    return selected_bank, heldout_bank, "auto_split_fixed_samples", "auto_from_roi_library"


def auto_heldout_requested(gamma_cfg: Mapping[str, Any]) -> bool:
    return int(gamma_cfg.get("auto_heldout_min_rois", 0)) > 0 or int(gamma_cfg.get("auto_heldout_max_rois", 0)) > 0


def auto_heldout_count(gamma_cfg: Mapping[str, Any], *, total_count: int) -> int:
    min_rois = int(gamma_cfg.get("auto_heldout_min_rois", 0))
    max_rois = int(gamma_cfg.get("auto_heldout_max_rois", min_rois))
    if min_rois <= 0 and max_rois <= 0:
        return 0
    return min(max(min_rois, 0), max(max_rois, 0), max(int(total_count) - 1, 0))
