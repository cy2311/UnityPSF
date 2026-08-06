from __future__ import annotations

import torch

from unity_psf.localization.conditioning import FullResZernikeConditioning
from unity_psf.localization.training_adapter import LocalizationTrainBatch, to_training_batch
from unity_psf.roi_library import ROIBank, ROIRecord
from unity_psf.training.loop import TrainingBatch


def build_roi_batch_provider(
    bank: ROIBank,
    *,
    batch_size: int,
    seed: int = 0,
    condition_providers_by_domain: dict[str, FullResZernikeConditioning] | None = None,
    append_domain_onehot: bool = False,
    domain_names: tuple[str, ...] | list[str] | None = None,
):
    records = tuple(bank.records)
    domain_order = tuple(str(name) for name in (domain_names or _bank_domain_order(records)))

    def provider(epoch: int) -> list[TrainingBatch]:
        if int(batch_size) <= 0:
            raise ValueError("batch_size must be positive")
        batches: list[TrainingBatch] = []
        for start in range(0, len(records), int(batch_size)):
            batch_records = records[start : start + int(batch_size)]
            batches.append(
                to_training_batch(
                    _records_to_localization_batch(
                        batch_records,
                        epoch=int(epoch),
                        seed=int(seed),
                        condition_providers_by_domain=condition_providers_by_domain,
                        append_domain_onehot=append_domain_onehot,
                        domain_names=domain_order,
                    )
                )
            )
        return batches

    return provider


def _records_to_localization_batch(
    records: tuple[ROIRecord, ...],
    *,
    epoch: int,
    seed: int,
    condition_providers_by_domain: dict[str, FullResZernikeConditioning] | None = None,
    append_domain_onehot: bool = False,
    domain_names: tuple[str, ...] = (),
) -> LocalizationTrainBatch:
    if not records:
        raise ValueError("ROI batch requires at least one record")
    model_input = torch.stack([torch.as_tensor(record.raw_frames_photon, dtype=torch.float32) for record in records], dim=0)
    bkg = torch.stack([torch.as_tensor(record.background_smoothed, dtype=torch.float32) for record in records], dim=0)
    height, width = int(model_input.shape[-2]), int(model_input.shape[-1])
    detect = torch.zeros((len(records), height, width), dtype=torch.float32)
    pxyz = torch.zeros((len(records), 1, 4), dtype=torch.float32)
    mask = torch.zeros((len(records), 1), dtype=torch.bool)
    for idx, record in enumerate(records):
        center = _record_center(record)
        x, y = center
        col = max(0, min(width - 1, int(x + 0.5)))
        row = max(0, min(height - 1, int(y + 0.5)))
        detect[idx, row, col] = 1.0
        pxyz[idx, 0] = torch.tensor([float(x), float(y), 0.0, float(model_input[idx].max().item())])
        mask[idx, 0] = True
    model_input_payload = model_input
    condition = None
    if condition_providers_by_domain:
        condition = _condition_batch_for_records(
            records,
            model_input=model_input,
            providers=condition_providers_by_domain,
            append_domain_onehot=append_domain_onehot,
            domain_names=domain_names,
        )
        model_input_payload = (model_input, condition)
    return LocalizationTrainBatch(
        model_input=model_input_payload,
        detect_tar=detect,
        bkg_tar=bkg,
        pxyz_tar=pxyz,
        mask_tar=mask,
        metadata={
            "epoch": int(epoch),
            "seed": int(seed),
            "source": "roi_bank",
            "roi_ids": [int(record.roi_id) for record in records],
            "roi_origin_xy_px": [tuple(float(v) for v in record.roi_origin_xy_px) for record in records],
            "domain_names": [str(record.domain_name) for record in records],
            **(
                {}
                if condition is None
                else {
                    "conditioning_mode": "film",
                    "condition_dim": int(condition.shape[1]),
                    "condition_feature_dim": int(condition.shape[1]) - (len(domain_names) if append_domain_onehot else 0),
                    "domain_count": len(domain_names),
                    "condition_domain_onehot_slice": (
                        int(condition.shape[1]) - len(domain_names),
                        int(condition.shape[1]),
                    )
                    if append_domain_onehot
                    else None,
                }
            ),
        },
    )


def _record_center(record: ROIRecord) -> tuple[float, float]:
    if record.emitters:
        return tuple(float(v) for v in record.emitters[0].local_xy_px)
    raw = torch.as_tensor(record.raw_frames_photon, dtype=torch.float32)
    projection = raw.sum(dim=0)
    flat_index = int(torch.argmax(projection).item())
    width = int(projection.shape[-1])
    return (float(flat_index % width), float(flat_index // width))


def _bank_domain_order(records: tuple[ROIRecord, ...]) -> tuple[str, ...]:
    names = []
    for record in records:
        name = str(record.domain_name)
        if name not in names:
            names.append(name)
    return tuple(names)


def _condition_batch_for_records(
    records: tuple[ROIRecord, ...],
    *,
    model_input: torch.Tensor,
    providers: dict[str, FullResZernikeConditioning],
    append_domain_onehot: bool,
    domain_names: tuple[str, ...],
) -> torch.Tensor:
    conditions = []
    height, width = int(model_input.shape[-2]), int(model_input.shape[-1])
    domain_to_index = {name: idx for idx, name in enumerate(domain_names)}
    for record in records:
        domain_name = str(record.domain_name)
        provider = providers[domain_name]
        x0, y0 = _condition_origin_for_record(record)
        condition = provider.condition_vector_from_xy(
            x0=x0,
            y0=y0,
            height=height,
            width=width,
            device=model_input.device,
            dtype=model_input.dtype,
        )
        if append_domain_onehot:
            onehot = torch.zeros((len(domain_names),), dtype=model_input.dtype, device=model_input.device)
            onehot[domain_to_index[domain_name]] = 1.0
            condition = torch.cat((condition, onehot), dim=0)
        conditions.append(condition)
    return torch.stack(conditions, dim=0).contiguous()


def _condition_origin_for_record(record: ROIRecord) -> tuple[int, int]:
    local = record.summary.get("domain_local_roi_origin_xy_px") if isinstance(record.summary, dict) else None
    if isinstance(local, (list, tuple)) and len(local) == 2:
        return int(round(float(local[0]))), int(round(float(local[1])))
    return int(round(float(record.roi_origin_xy_px[0]))), int(round(float(record.roi_origin_xy_px[1])))
