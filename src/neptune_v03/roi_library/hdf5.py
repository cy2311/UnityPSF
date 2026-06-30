from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import h5py
import numpy as np

from .types import FORMAT_VERSION, EmitterPosterior, ROIBank, ROIRecord


COMPRESSION = "gzip"
STRING_DTYPE = h5py.string_dtype(encoding="utf-8")


def save_roi_bank(bank: ROIBank, path: str | Path) -> None:
    with H5ROIBankWriter(
        path,
        config=bank.config,
        metadata=bank.metadata,
        empty_grid_cell_ids=bank.empty_grid_cell_ids,
        format_version=bank.format_version,
    ) as writer:
        for record in bank.records:
            writer.append(record)


def load_roi_bank(path: str | Path, roi_indices: Sequence[int] | None = None) -> ROIBank:
    with h5py.File(path, "r") as handle:
        config = _read_json_attr(handle, "config_json")
        metadata = _read_json_attr(handle, "metadata_json")
        empty_cells = tuple(int(v) for v in handle["empty_grid_cell_ids"][...].tolist())
        format_version = int(handle.attrs["format_version"])
        records = tuple(_read_records(handle, roi_indices=roi_indices))
    return ROIBank(
        records=records,
        config=config,
        metadata=metadata,
        empty_grid_cell_ids=empty_cells,
        format_version=format_version,
    )


class H5ROIBankWriter:
    def __init__(
        self,
        path: str | Path,
        *,
        config: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        empty_grid_cell_ids: Sequence[int] = (),
        format_version: int = FORMAT_VERSION,
        compression: str = COMPRESSION,
    ) -> None:
        self.path = Path(path)
        self.config = dict(config or {})
        self.metadata = dict(metadata or {})
        self.empty_grid_cell_ids = tuple(int(v) for v in empty_grid_cell_ids)
        self.format_version = int(format_version)
        self.compression = compression
        self._handle: h5py.File | None = None
        self._rois: h5py.Group | None = None
        self._emitters: h5py.Group | None = None
        self._raw_shape: tuple[int, ...] | None = None
        self._bg_shape: tuple[int, ...] | None = None
        self._record_count = 0
        self._emitter_count = 0

    def __enter__(self) -> H5ROIBankWriter:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = h5py.File(self.path, "w")
        self._handle.attrs["format_version"] = self.format_version
        self._handle.attrs["config_json"] = json.dumps(self.config, sort_keys=True)
        self._handle.attrs["metadata_json"] = json.dumps(self.metadata, sort_keys=True)
        self._handle.create_dataset("empty_grid_cell_ids", data=np.asarray(self.empty_grid_cell_ids, dtype=np.int32))
        self._rois = self._handle.create_group("rois")
        self._emitters = self._handle.create_group("emitters")
        self._create_scalar_roi_datasets()
        self._create_emitter_datasets()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def append(self, record: ROIRecord) -> None:
        if self._rois is None or self._emitters is None:
            raise RuntimeError("H5ROIBankWriter must be used as a context manager")
        raw = _as_float32(record.raw_frames_photon)
        bg_mu = _as_float32(record.background_mu)
        bg_smoothed = _as_float32(record.background_smoothed)
        self._ensure_roi_array_datasets(raw_shape=raw.shape, bg_shape=bg_mu.shape)
        if bg_smoothed.shape != self._bg_shape:
            raise ValueError(f"background_smoothed shape {bg_smoothed.shape} does not match {self._bg_shape}")

        roi_index = self._record_count
        emitter_start = self._emitter_count
        emitters = tuple(record.emitters)

        _append(self._rois["roi_id"], np.asarray([record.roi_id], dtype=np.int64))
        _append_string(self._rois["domain_name"], [record.domain_name])
        _append(self._rois["frame_window"], np.asarray([record.frame_window], dtype=np.int64))
        _append(self._rois["roi_origin_xy_px"], np.asarray([record.roi_origin_xy_px], dtype=np.float32))
        _append(self._rois["grid_cell_id"], np.asarray([record.grid_cell_id], dtype=np.int32))
        _append_string(self._rois["summary_json"], [json.dumps(record.summary, sort_keys=True)])
        _append(self._rois["emitter_start"], np.asarray([emitter_start], dtype=np.int64))
        _append(self._rois["emitter_count"], np.asarray([len(emitters)], dtype=np.int64))
        _append(self._rois["raw_frames_photon"], raw)
        _append(self._rois["background_mu"], bg_mu)
        _append(self._rois["background_smoothed"], bg_smoothed)
        self._append_emitters(emitters)

        self._record_count = roi_index + 1

    def _create_scalar_roi_datasets(self) -> None:
        assert self._rois is not None
        _create_resizable_dataset(self._rois, "roi_id", shape_tail=(), dtype=np.int64, compression=None)
        _create_resizable_dataset(self._rois, "domain_name", shape_tail=(), dtype=STRING_DTYPE, compression=None)
        _create_resizable_dataset(self._rois, "frame_window", shape_tail=(2,), dtype=np.int64, compression=None)
        _create_resizable_dataset(self._rois, "roi_origin_xy_px", shape_tail=(2,), dtype=np.float32, compression=None)
        _create_resizable_dataset(self._rois, "grid_cell_id", shape_tail=(), dtype=np.int32, compression=None)
        _create_resizable_dataset(self._rois, "summary_json", shape_tail=(), dtype=STRING_DTYPE, compression=None)
        _create_resizable_dataset(self._rois, "emitter_start", shape_tail=(), dtype=np.int64, compression=None)
        _create_resizable_dataset(self._rois, "emitter_count", shape_tail=(), dtype=np.int64, compression=None)

    def _create_emitter_datasets(self) -> None:
        assert self._emitters is not None
        _create_resizable_dataset(self._emitters, "probability", shape_tail=(), dtype=np.float32, compression=self.compression)
        _create_resizable_dataset(self._emitters, "cell_xy_px", shape_tail=(2,), dtype=np.float32, compression=self.compression)
        _create_resizable_dataset(self._emitters, "mu_xy_px", shape_tail=(2,), dtype=np.float32, compression=self.compression)
        _create_resizable_dataset(self._emitters, "sigma_xy_px", shape_tail=(2,), dtype=np.float32, compression=self.compression)
        _create_resizable_dataset(self._emitters, "mu_z_nm", shape_tail=(), dtype=np.float32, compression=self.compression)
        _create_resizable_dataset(self._emitters, "sigma_z_nm", shape_tail=(), dtype=np.float32, compression=self.compression)
        _create_resizable_dataset(self._emitters, "mu_photons", shape_tail=(), dtype=np.float32, compression=self.compression)
        _create_resizable_dataset(self._emitters, "sigma_photons", shape_tail=(), dtype=np.float32, compression=self.compression)
        _create_resizable_dataset(self._emitters, "local_xy_px", shape_tail=(2,), dtype=np.float32, compression=self.compression)
        _create_resizable_dataset(self._emitters, "full_xy_px", shape_tail=(2,), dtype=np.float32, compression=self.compression)
        _create_resizable_dataset(self._emitters, "frame_index", shape_tail=(), dtype=np.int64, compression=self.compression)

    def _ensure_roi_array_datasets(self, *, raw_shape: tuple[int, ...], bg_shape: tuple[int, ...]) -> None:
        assert self._rois is not None
        if self._raw_shape is None:
            self._raw_shape = tuple(raw_shape)
            self._bg_shape = tuple(bg_shape)
            _create_resizable_dataset(
                self._rois,
                "raw_frames_photon",
                shape_tail=self._raw_shape,
                dtype=np.float32,
                compression=self.compression,
            )
            _create_resizable_dataset(
                self._rois,
                "background_mu",
                shape_tail=self._bg_shape,
                dtype=np.float32,
                compression=self.compression,
            )
            _create_resizable_dataset(
                self._rois,
                "background_smoothed",
                shape_tail=self._bg_shape,
                dtype=np.float32,
                compression=self.compression,
            )
        if tuple(raw_shape) != self._raw_shape:
            raise ValueError(f"raw_frames_photon shape {raw_shape} does not match {self._raw_shape}")
        if tuple(bg_shape) != self._bg_shape:
            raise ValueError(f"background_mu shape {bg_shape} does not match {self._bg_shape}")

    def _append_emitters(self, emitters: tuple[EmitterPosterior, ...]) -> None:
        assert self._emitters is not None
        count = len(emitters)
        if count == 0:
            return
        _append(self._emitters["probability"], np.asarray([e.probability for e in emitters], dtype=np.float32))
        _append(self._emitters["cell_xy_px"], _array2([e.cell_xy_px for e in emitters], count))
        _append(self._emitters["mu_xy_px"], _array2([e.mu_xy_px for e in emitters], count))
        _append(self._emitters["sigma_xy_px"], _array2([e.sigma_xy_px for e in emitters], count))
        _append(self._emitters["mu_z_nm"], np.asarray([e.mu_z_nm for e in emitters], dtype=np.float32))
        _append(self._emitters["sigma_z_nm"], np.asarray([e.sigma_z_nm for e in emitters], dtype=np.float32))
        _append(self._emitters["mu_photons"], np.asarray([e.mu_photons for e in emitters], dtype=np.float32))
        _append(self._emitters["sigma_photons"], np.asarray([e.sigma_photons for e in emitters], dtype=np.float32))
        _append(self._emitters["local_xy_px"], _array2([e.local_xy_px for e in emitters], count))
        _append(self._emitters["full_xy_px"], _array2([e.full_xy_px for e in emitters], count))
        _append(self._emitters["frame_index"], np.asarray([e.frame_index for e in emitters], dtype=np.int64))
        self._emitter_count += count


def _read_records(handle: h5py.File, *, roi_indices: Sequence[int] | None) -> Iterable[ROIRecord]:
    rois = handle["rois"]
    indices = range(rois["roi_id"].shape[0]) if roi_indices is None else [int(v) for v in roi_indices]
    for roi_index in indices:
        start = int(rois["emitter_start"][roi_index])
        count = int(rois["emitter_count"][roi_index])
        stop = start + count
        yield ROIRecord(
            roi_id=int(rois["roi_id"][roi_index]),
            domain_name=_decode_string(rois["domain_name"][roi_index]),
            frame_window=tuple(int(v) for v in rois["frame_window"][roi_index].tolist()),
            roi_origin_xy_px=tuple(float(v) for v in rois["roi_origin_xy_px"][roi_index].tolist()),
            raw_frames_photon=np.asarray(rois["raw_frames_photon"][roi_index], dtype=np.float32),
            background_mu=np.asarray(rois["background_mu"][roi_index], dtype=np.float32),
            background_smoothed=np.asarray(rois["background_smoothed"][roi_index], dtype=np.float32),
            grid_cell_id=int(rois["grid_cell_id"][roi_index]),
            emitters=tuple(_read_emitters(handle["emitters"], start, stop)),
            summary=json.loads(_decode_string(rois["summary_json"][roi_index])),
        )


def _read_emitters(group: h5py.Group, start: int, stop: int) -> Iterable[EmitterPosterior]:
    for index in range(start, stop):
        yield EmitterPosterior(
            probability=float(group["probability"][index]),
            cell_xy_px=_read_pair(group["cell_xy_px"], index),
            mu_xy_px=_read_pair(group["mu_xy_px"], index),
            sigma_xy_px=_read_pair(group["sigma_xy_px"], index),
            mu_z_nm=float(group["mu_z_nm"][index]),
            sigma_z_nm=float(group["sigma_z_nm"][index]),
            mu_photons=float(group["mu_photons"][index]),
            sigma_photons=float(group["sigma_photons"][index]),
            local_xy_px=_read_pair(group["local_xy_px"], index),
            full_xy_px=_read_pair(group["full_xy_px"], index),
            frame_index=int(group["frame_index"][index]),
        )


def _create_resizable_dataset(
    group: h5py.Group,
    name: str,
    *,
    shape_tail: tuple[int, ...],
    dtype: object,
    compression: str | None,
) -> h5py.Dataset:
    shape = (0, *shape_tail)
    maxshape = (None, *shape_tail)
    kwargs: dict[str, Any] = {
        "shape": shape,
        "maxshape": maxshape,
        "dtype": dtype,
        "chunks": _chunks(shape),
    }
    if compression is not None:
        kwargs["compression"] = compression
    return group.create_dataset(name, **kwargs)


def _append(dataset: h5py.Dataset, values: np.ndarray) -> None:
    array = np.asarray(values, dtype=dataset.dtype)
    if array.ndim == dataset.ndim - 1:
        array = array[None, ...]
    start = int(dataset.shape[0])
    count = int(array.shape[0])
    dataset.resize(start + count, axis=0)
    if count:
        dataset[start : start + count] = array


def _append_string(dataset: h5py.Dataset, values: Sequence[str]) -> None:
    start = int(dataset.shape[0])
    dataset.resize(start + len(values), axis=0)
    dataset[start : start + len(values)] = np.asarray(values, dtype=STRING_DTYPE)


def _chunks(shape: tuple[int, ...]) -> tuple[int, ...]:
    if len(shape) <= 1:
        return (max(1, min(int(shape[0]), 1024)),)
    return (1, *tuple(max(1, int(value)) for value in shape[1:]))


def _array2(values: Sequence[tuple[float, float]], count: int) -> np.ndarray:
    if count == 0:
        return np.empty((0, 2), dtype=np.float32)
    return np.asarray(values, dtype=np.float32).reshape(count, 2)


def _as_float32(value: np.ndarray) -> np.ndarray:
    return np.asarray(value, dtype=np.float32)


def _read_pair(dataset: h5py.Dataset, index: int) -> tuple[float, float]:
    values = dataset[index].tolist()
    return (float(values[0]), float(values[1]))


def _decode_string(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _read_json_attr(handle: h5py.File, name: str) -> dict[str, Any]:
    return json.loads(_decode_string(handle.attrs.get(name, "{}")))
