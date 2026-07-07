from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np


INT_COLUMNS = {"frame", "tile_index"}
STRING_COLUMNS = {"sample_tiff", "sample_file", "sample_name", "source_file"}
RENDER_COLUMNS = ("x_px", "y_px", "z", "prob", "x_sig", "y_sig")


def is_h5_path(path: str | Path) -> bool:
    return str(path).lower().endswith((".h5", ".hdf5"))


def default_predictions_path(infer_dir: str | Path, *, stem: str = "predictions_merged") -> Path:
    infer_dir = Path(infer_dir)
    h5_path = infer_dir / f"{stem}.h5"
    if h5_path.is_file():
        return h5_path
    return infer_dir / f"{stem}.csv"


def prediction_fieldnames(path: str | Path) -> list[str]:
    path = Path(path)
    if is_h5_path(path):
        with h5py.File(path, "r") as handle:
            if "columns_json" in handle.attrs:
                return list(json.loads(handle.attrs["columns_json"]))
            return list(handle["locs"].keys())
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or [])


def iter_prediction_rows(path: str | Path, *, chunk_size: int = 65536):
    path = Path(path)
    if is_h5_path(path):
        fieldnames = prediction_fieldnames(path)
        with h5py.File(path, "r") as handle:
            group = handle["locs"]
            if not fieldnames:
                return
            count = int(group[fieldnames[0]].shape[0]) if fieldnames[0] in group else 0
            for start in range(0, count, int(chunk_size)):
                stop = min(count, start + int(chunk_size))
                arrays = {key: np.asarray(group[key][start:stop]) for key in fieldnames if key in group}
                for ix in range(stop - start):
                    row = {}
                    for key in arrays:
                        value = arrays[key][ix]
                        if hasattr(value, "item"):
                            value = value.item()
                        if isinstance(value, bytes):
                            value = value.decode("utf-8")
                        row[key] = value
                    yield row
        return
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            yield dict(row)


def _dtype_for_column(column: str) -> np.dtype:
    if column in STRING_COLUMNS:
        return h5py.string_dtype(encoding="utf-8")
    if column in INT_COLUMNS:
        return np.dtype("int32")
    return np.dtype("float32")


def _missing_value_for_column(column: str) -> int | float | str:
    if column in STRING_COLUMNS:
        return ""
    if column in INT_COLUMNS:
        return -1
    return float("nan")


def _coerce_value(column: str, value: object) -> int | float | str:
    if value in {None, ""}:
        return _missing_value_for_column(column)
    if column in STRING_COLUMNS:
        return str(value)
    if column in INT_COLUMNS:
        return int(float(value))
    return float(value)


class H5PredictionWriter:
    def __init__(
        self,
        path: str | Path,
        *,
        fieldnames: Iterable[str],
        chunk_size: int = 65536,
        compression: str | None = "lzf",
    ) -> None:
        self.path = Path(path)
        self.fieldnames = list(fieldnames)
        self.chunk_size = max(int(chunk_size), 1)
        self.compression = compression
        self.handle: h5py.File | None = None
        self.group: h5py.Group | None = None
        self.count = 0

    def __enter__(self) -> "H5PredictionWriter":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = h5py.File(self.path, "w")
        self.handle.attrs["schema"] = "infer_recon_predictions_h5_v0.1"
        self.handle.attrs["columns_json"] = json.dumps(self.fieldnames)
        self.group = self.handle.create_group("locs")
        for column in self.fieldnames:
            self.group.create_dataset(
                column,
                shape=(0,),
                maxshape=(None,),
                chunks=(self.chunk_size,),
                dtype=_dtype_for_column(column),
                compression=self.compression,
                shuffle=self.compression is not None,
            )
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.handle is not None:
            self.handle.attrs["count"] = int(self.count)
            self.handle.close()
        self.handle = None
        self.group = None

    def append_rows(self, rows: Iterable[dict[str, object]]) -> None:
        if self.group is None:
            raise RuntimeError("H5PredictionWriter must be used as a context manager")
        buffered = list(rows)
        if not buffered:
            return
        start = int(self.count)
        stop = start + len(buffered)
        for column in self.fieldnames:
            dataset = self.group[column]
            dataset.resize((stop,))
            values = np.asarray([_coerce_value(column, row.get(column)) for row in buffered], dtype=dataset.dtype)
            dataset[start:stop] = values
        self.count = stop


def _read_h5_render_arrays(path: Path, prob_threshold: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    with h5py.File(path, "r") as handle:
        group = handle["locs"]
        missing = [key for key in ("x_px", "y_px") if key not in group]
        if missing:
            raise KeyError(f"predictions h5 is missing required render columns: {missing}")
        prob = np.asarray(group["prob"][:] if "prob" in group else np.ones(group["x_px"].shape, dtype=np.float32), dtype=np.float32)
        keep = prob >= float(prob_threshold)
        x = np.asarray(group["x_px"][:], dtype=np.float32)[keep]
        y = np.asarray(group["y_px"][:], dtype=np.float32)[keep]
        z = np.asarray(group["z"][:] if "z" in group else np.zeros(group["x_px"].shape, dtype=np.float32), dtype=np.float32)[keep]
        prob = prob[keep]
        x_sig = np.asarray(group["x_sig"][:], dtype=np.float32)[keep] if "x_sig" in group else None
        y_sig = np.asarray(group["y_sig"][:], dtype=np.float32)[keep] if "y_sig" in group else None
    return x, y, z, prob, x_sig, y_sig


def _read_csv_render_arrays(path: Path, prob_threshold: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    xs: list[float] = []
    ys: list[float] = []
    zs: list[float] = []
    probs: list[float] = []
    x_sigs: list[float] = []
    y_sigs: list[float] = []
    if path.stat().st_size == 0:
        empty = np.asarray([], dtype=np.float32)
        return empty, empty, empty, empty, None, None
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        has_x_sig = "x_sig" in (reader.fieldnames or [])
        has_y_sig = "y_sig" in (reader.fieldnames or [])
        for row in reader:
            prob = float(row.get("prob", 1.0) or 1.0)
            if prob < prob_threshold:
                continue
            xs.append(float(row["x_px"]))
            ys.append(float(row["y_px"]))
            zs.append(float(row.get("z", 0.0) or 0.0))
            probs.append(prob)
            if has_x_sig:
                value = row.get("x_sig", "")
                x_sigs.append(float(value) if value not in {"", None} else float("nan"))
            if has_y_sig:
                value = row.get("y_sig", "")
                y_sigs.append(float(value) if value not in {"", None} else float("nan"))
    return (
        np.asarray(xs, dtype=np.float32),
        np.asarray(ys, dtype=np.float32),
        np.asarray(zs, dtype=np.float32),
        np.asarray(probs, dtype=np.float32),
        np.asarray(x_sigs, dtype=np.float32) if has_x_sig else None,
        np.asarray(y_sigs, dtype=np.float32) if has_y_sig else None,
    )


def read_render_arrays(path: str | Path, prob_threshold: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    path = Path(path)
    if is_h5_path(path):
        return _read_h5_render_arrays(path, prob_threshold)
    return _read_csv_render_arrays(path, prob_threshold)
