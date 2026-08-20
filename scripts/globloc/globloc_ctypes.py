from __future__ import annotations

import ctypes
from pathlib import Path

import numpy as np
import numpy.ctypeslib as ctl


class GlobLocFit:
    """Small Linux host wrapper around the official GlobLoc CUDA kernel."""

    def __init__(self, library_path: Path):
        self.library_path = Path(library_path).resolve()
        self.library = ctypes.CDLL(str(self.library_path))
        self.fit_function = self.library.globloc_fit_multichannel_emccd_spline
        float_ptr = ctl.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS")
        int_ptr = ctl.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS")
        self.fit_function.argtypes = [
            float_ptr,
            int_ptr,
            ctypes.c_int,
            float_ptr,
            float_ptr,
            float_ptr,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            float_ptr,
            float_ptr,
            float_ptr,
        ]
        self.fit_function.restype = ctypes.c_int

    @staticmethod
    def parameter_count(shared: np.ndarray, no_channels: int) -> int:
        shared_row = np.asarray(shared, dtype=np.int32)[0]
        return int(5 * no_channels - int(shared_row.sum()) * (no_channels - 1))

    def fit(
        self,
        data: np.ndarray,
        shared: np.ndarray,
        iterations: int,
        coeff: np.ndarray,
        dT_all: np.ndarray,
        init_z: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        data = np.ascontiguousarray(data, dtype=np.float32)
        shared = np.ascontiguousarray(shared, dtype=np.int32)
        coeff = np.ascontiguousarray(coeff, dtype=np.float32)
        dT_all = np.ascontiguousarray(dT_all, dtype=np.float32)
        init_z = np.ascontiguousarray(init_z, dtype=np.float32)
        if data.ndim != 4 or data.shape[0] != 2 or data.shape[2] != data.shape[3]:
            raise ValueError(f"expected data shape (2, N, sz, sz), got {data.shape}")
        nfits = int(data.shape[1])
        if nfits <= 0:
            raise ValueError("cannot fit an empty batch")
        if shared.shape != (nfits, 5):
            raise ValueError(f"expected shared shape {(nfits, 5)}, got {shared.shape}")
        if dT_all.shape != (nfits, 4, 5):
            raise ValueError(f"expected dTAll shape {(nfits, 4, 5)}, got {dT_all.shape}")
        if init_z.shape != (nfits,):
            raise ValueError(f"expected initZ shape {(nfits,)}, got {init_z.shape}")
        if coeff.ndim != 5 or coeff.shape[0] != 2 or coeff.shape[1] != 64:
            raise ValueError(f"expected coeff shape (2, 64, z, y, x), got {coeff.shape}")

        no_parameters = self.parameter_count(shared, 2)
        parameters = np.empty((no_parameters + 1, nfits), dtype=np.float32)
        crlbs = np.empty((no_parameters, nfits), dtype=np.float32)
        log_likelihood = np.empty((nfits,), dtype=np.float32)
        status = self.fit_function(
            data,
            shared,
            int(iterations),
            coeff,
            dT_all,
            init_z,
            int(data.shape[2]),
            int(coeff.shape[4]),
            int(coeff.shape[3]),
            int(coeff.shape[2]),
            nfits,
            2,
            parameters,
            crlbs,
            log_likelihood,
        )
        if status != 0:
            raise RuntimeError(f"GlobLoc CUDA fit failed with status {status}")
        return parameters, crlbs, log_likelihood
