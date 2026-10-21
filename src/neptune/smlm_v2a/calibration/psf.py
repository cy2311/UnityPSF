from typing import Union, Tuple
import numbers

import math
import torch

from ..generic.localizations import FrameLocSet
from ..utils.helpers import set_device, to_tensor


class DeltaPSF:
    """ Representation of a Delta-function PSF (in the unit of camera pixel sizes) """
    
    def __init__(self, nchannels:int):
        """
        Args:
            nchannels:      number of channels
        """
        if nchannels < 1:
            raise ValueError(f"{nchannels} is unsupported number of channels, must be positive integers")
        self.nchannels = nchannels
        
    def psf_conv(self, loc:FrameLocSet, xextent:Tuple[int, int]=None, yextent:Tuple[int, int]=None, ix_low:int=None, ix_high:int=None, 
                 chID:int=0, buffer:float=4e9, device:Union[torch.device, str]=None) -> torch.BoolTensor:
        """
        Convolves the PSF to the input locset
        Args:
            locs:       includes indf (nlocs,), xyz (nlocs, ndim), phot (nlocs,) and default extents on each axis
            xextent:    the extent in x-axis
            yextent:    the extent in y-axis
            ix_low:     lower frame range
            ix_high:    upper frame range
            chID:       channel ID for convolution
            buffer:     the buffer size in byte on GPU
        Returns:
            img_phots:  (nfrm, imszy, imszx), 2D image of the convolution result
        """
        if self.nchannels != loc.nchannels:
            raise ValueError(f"nchannels mismatch: psf.nchannels={self.nchannels}, locs.nchannels={loc.nchannels}")
        
        _device = set_device(device)
        
        xextent = loc.xextent if xextent is None else xextent
        yextent = loc.yextent if yextent is None else yextent
        ix_low = loc.indf.min() if ix_low is None else ix_low
        ix_high = loc.indf.max() + 1 if ix_high is None else ix_high
        
        _loc = loc.get_infov(xextent, yextent)
        _loc = _loc.get_subframe((ix_low, ix_high))
        _loc.indf -= ix_low

        if self.nchannels > 1 and chID > 0:
            _loc = _loc.get_channel(chID)

        if len(_loc) == 0:
            return torch.zeros((ix_high - ix_low, yextent[1]-yextent[0], xextent[1]-xextent[0]),
                               dtype=torch.bool, device=_device)
                
        nfrms = ix_high - ix_low
        im_detect = torch.zeros((nfrms, yextent[1]-yextent[0], xextent[1]-xextent[0]), dtype=torch.bool, device=_device)

        # split the localization into chunks
        _nchunks = math.ceil(len(_loc) / buffer)
        _locs = _loc.split_to_chunks(_nchunks)
        for l in _locs:
            _xyz = l.xyz.to(_device)
            _indf = l.indf.to(_device)
            ind_x = _xyz[:, 0].floor().int()
            ind_y = _xyz[:, 1].floor().int()
            im_detect.index_put_(indices=(_indf, ind_y-yextent[0], ind_x-xextent[0]), values=torch.tensor(1).bool(), accumulate=False)
        
        return im_detect
    


class CSplinePSF:
    """ Representation of a cubic-spline approximated PSF (in the unit of camera pixel sizes) """

    def __init__(self, ndim:int, nchannels:int, coeff_sizes:Tuple[int, int, int], coeff:torch.Tensor):
        """
        Args:
            ndim:           number of dimension
            nchannels:      number of channels
            coeff_sizes:    [coeff_xsize, coeff_ysize, coeff_zsize], sizes must be odd
            coeff:          (nchannels, prod(coeff_sizes), 4**ndim) cubic spline coefficients for each pixel of the camera PSF
        """
        if ndim not in {2, 3}:
            raise ValueError(f"{ndim} is unsupported dimensionality, only 2 or 3 is supported")
        if len(coeff_sizes) != ndim:
            raise ValueError(f"ndim={ndim} but the len(coeff_sizes)={len(coeff_sizes)}")
        for i, _sz in enumerate(coeff_sizes):
            if not isinstance(_sz, numbers.Integral):
                raise ValueError(f"coeff_size[{i}] should be integral but is {_sz}")
            if _sz % 2 == 0:
                raise ValueError(f"coeff_size[{i}] should be odd but is {_sz}")
        
        if nchannels < 1:
            raise ValueError(f"{nchannels} is unsupported number of channels, must be positive integers")
        if coeff.shape[0] != nchannels:
            raise ValueError(f"shape mismatch: the coeff should be of shape (nchannels, prod(coeff_sizes), 4**ndim), but the coeff.shape[0]={coeff.shape[0]}")
        if coeff.shape[1] != math.prod(coeff_sizes):
            raise ValueError(f"shape mismatch: the coeff should be of shape (nchannels, prod(coeff_sizes), 4**ndim), but the coeff.shape[1]={coeff.shape[1]}")
        if coeff.shape[2] != 4**ndim:
            raise ValueError(f"shape mismatch: the coeff should be of shape (nchannels, prod(coeff_sizes), 4**ndim), but the coeff.shape[2]={4**ndim}")
        
        self.ndim = ndim
        self.nchannels = nchannels
        self.coeff_sizes = to_tensor(coeff_sizes, dtype=torch.int32, device='cpu')
        self.coeff = to_tensor(coeff, dtype=torch.float32, device='cpu')

    def _get_deltaf(self, deltas:torch.Tensor) -> torch.Tensor:
        """
        calculate part of the deltaf for calculation of the PSF values
        Args:
            deltas:     (nlocs, ndim), the deltas for each localization
        Returns:
            delta_f:    (nlocs, 4**ndim)
        """
        exponents = torch.arange(4, dtype=deltas.dtype, device=deltas.device)
        x_pow = deltas[:, [0]] ** exponents
        y_pow = deltas[:, [1]] ** exponents
        if self.ndim == 2:
            delta_f = y_pow[:, :, None] * x_pow[:, None, :]
        else:
            z_pow = deltas[:, [2]] ** exponents
            delta_f = z_pow[:, :, None, None] * y_pow[:, None, :, None] * x_pow[:, None, None, :]
        return delta_f.reshape(-1, 4**self.ndim)

    def _get_deltaf_deriv(self, deltas:torch.Tensor, var:str) -> torch.Tensor:
        """
        calculate part of the derivatives of deltaf over an variable
        Args:
            deltas:     (nlocs, ndim), the deltas for each localization
            var:        {'x', 'y', 'z'}, the independent variable for the derivative
        Returns:
            delta_dfd_: (nlocs, 4**ndim)
        """
        if var not in {'x', 'y', 'z'}:
            raise ValueError(f"{var} is unsupported aixs for derivatives, only valid for 'x', 'y', or 'z'")
        if self.ndim == 2 and var == 'z':
            raise ValueError("derivative over z is unsupported for 2d PSF")
        
        exponents = torch.arange(4, dtype=deltas.dtype, device=deltas.device)
        x_pow = deltas[:, [0]] ** exponents[None, :]
        y_pow = deltas[:, [1]] ** exponents[None, :]
        if self.ndim == 3:
            z_pow = deltas[:, [2]] ** exponents[None, :]

        if var == 'x':
            x_pow = torch.roll(x_pow, shifts=1, dims=1)
            x_pow *= exponents
        if var == 'y':
            y_pow = torch.roll(y_pow, shifts=1, dims=1)
            y_pow *= exponents
        if self.ndim == 3 and var == 'z':
            z_pow = torch.roll(z_pow, shifts=1, dims=1)
            z_pow *= exponents

        if self.ndim == 2:
            delta_dfd_ = y_pow[:, :, None] * x_pow[:, None, :]
        else:
            delta_dfd_ = z_pow[:, :, None, None] * y_pow[:, None, :, None] * x_pow[:, None, None, :]
        return delta_dfd_.reshape(-1, 4**self.ndim)

    def psf_conv(self, loc:FrameLocSet, xextent:Tuple[int, int]=None, yextent:Tuple[int, int]=None, ix_low:int=None, ix_high:int=None, 
                 chID:int=0, buffer:float=4e9, device:Union[torch.device, str]=None) -> torch.FloatTensor:
        """
        Convolves the PSF to the input locset
        Args:
            locs:       includes indf (nlocs,), xyz (nlocs, ndim), phot (nlocs,) and default extents on each axis
            xextent:    the extent in x-axis
            yextent:    the extent in y-axis
            ix_low:     lower frame range
            ix_high:    upper frame range
            chID:       channel ID for convolution
            buffer:     the buffer size in byte on GPU
        Returns:
            img_phots:  (nfrm, imszy, imszx), 2D image of the convolution result
        """
        if self.ndim != loc.ndim:
            raise ValueError(f"ndim mismatch: psf.ndim={self.ndim}, locs.ndim={loc.ndim}")
        if self.nchannels != loc.nchannels:
            raise ValueError(f"nchannels mismatch: psf.nchannels={self.nchannels}, locs.nchannels={loc.nchannels}")
        
        _device = set_device(device)
        _coeff = self.coeff.to(_device)
        _coeff_sizes = self.coeff_sizes.to(_device)

        xextent = loc.xextent if xextent is None else xextent
        yextent = loc.yextent if yextent is None else yextent
        ix_low = loc.indf.min() if ix_low is None else ix_low
        ix_high = loc.indf.max() + 1 if ix_high is None else ix_high
        
        _loc = loc.get_infov(xextent, yextent)
        _loc = _loc.get_subframe((ix_low, ix_high))
        _loc.indf -= ix_low

        if self.nchannels > 1 and chID > 0:
            _loc = _loc.get_channel(chID)

        if len(_loc) == 0:
            return torch.zeros((ix_high - ix_low, yextent[1]-yextent[0], xextent[1]-xextent[0]),
                               dtype=loc.f_type, device=_device)
                
        nfrms = ix_high - ix_low
        im_phots = torch.zeros((nfrms, yextent[1]-yextent[0], xextent[1]-xextent[0]), dtype=loc.f_type, device=_device)

        # split the localization into chunks
        _nmax = buffer / _coeff_sizes[0] / _coeff_sizes[1] / (4**self.ndim) / 4
        _nchunks = math.ceil(len(_loc) / _nmax)
        _locs = _loc.split_to_chunks(_nchunks)

        for l in _locs:
            # offset of each localization to help identify the pixels on which psf and its derivatives are calculated
            _xyz = l.xyz.to(_device)
            _phot = l.phot.to(_device)
            _indf = l.indf.to(_device)

            offsets_x = _xyz[:, 0].floor().int() - _coeff_sizes[0] // 2
            offsets_y = _xyz[:, 1].floor().int() - _coeff_sizes[1] // 2
            if self.ndim == 3:
                offsets_z = _coeff_sizes[2] // 2 - _xyz[:, 2].floor().int() 
            
            # deltas of each localization for calculation of cubic splines
            deltas = 1.0 - (_xyz - _xyz.floor())
            delta_f = self._get_deltaf(deltas)

            # psf values at each splineID of each localization
            spIDx = torch.arange(_coeff_sizes[0], dtype=torch.int32, device=_device)
            spIDy = torch.arange(_coeff_sizes[1], dtype=torch.int32, device=_device)
            spID = spIDy[None, :, None] * _coeff_sizes[0] + spIDx[None, None, :]
            if self.ndim == 3:
                spIDz = offsets_z + _coeff_sizes[2] // 2
                spID = spIDz[:, None, None] * _coeff_sizes[0] * _coeff_sizes[1] + spID
            psf_val = torch.sum(_coeff[chID][spID] * delta_f[:, None, None, :], dim=-1)
            mu = _phot[:, None, None] * psf_val
            mu = mu.view(-1)

            # map from splineId to pxIDs
            #indf = _indf.repeat_interleave(self.coeff_sizes[0] * self.coeff_sizes[1])
            indf = _indf.repeat_interleave(_coeff_sizes[0] * _coeff_sizes[1])
            pxIDy, pxIDx = torch.meshgrid(spIDy, spIDx, indexing='ij')
            pxIDx = offsets_x[:, None, None] + pxIDx[None, :, :]
            pxIDy = offsets_y[:, None, None] + pxIDy[None, :, :]
            pxIDx = pxIDx.view(-1)
            pxIDy = pxIDy.view(-1)
            mask = (pxIDx >= xextent[0]) & (pxIDx < xextent[1]) & (pxIDy >= yextent[0]) & (pxIDy < yextent[1])
            im_phots.index_put_(indices=(indf[mask], pxIDy[mask]-yextent[0], pxIDx[mask]-xextent[0]), values=mu[mask], accumulate=True)
        return im_phots
    
    @classmethod
    def parse(cls, calibration):
        return cls(ndim = calibration.ndim,
                   nchannels = calibration.nchannels,
                   coeff_sizes = calibration.CSplinePSF.coeff_sizes,
                   coeff = calibration.CSplinePSF.coeff)
