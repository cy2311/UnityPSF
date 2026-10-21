from abc import ABC, abstractmethod
from typing import Union, Tuple

import math
import torch

from ..generic.localizations import FrameLocSet, EmitterLocSet
from ..utils.helpers import to_tensor

class SturcturePrior(ABC):
    """
    Abstract structure which can be sampled from
    All structures is defined in the unit of camera pixel size for easier channel transformation and PSF convolution
    """

    def __init__(self, ndim:int, cam_pxszs:Tuple[float, float, float], 
                 xextent:Tuple[int, int], yextent:Tuple[int, int], zextent:Tuple[int, int]=None,
                 f_type:torch.dtype=torch.float32):
        """
        Args:
            ndim:       dimensionality
            cam_pxszs:  [cam_pxszx, cam_pxszy, cam_pxszz/zstep], size of camera pixels in nm
            xextent:    extent in x in the unint of cam_pxszx, inclusion for xextent[0], exclusion for xextent[1]
            yextent:    extent in y in the unit of cam_pxszy.
            zextent:    extent in z in the unit of cam_pxszz/zstepsz
            f_type:     float type for sampling
        """
        if ndim not in {2, 3}:
            raise ValueError(f"Only 2- or 3-dimensionality is supported, but the given ndim is {ndim}")
        if len(cam_pxszs) < ndim:
            raise ValueError(f"insufficient pixel size information for strucutre of {ndim} dimension")
        self.ndim = ndim
        self.cam_pxszs = cam_pxszs
        self.f_type = f_type
        
        if xextent[0] >= xextent[1]:
            raise ValueError(f"invalid xextent where {xextent[1]} <= {xextent[0]}")
        if yextent[0] >= yextent[1]:
            raise ValueError(f"invalid yextent where {yextent[1]} <= {yextent[0]}")
        self.xextent = xextent
        self.yextent = yextent
        self.zextent = None
        if ndim == 3:
            if zextent[0] >= zextent[1]:
                raise ValueError(f"invalid zextent where {zextent[1]} <= {zextent[0]}")
            self.zextent = zextent
        
        self.scale = [self.xextent[1]-self.xextent[0], self.yextent[1]-self.yextent[0]]
        self.shift = [self.xextent[0], self.yextent[0]]
        if self.ndim == 3:
            self.scale.append(self.zextent[1] - self.zextent[0])
            self.shift.append(self.zextent[0])            
    
    @property
    def area_px(self) -> float:
        """
        calculate the area (2D projection of a volume) occupied by the structure.
        area should be given in the unit of px^2.
        """
        return (self.xextent[1] - self.xextent[0]) * (self.yextent[1] - self.yextent[0])
    
    @property
    def area_um(self) -> float:
        """
        calculate the area (2D projection of a volume) occupied by the structure.
        area should be given in the unit of um**2.
        """
        return self.area_px * math.prod(self.cam_pxszs[:2]) * 1e-6
    
    @abstractmethod
    def sample(self, nemitters:int, device:torch.device) -> torch.Tensor:
        """
        sample the localizations of n=nemitters emitters from structure.
        Args:
            nemitters:  number of emitters
            device:     str {'cpu', 'cuda:0', 'cuda'} 
        Returns:
            xyz:        (nemitters, ndim) float tensor, unifromly distributed localizations within the given extent 
        """
        raise NotImplementedError
    


class RandomStructure(SturcturePrior):
    """Structure with emitters uniformly distributed in a 3D cubic / 2D square"""

    def __init__(self, ndim:int, cam_pxszs:Tuple[float, float, float], 
                 xextent:Tuple[int, int], yextent:Tuple[int, int], zextent:Tuple[int, int]=None,
                 f_type:torch.dtype=torch.float32):
        """
        Args:
            ndim:       dimensionality
            cam_pxszs:  [cam_pxszx, cam_pxszy, cam_pxszz/zstep], size of camera pixels in nm
            xextent:    extent in x in the unint of cam_pxszx, inclusion for xextent[0], exclusion for xextent[1]
            yextent:    extent in y in the unit of cam_pxszy.
            zextent:    extent in z in the unit of cam_pxszz/zstepsz
            f_type:     float type for sampling
        """
        super().__init__(ndim=ndim, cam_pxszs=cam_pxszs, xextent=xextent, yextent=yextent, zextent=zextent, f_type=f_type)
    
    def sample(self, nemitters:int, device:Union[torch.device, str]) -> torch.Tensor:
        """
        uniformly seed the emitters into the emitter_extent
        Args:
            nemitters:  number of emitters
            device:     str {'cpu', 'cuda:0', 'cuda'} 
        Returns:
            xyz:        (nemitters, ndim) float tensor, unifromly distributed localizations within the given extent 
        """
        _scale = torch.tensor(self.scale, dtype=self.f_type, device=device)
        _shift = to_tensor(self.shift, dtype=self.f_type, device=device)
        return torch.rand((nemitters, self.ndim), dtype=self.f_type, device=device) * _scale[None, :self.ndim] + _shift[None, :self.ndim]
    
    @classmethod
    def parse(cls, loc_meta:Union[FrameLocSet, EmitterLocSet], param):
        """
        Args:
            loc_meta:   the LocSet instance
            param:      configured parameters
        """
        return cls(ndim=loc_meta.ndim,
                   cam_pxszs=param.Camera.cam_pxszs,
                   xextent=loc_meta.xextent,
                   yextent=loc_meta.yextent,
                   zextent=loc_meta.zextent,
                   f_type=loc_meta.f_type)