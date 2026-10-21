from abc import ABC, abstractmethod
from typing import Union
import torch


class Background(ABC):

    _forward_modes = ('like', 'cum', 'tuple')
    
    def sanity_check(self, return_type:str):
        if return_type not in self._forward_modes:
            raise ValueError(f"Unsupported return type. Supported are {self._forward_modes}")
    
    @abstractmethod
    def sample(self, size:torch.Size, f_type:torch.dtype=torch.float32, device:Union[str, torch.device]=None) -> torch.Tensor:
        """
        Args:
            size:   size of the sample
            dtype:  torch.dtype for the sampling
            device: str, {'cpu', 'cuda:0', 'cuda'}
        Returns:
            background sample
        """
        raise NotImplementedError

    def sample_like(self, x:torch.Tensor) -> torch.Tensor:
        """Samples background in the shape and on the device as the input."""
        return self.sample(size=x.size(), f_type=x.dtype, device=x.device)
    
    def forward(self, x:torch.Tensor, return_type:str='tuple'):
        """Samples background in the shape and on the device as the input."""
        
        self.sanity_check(return_type)
        
        bkg = self.sample_like(x)
        if return_type == 'like':
            return bkg
        if return_type == 'cum':
            return bkg + x
        return x + bkg, bkg
    


class UniformBackground(Background):
    """Spatially constant background"""

    def __init__(self, bkg_uniform:Union[float, tuple]):

        super().__init__()
        self.bkg_uniform = bkg_uniform
        if isinstance(self.bkg_uniform, float):
            self._bkg_distribution_impl = lambda sz : bkg_uniform + torch.zeros((sz,))
        elif isinstance(bkg_uniform, (list, tuple)):
            _shift = bkg_uniform[0]
            _scale = bkg_uniform[1] - bkg_uniform[0]
            self._bkg_distribution_impl = lambda sz : _shift + _scale * torch.rand((sz,))
    
    def sample(self, size:torch.Size, f_type:torch.dtype=torch.float32, device:Union[str, torch.device]=None) -> torch.Tensor:

        assert len(size) in (2, 3, 4), "Not implemented size spec."
        
        # create as many as sample as there are batch-dims
        bkg = self._bkg_distribution_impl(size[0] if len(size)>=3 else 1)
        if len(size) >= 3:
            bkg = bkg.view(-1, *((1,) * (len(size) - 1)))
        
        return bkg.to(device, dtype=f_type) + torch.zeros(size, dtype=f_type, device=device)