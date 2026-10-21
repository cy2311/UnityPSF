from typing import Union
import numbers
import torch

from ..generic.localizations import FrameLocSet, EmitterLocSet
from ..utils.helpers import to_tensor, to_tensorslicer

class FracNDistributor:
    """distributing photons of each emitter to different channels"""

    def __init__(self, nchannels:int, fluorID:Union[int, torch.IntTensor]=None, weights:torch.Tensor=None, 
                 frac_mu:torch.Tensor=None, f_type:torch.dtype=torch.float32):
        """
        Args:
            nchannels:  number of channels
            fluorID:    the index of a type of fluorophore 
                        so that frac_mu[fluorID] represents the channel distribution of the specified flurophore
            weights:    (n_fltypes,) the population of different types of fluorophores
                        fluorID and weights must be XOR, and cannot be both specified or None
            frac_mu:    (n_fltypes, nchannels) the distribution across the nchannels of each type of fluorophore
                        the distribution will be uniformly sampled within [0, 1] if not given
            f_type:     torch.dtype, float data type, default = torch.float32
        """
    
        assert nchannels > 1, f"Only supported for multi-channels but input nchannels = {nchannels}"
        self.nchannels = nchannels
        self.f_type = f_type
        
        if frac_mu is not None: 
            # sanity check
            if (fluorID is None and weights is None) or (fluorID is not None and weights is not None):
                raise ValueError("density and emitter evaerge must be parsed XOR. Not both or none.")
            
            # setup the frac_mu
            assert len(frac_mu.shape) == 2, "frac_mu must be 2D-matrix of size (n_fltypes, nchannels)"
            assert frac_mu.shape[1] == self.nchannels, f"nchannels mismatch: nchannels={self.nchannels}, frac_mu.shape[1]={frac_mu.shape[1]}"
            self.frac_mu = self._norm(frac_mu, axis=1)

            # setup weights
            if weights is not None:
                assert len(weights) == len(frac_mu), f"n_fltypes mismatch: len(weights)={len(weights)}, len(frac_mu)={len(frac_mu)}"
                self.weights = self._norm(weights, axis=0)
                self.fluorID = None 
                self.sample_core = self._distribute_mixedFluor   
            
            # setup fluorID
            if fluorID is not None:
                self.fluorID = to_tensorslicer(fluorID)
                self.weights = None
                self.sample_core = self._distribute_specFluor
            
        else:
            self.frac_mu = None
            self.fluorID = None
            self.weights = None
            self.sample_core = self._distribute_rand
        
    @staticmethod
    def _norm(arr:torch.Tensor, axis:int=-1) -> torch.Tensor:
        """normalize the input tensor along the given axis (-1 by default)"""
        dum = arr.sum(axis).unsqueeze(axis)
        return arr / dum

    def _distribute_rand(self, n_emitters:int, device:Union[torch.device, str]) -> torch.Tensor:
        """uniformly distributing the photons across each channel"""
        frac_mu = torch.rand((n_emitters, self.nchannels), dtype=self.f_type, device=device)
        return self._norm(frac_mu, axis=1)
    
    def _distribute_specFluor(self, n_emitters:int, device:Union[torch.device, str]) -> torch.Tensor:
        """distribute the photons for all the emitters with a fixed distribution"""
        frac_mu = to_tensor(self.frac_mu, dtype=self.f_type, device=device)
        if len(self.fluorID) == 1:
            return frac_mu[self.fluorID].repeat(n_emitters, 1)
        else:
            assert len(self.fluorID) == n_emitters, f"n_emitters mismatch: len(fluorID)={len(self.fluorID)}, expected is {n_emitters}"
            return frac_mu[self.fluorID]
    
    def _distribute_mixedFluor(self, n_emitters:int, device:Union[torch.device, str]) -> torch.Tensor:
        """
        Sample different fluorophore types for each emitter accroding to the weights
        distribute the photons of each emitter a fixed distribution
        """
        weights = to_tensor(self.weights, dtype=self.f_type, device=device)
        frac_mu = to_tensor(self.frac_mu, dtype=self.f_type, device=device)
        _idx = weights.multinomial(num_samples=n_emitters, replacement=True)
        return frac_mu[_idx]
    
    def sample(self, n_emitters:int, device:Union[torch.device, str], ch:Union[str, int]='all') -> torch.Tensor:
        fracN = self.sample_core(n_emitters, f_type=self.f_type, device=device)
        if isinstance(ch, numbers.Number):
            return fracN[:, [ch]]
        else:
            return fracN[:, [0]] if ch == 'ref' else fracN[:, 1:] if ch == 'tar' else fracN
        
    @classmethod
    def parse(cls, loc_meta:Union[FrameLocSet, EmitterLocSet], param):
        """
        Args:
            loc_meta:   the LocSet instance
            param:      configured parameters
        """
        return cls(nchannels=loc_meta.nchannels,
                   fluorID=param.Simulation.fluorID,
                   weights=param.Simulation.weights,
                   frac_mu=param.Simulation.frac_mu,
                   f_type=loc_meta.f_type)