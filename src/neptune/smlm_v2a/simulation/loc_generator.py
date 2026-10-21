from abc import ABC, abstractmethod
from typing import Union, Tuple

import numpy as np
import torch

from .structure_prior import SturcturePrior, RandomStructure
from .chnl_distributor import FracNDistributor

from ..generic.localizations import EmitterLocSet, FrameLocSet
from ..utils import hardware
from ..utils.helpers import set_device



class EmitterSampler(ABC):
    """
    Abstract emitter sampler.
    Localizations of the emitters must be sampled in the unit of camera pixels for easier channel transformation and PSF convolution. 
    All inheriting class must implement a sample method.
    """

    def __init__(self, frame_rng:Tuple[int, int], structure:SturcturePrior, 
                 frac_distr:FracNDistributor=None, R2Ts:torch.Tensor=None, Ts2R:torch.Tensor=None,
                 i_type:torch.dtype=torch.int32, f_type:torch.dtype=torch.float32):
        """
        Args:
            frame_rng:  start and end frame index to sample the emitters (inclusion for frame_rng[0], exclusion for frame_rng[1])
            structure:  the geometric structure defined by StructurePrior
            frac_distr: the photon distributor across channels defined by FracNDistributor
            R2Ts:       (nchannels - 1, 2, warpdeg**2), polynomial warpping coefficients from the reference (0-th) channel to the others
            Ts2R:       (nchannels - 1, 2, warpdeg**2), polynomial warpping coefficients from the others to the reference (0-th) channel
            i_type:     torch.dtype, integer data type, default = torch.int32
            f_type:     torch.dtype, float data type, default = torch.float32
        """
        if frame_rng[0] > frame_rng[1]:
            raise ValueError("frame_rng should be ascendingly sorted")
        self.ix_low = frame_rng[0]
        self.ix_high = frame_rng[0] + 1 if frame_rng[1] == frame_rng[0] else frame_rng[1] 

        self.structure = structure
        self.frac_distr = frac_distr
        self.R2Ts = R2Ts
        self.Ts2R = Ts2R
        self.i_type = i_type     
        self.f_type = f_type

    @property
    def meta(self) -> dict:
        return {'ndim':self.structure.ndim,
                'xextent':self.structure.xextent,
                'yextent':self.structure.yextent,
                'zextent':self.structure.zextent,
                'nchannels':self.frac_distr.nchannels if self.frac_distr is not None else 1,
                'R2Ts':self.R2Ts,
                'Ts2R':self.Ts2R,
                'i_type':self.i_type,
                'f_type':self.f_type}

    @abstractmethod
    def em_avg(self) -> float:
        """average number of emitters per frame"""
        raise NotImplementedError

    @abstractmethod
    def sample(self, device:Union[torch.device, str]) -> FrameLocSet:
        """
        Returns FrameLocSet object
        Args:
            device: str {'cpu', 'cuda:0', 'cuda'}
        Returns:
            FrameLocSet: the sampled FrameLocSet        
        """
        raise NotImplementedError
    


class EmitterSamplerFrameIndependent(EmitterSampler):
    """
    Simple Emitter sampler. 
    Samples localization of each emitter from a structure (in the unit of camera pixels). 
    Samples photon of each emitter uniformly from photon_range = (photon_min, photon_max).
    Independent from the blinking kinetics.
    """

    def __init__(self, frame_rng:Tuple[int, int], structure:SturcturePrior, 
                 photon_range:tuple, density:float=None, em_avg:float=None,
                 frac_distr:FracNDistributor=None, R2Ts:torch.Tensor=None, Ts2R:torch.Tensor=None,
                 i_type:torch.dtype=torch.int32, f_type:torch.dtype=torch.float32):
        """
        Args:
            frame_rng:      start and end frame index to sample the emitters (inclusion for frame_rng[0], exclusion for frame_rng[1])
            structure:      the geometric structure defined by StructurePrior
            photon_range:   (photon_min, photon_max), range of photon values to uniformly sample from
            density:        target density in um**-2 of the emitters (exactly only when em_avg is None)
            em_avg:         target average number of the emitters (exactly only when density is None)
            frac_distr:     the photon distributor across channels defined by FracNDistributor
            R2Ts:           (nchannels - 1, 2, warpdeg**2), polynomial warpping coefficients from the reference (0-th) channel to the others
            Ts2R:           (nchannels - 1, 2, warpdeg**2), polynomial warpping coefficients from the others to the reference (0-th) channel
            i_type:         torch.dtype, integer data type, default = torch.int32
            f_type:         torch.dtype, float data type, default = torch.float32
        """
    
        super().__init__(frame_rng=frame_rng, structure=structure, frac_distr=frac_distr, 
                         R2Ts=R2Ts, Ts2R=Ts2R, i_type=i_type, f_type=f_type)
        
        if (density is None and em_avg is None) or (density is not None and em_avg is not None):
            raise ValueError("density and emitter on averge per frame, must be parsed XOR. Not both or none.")
        self._density = density if not density is None else em_avg / self.structure.area_um
        self._em_avg = em_avg if not em_avg is None else density * self.structure.area_um
        self._photon_offset = photon_range[0]
        self._photon_scale = photon_range[1] - photon_range[0]
        
    @property
    def em_avg(self) -> float:
        """average number of emitters per frame"""
        return self._em_avg
    
    def _photon_sample(self, nlocs:int, device:Union[torch.device, str]) -> torch.Tensor:
        """
        Samples photons via uniform distribution
        Args:
            nlocs:  number of localizations
            device: str {'cpu', 'cuda:0', 'cuda'}
        Returns:
            torch.Tensor:   (nlocs,) sampled photon counts for each localizations
        """
        return self._photon_offset + self._photon_scale * torch.rand((nlocs,), dtype=self.f_type, device=device)

    def sample(self, device:Union[torch.device, str]) -> FrameLocSet:
        """
        Returns FrameLocSet object
        Args:
            device: str {'cpu', 'cuda:0', 'cuda'}
        Returns:
            FrameLocSet: the sampled FrameLocSet        
        """
        _device = set_device(device)

        nfrm = self.ix_high - self.ix_low
        n_emitters = torch.poisson(self._em_avg + torch.zeros((nfrm,), dtype=self.f_type, device=_device)).type(self.i_type)
        nlocs = n_emitters.sum()

        id = torch.arange(nlocs, dtype=self.i_type, device='cpu')
        indf = torch.repeat_interleave(torch.arange(self.ix_low, self.ix_high, dtype=self.i_type, device=_device), n_emitters)
        xyz = self.structure.sample(nlocs, _device)
        phot = self._photon_sample(nlocs, _device)
        fracN = self.frac_distr.sample(nlocs, _device) if self.frac_distr is not None else None

        indf, xyz, phot, fracN = hardware.to_device([indf, xyz, phot, fracN], device='cpu')

        loc = FrameLocSet(**self.meta)
        loc.init_load(id=id, indf=indf, xyz=xyz, phot=phot, fracN=fracN)
        return loc
    
    @classmethod
    def parse(cls, frame_rng:Tuple[int, int], loc_meta:FrameLocSet, param):
        structure = RandomStructure.parse(loc_meta, param)
        frac_distr = FracNDistributor.parse(loc_meta, param) if loc_meta.nchannels > 1 else None
        return cls(frame_rng = frame_rng, 
                   structure = structure,
                   photon_range = param.Simulation.photon_range,
                   density = param.Simulation.density,
                   em_avg = param.Simulation.em_avg,
                   frac_distr = frac_distr,
                   R2Ts = loc_meta.R2Ts,
                   Ts2R = loc_meta.Ts2R,
                   i_type = loc_meta.i_type,
                   f_type = loc_meta.f_type)
    


class EmitterSamplerBlinking(EmitterSampler):
    """
    Blinking emitter sampler. 
    Samples localizations of each emitter from a structure (in the unit of camera pixels). 
    Samples photon_flux of each emitter from a Gaussian distribution photon_mu_sig = (mu, sigma).
    Samples the time t0 when a fluorophores is turned on uniformly from frame_range = (ix_low, ix_high)
    Samples the time tau of a fluorophore being fluorescence from en Exponential distribution with a given lifetime_avg
    Each fluoreophore is turned on once during the whole frame_range 
    """

    def __init__(self, frame_rng:Tuple[int, int], structure:SturcturePrior, 
                 photon_mu_sig:tuple, lifetime_avg:float, photon_threshold:float=None, density:float=None, em_avg:float=None, 
                 frac_distr:FracNDistributor=None, R2Ts:torch.Tensor=None, Ts2R:torch.Tensor=None,
                 i_type:torch.dtype=torch.int32, f_type:torch.dtype=torch.float32):
        """
        Args:
            frame_rng:          start and end frame index to sample the emitters (inclusion for frame_rng[0], exclusion for frame_rng[1])
            structure:          strcuture (in nanometre) to sample from
            photon_mu_sig:      (mu, sigma), mean and sigma of the Gaussian distribution for photon_flux sampling
            lifetime_avg:       the average lifetime of the exponential distribution for lifetime sampling
            photon_threshold:   threshold for photons
            density:            target density in um**-2 of the emitters (exactly only when em_avg is None)
            em_avg:             target average number of the emitters (exactly only when density is None)
            frac_distr:         the photon distributor across channels defined by FracNDistributor
            R2Ts:               (nchannels - 1, 2, warpdeg**2), polynomial warpping coefficients from the reference (0-th) channel to the others
            Ts2R:               (nchannels - 1, 2, warpdeg**2), polynomial warpping coefficients from the others to the reference (0-th) channel
            i_type:             torch.dtype, integer data type, default = torch.int32
            f_type:             torch.dtype, float data type, default = torch.float32
        """
    
        super().__init__(frame_rng=frame_rng, structure=structure, frac_distr=frac_distr, 
                         R2Ts=R2Ts, Ts2R=Ts2R, i_type=i_type, f_type=f_type)
        
        if (density is None and em_avg is None) or (density is not None and em_avg is not None):
            raise ValueError("density and emitter evaerge must be parsed XOR. Not both or none.")
        
        # making a scalar to tensor so that the distributions are on the same device
        self.photon_mu_sig = photon_mu_sig
        self.photon_threshold = photon_threshold

        self.lifetime_avg = lifetime_avg

        self.ix_low_extended = self.ix_low - 3.0 * self.lifetime_avg
        self.ix_high_extended = self.ix_high + 3.0 * self.lifetime_avg
        
        self._density = density if not density is None else em_avg / self.structure.area_um
        self._em_avg = em_avg if not em_avg is None else self._density * structure.area_um
        self._em_avg_total = self._em_avg * (self.ix_high_extended - self.ix_low_extended) / (self.lifetime_avg + 1.0)

    @property
    def em_avg(self) -> float:
        """average number of emitters per frame"""
        return self._em_avg
    
    def _photon_sample(self, nemitters:int, device:Union[torch.device, str]) -> torch.Tensor:
        """
        Samples the photon_flux for each emitter
        Args:
            nemitters:  number of emitters to sample
            device:     str {'cpu', 'cuda:0', 'cuda'}
        Returns:
            torch.Tensors:  (nemitters,) the sampled photon_flux for each emitter
        """
        return self.photon_mu_sig[0] + self.photon_mu_sig[1] * torch.randn((nemitters,), dtype=self.f_type, device=device)
    
    def _tau_sample(self, nemitters:int, device:Union[torch.device, str]) -> torch.Tensor:
        """
        Samples the tau_on for each emitter
        Args:
            nemitters:  number of emitters to sample
            device:     str {'cpu', 'cuda:0', 'cuda'}
        Returns:
            torch.Tensors:  (nemitters,) the sampled tau_on for each emitter
        """
        return -self.lifetime_avg * torch.log(torch.rand((nemitters,), dtype=self.f_type, device=device))
    
    def _t0_sample(self, nemitters:int, device:Union[torch.device, str]) -> torch.Tensor:
        """
        Samples the t0 for each emitter
        Args:
            nemitters:  number of emitters to sample
            device:     str {'cpu', 'cuda:0', 'cuda'}
        Returns:
            torch.Tensors:  (nemitters,) the sampled t0 for each emitter
        """
        return self.ix_low_extended + (self.ix_high_extended - self.ix_low_extended) * torch.rand((nemitters,), dtype=self.f_type, device=device)
    
    def _sample(self, device:torch.device) -> EmitterLocSet:
        """
        Samples localizations emitter-wise. Returns EmitterLocSet object
        Args:
            device:         str {'cpu', 'cuda:0', 'cuda'}
        Returns:
            EmitterLocSet:  the sampled blinking emitters
        """
        rng = np.random.default_rng()
        n_emitters = rng.poisson(lam=self._em_avg_total)

        xyz = self.structure.sample(n_emitters, device)
        phot = self._photon_sample(n_emitters, device)
        if self.photon_threshold is not None:
            phot = torch.clamp(phot, self.photon_threshold)
        t0 = self._t0_sample(n_emitters, device)
        tau = self._tau_sample(n_emitters, device)
        te = t0 + tau
        fracN = self.frac_distr.sample(n_emitters, device) if self.frac_distr is not None else None
        
        xyz, phot, t0, te, fracN = hardware.to_device([xyz, phot, t0, te, fracN], device='cpu')

        loc = EmitterLocSet(**self.meta)
        loc.init_load(xyz=xyz, phot=phot, fracN=fracN, t0=t0, te=te)
        return loc
    
    def sample(self, device:torch.device) -> FrameLocSet:
        """
        Samples localizations frame-wise. Returns FrameLocSet object
        Args:
            device:         str {'cpu', 'cuda:0', 'cuda'}
        Returns:
            FrameLocSet:    the localizations from the sampled emitters
        """
        emitterloc = self._sample(device)
        return emitterloc.to_frameloc((self.ix_low, self.ix_high))
    
    @classmethod
    def parse(cls, frame_rng:Tuple[int, int], loc_meta:FrameLocSet, param):
        structure = RandomStructure.parse(loc_meta, param)
        frac_distr = FracNDistributor.parse(loc_meta, param) if loc_meta.nchannels > 1 else None
        return cls(frame_rng = frame_rng, 
                   structure = structure,
                   photon_mu_sig = param.Simulation.photon_mu_sig,
                   lifetime_avg = param.Simulation.lifetime_avg,
                   photon_threshold = param.Simulation.photon_threshold,
                   density = param.Simulation.density,
                   em_avg = param.Simulation.em_avg,
                   frac_distr = frac_distr,
                   R2Ts = loc_meta.R2Ts,
                   Ts2R = loc_meta.Ts2R,
                   i_type = loc_meta.i_type,
                   f_type = loc_meta.f_type)
