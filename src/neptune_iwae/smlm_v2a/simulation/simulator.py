from typing import Union, Tuple

import torch

from .loc_generator import EmitterSampler
from .bkg_generator import Background

from ..generic.localizations import FrameLocSet
from ..calibration.camera import Camera
from ..calibration.psf import DeltaPSF, CSplinePSF
from ..utils.helpers import set_device


class Simulation:
    """Main simulation module"""

    def __init__(self, cam:Camera, psf:CSplinePSF, loc_sampler:EmitterSampler, bkg_sampler:Background=None, 
                 phot_th:float=0.0, n_buffer:int=500, device:Union[torch.device, str]=None):
        """
        Initiation simulation
        Args:
            cam:            Camera that has the 'forward' method to sample the camera noise from a photon_frame
            psf:            CSplinePSF that has the 'psf_conv' method to convolve the localizations with psf kernel
            loc_sampler:    EmitterSampler that has the 'sample' method to sample the localizations
            bkg_sampler:    Background that has the 'forward' method to sample the background
            phot_th:        photon threshold applied to target generator (loc2tar)
            n_buffer:       maximum number of emitters per frame (within emitter.extents) allowed to generate the target
            i_type:         integer data type, default = torch.int32
            f_type:         float data type, default = torch.float32
            device:         str {'cpu', 'cuda:0', 'cuda'}
        """
        self.device = set_device(device)

        self.cam = cam
        self.psf = psf
        self.loc_sampler = loc_sampler
        self.bkg_sampler = bkg_sampler

        self.phot_th = phot_th
        self.n_buffer = n_buffer
        
    @property
    def xextent(self) -> Tuple[int, int]:
        return self.loc_sampler.structure.xextent
    
    @property
    def yextent(self) -> Tuple[int, int]:
        return self.loc_sampler.structure.yextent
    
    @property
    def ix_low(self) -> int:
        return self.loc_sampler.ix_low
    
    @property
    def ix_high(self) -> int:
        return self.loc_sampler.ix_high

    def _safety_check(self, n_max:int):
        """check if the n_max is valid to generate the targets"""
        if n_max <= 0:
            raise ValueError("No valid emitters to generate the target")
        elif n_max > self.n_buffer:
            raise ValueError(f"Number of actual emitters exceeds the allowed maximum numbers {self.n_buffer}")
        
    def loc2frames(self, loc:FrameLocSet) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        get the frames from a FrameLocSet via psf.psf_conv
        Args:
            loc:            the FrameLocSet
        Returns:
            torch.Tensor:   (nframes, loc.yextent[1]-loc.yextent[0], loc.xextent[1]-loc.xextent[0]) simulated imaging frames
            torch.Tensor:   (nframes, loc.yextent[1]-loc.yextent[0], loc.xextent[1]-loc.xextent[0]) simulated background frames
        """
        frames = self.psf.psf_conv(loc, device=self.device)
        if self.bkg_sampler is not None:
            frames, bkg_frames = self.bkg_sampler.forward(frames)
        else:
            bkg_frames = None
        frames = self.cam.forward(loc.xextent, loc.yextent, frames)
        return frames, bkg_frames
    
    def loc2detection(self, loc:FrameLocSet) -> torch.BoolTensor:
        """
        get the binary frames from a FrameLocSet indicating if a pixel has emitters
        Args:
            loc:            the FrameLocSet
        Returns:
            torch.Tensor:   (nframes, loc.yextent[1]-loc.yextent[0], loc.xextent[1]-loc.xextent[0]) simulated detection image
        """
        return DeltaPSF(loc.nchannels).psf_conv(loc, device=self.device)
        
    def loc2targets(self, loc:FrameLocSet) -> Tuple[torch.Tensor, torch.BoolTensor]:
        """
        convert a FrameLocSet to target
        Args:
            loc:            the FrameLocSet
        Returns:
            torch.Tensor:       (N, n_max, 4) the pxyz target
            torch.BoolTensor:   (N, n_max) the activation of the pxyz_tar
        """
        loc = loc.phot_filter(self.phot_th)
        loc = loc.sort_by_frame()
        _indf = loc.indf.to(self.device)
        _phot = loc.phot.to(self.device)
        _xyz = loc.xyz.to(self.device)

        ix_low = _indf.min()
        ix_high = _indf.max() + 1
        n_frames = ix_high - ix_low

        frame_set, n_loc = torch.unique_consecutive(_indf, return_counts=True)
        n_max = n_loc.max()
        self._safety_check(n_max)
        
        pxyz_tar = torch.zeros((n_frames, n_max, 4), dtype=loc.f_type, device=self.device)
        mask_tar = torch.full((n_frames, n_max), False, device=self.device)
        for i, f in enumerate(frame_set):
            ix = _indf == f
            f -= ix_low
            mask_tar[f, :n_loc[i]] = True
            pxyz_tar[f, :n_loc[i], 0] = _phot[ix]
            pxyz_tar[f, :n_loc[i], 1:] = _xyz[ix]

        return pxyz_tar, mask_tar
    
    def sample(self) -> Tuple[FrameLocSet, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.BoolTensor]:
        """
        Sample a new set of localizations and forward through the simulation pipeline
        Returns:
            FrameLocSet:    Sampled localizations
            torch.Tensor:   (nframes, H, W), Simulated imaging frames according to the sampled localizations
            torch.Tensor:   (nframes, H, W), Simulated detection frames
            torch.Tensor:   (nframes, H, W), Simulated background frames
            torch.Tensor:   (nframes, n_max, 4), pxyz_mu target according to the sampled localizations
            torch.Tensor:   (nframes, n_max), mask_mu activation mask of pxyz_mu
        """
        loc = self.loc_sampler.sample(self.device)
        frames, bkg_frames = self.loc2frames(loc)
        detect_frames = self.loc2detection(loc)
        pxyz_tar, mask_tar = self.loc2targets(loc)    
        return loc, frames, detect_frames, bkg_frames, pxyz_tar, mask_tar
    
    @classmethod
    def parse(cls, cam:Camera, psf:CSplinePSF, loc_sampler:EmitterSampler, bkg_sampler:Background, param):
        return cls(cam=cam, psf=psf, loc_sampler=loc_sampler, bkg_sampler=bkg_sampler,
                   phot_th = param.DataFilter.emitter_label_photon_min,
                   n_buffer = param.DataFilter.max_number_targets,
                   device = param.Hardware.device_simulation)