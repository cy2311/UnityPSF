from abc import ABC, abstractmethod
from typing import Tuple
import collections.abc
import time

import torch
import torch.utils.data
from torch.utils.data import Dataset

from .data_factory import FrameRescalar, BkgFrameRescalar, TargetProcess

from ..generic.localizations import FrameLocSet
from ..simulation.simulator import Simulation
from ..utils import hardware


class SMLMDataset(Dataset, ABC):
    """
    SMLM base dataset
    Attributes:
        loc:            FrameLocSet, source FrameLocSet for training and testing
        frames:         (nframes, imheight, imwidth) acquisition frames corresponding to the loc or simulated from the loc
        detect_frames:  (nframes, imheight, imwidth) groundtruth binary frames indicating if a pixel has an emitter
        bkg_frames:     (nframes, imheight, imwidth) groundtruth bkg_frames
        pxyz_tar:       (nframes, n_max, 4) pxyz target generated from the loc
        mask_tar:       (nframes, n_max) activation mask for pxyz_tar
        
        frame_proc:     FrameRescalar, Frame rescaling processing
        bkg_frame_proc: BkgFrameRescalar, Background frame rescaling processing
        tar_proc:       TargetProcess, Target processing including attribute disabling and rescaling
        
        frame_window:   Number of frames per sample batch (e.g. 1 for SingleUNet and 1 or 3 for DoubleUNet)
        return_loc:     bool, True to include the indexed FrameLocSet in the returns of __getitem__()
    NOTE:
        the simulation and processing can be executed on desired device but will return to cpu
    """

    def __init__(self, *, frame_proc:FrameRescalar, bkg_frame_proc:BkgFrameRescalar, tar_proc:TargetProcess, frame_window:int, return_loc:bool):
        super().__init__()
        
        self.frame_proc = frame_proc
        self.bkg_frame_proc = bkg_frame_proc
        self.tar_proc = tar_proc

        self._frames = None
        self._detect_frames = None
        self._bkg_frames = None
        self._pxyz_tar = None
        self._mask_tar = None
        self._loc_tar = None
        
        if frame_window is not None and frame_window % 2 != 1:
            raise ValueError(f"Unsupported frame window: Frame window must be odd integerated, not {frame_window}")
        self.frame_window = frame_window
        self.return_loc = return_loc

    def __len__(self) -> int:
        return self._frames.size(0) - (self.frame_window - 1)
    
    def _pad_index(self, ix:int) -> int:
        """pad the index ix adapting to the frame window"""
        return ix + (self.frame_window - 1) // 2

    def _get_frames(self, frames:torch.Tensor, ix:int) -> torch.Tensor:
        """
        Get the frames `self.frame_window` around the ix-th frame
        Returns:
            torch.Tensor: (self.frame_window,...) cropped frames around the idx-th frame
        """
        hw = (self.frame_window - 1) // 2
        frame_ix = torch.arange(ix - hw, ix + hw + 1)
        return frames[frame_ix]
    
    def _process_sample(self, frames:torch.Tensor, bkg_frames:torch.Tensor, pxyz_tar:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Process the frames, background frames, and pxyz_tar
        Returns:
            frames:     ((nframes, )H, W), processed frames, should be squezzed if channels_in == 1
            bkg_frames: ((nframes, )H, W), processed background frames
            pxyz_tar:   ((nframes, )n_max, 4), processed target parameter
        """
        if self.frame_proc is not None:
            frames = self.frame_proc.forward(frames)
        if self.bkg_frame_proc is not None:
            bkg_frames = self.bkg_frame_proc.forward(bkg_frames)
        if self.tar_proc is not None:
            pxyz_tar = self.tar_proc.forward(pxyz_tar)
        return frames, bkg_frames, pxyz_tar
    
    def _nan_check(self):
        """ Checks if NaN is contained in the dataset """
        if torch.isnan(self._frames).any():
            raise ValueError("Warning: NaN detected in simulated frames, replacing with zeros")
        if torch.isnan(self._bkg_frames).any():
            raise ValueError("Warning: NaN detected in simulated bkg_frames, replacing with zeros")
        if torch.isnan(self._detect_frames).any():
            raise ValueError("Warning: NaN detected in simulated detect_frames, replacing with zeros")
        if torch.isnan(self._pxyz_tar).any():
            raise ValueError("Warning: NaN detected in simulated pxyz_tar, replacing with zeros")
        if torch.isnan(self._mask_tar).any():
            raise ValueError("Warning: NaN detected in simulated mask_tar, replacing with zeros")
    
    @abstractmethod
    def __getitem__(self, idx:int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.BoolTensor, FrameLocSet]:
        """
        Returns:
            torch.Tensor:       sub_frame, (frame_window, H, W) the simulated / acquisition frames, and being processed by the FrameRescalar
            torch.Tensor:       sub_detect_frame, (H, W) the simulated binary frame indicating if a pixel has emitter
            torch.Tensor:       sub_bkg_frame, (H, W) the simulated background frame, and being processed by the BkgFrameRescalar
            torch.Tensor:       sub_pxyz_tar, (n_max, 4) the target pxyz values generated from the simulated loc, and being processed by the TargetProcess
            torch.BoolTensor:   sub_mask_tar, (n_max,) the activation mask for pxyz_tar
            FrameLocSet:        sub_loc, the localization set at a given index, default is None
        """
        raise NotImplementedError


    
class SMLMStaticDataset(SMLMDataset):
    """
    A static SMLMDataset
    NOTE: 1. source frames, bkg_frames, pxyz_tar, and mask_tar should all be on cpu
    NOTE: 2. indf of target loc (loc_tar) should be adjusted according to the frame_window
    """

    def __init__(self, frames:torch.Tensor, detect_frames:torch.Tensor, bkg_frames:torch.Tensor, 
                 pxyz_tar:torch.Tensor, mask_tar:torch.BoolTensor, loc:FrameLocSet, *, 
                 frame_proc:FrameRescalar=None, bkg_frame_proc:FrameRescalar=None, tar_proc:TargetProcess=None, 
                 frame_window:int=3, return_loc:bool=True):
        super().__init__(frame_proc=frame_proc, bkg_frame_proc=bkg_frame_proc, tar_proc=tar_proc, frame_window=frame_window, return_loc=return_loc)
        
        if frames.dim() != 3:
            raise ValueError(f"{frames.dim()} is not supported dimensions. Frames must be 3-dimentional of size (N, H, W).")
        
        if self.return_loc:
            self._loc_perframe = loc.split_by_frame(loose=True)
        
        self._frames = frames
        self._detect_frames = detect_frames
        self._bkg_frames = bkg_frames
        self._pxyz_tar = pxyz_tar
        self._mask_tar = mask_tar
        self._loc_tar = loc.realign_frame(frame_offset = -(self.frame_window - 1) // 2)
        self._nan_check()
    
    def __getitem__(self, idx:int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.BoolTensor, FrameLocSet]:
        """
        Returns:
            torch.Tensor:       sub_frame, (frame_window, H, W) the simulated / acquisition frames, and being processed by the FrameRescalar
            torch.Tensor:       sub_detect_frame, (H, W) the simulated binary frame indicating if a pixel has emitter
            torch.Tensor:       sub_bkg_frame, (H, W) the simulated background frame, and being processed by the BkgFrameRescalar
            torch.Tensor:       sub_pxyz_tar, (n_max, 4) the target pxyz values generated from the simulated loc, and being processed by the TargetProcess
            torch.BoolTensor:   sub_mask_tar, (n_max,) the activation mask for pxyz_tar
            FrameLocSet:        sub_loc, the localization set at a given index, default is None
        """
        ix = self._pad_index(idx)
        
        sub_frames = self._get_frames(self._frames, ix)
        sub_detect_frame = self._detect_frames[ix]
        sub_bkg_frame = self._bkg_frames[ix] if self._bkg_frames is not None else None
        sub_pxyz_tar = self._pxyz_tar[ix]
        sub_mask_tar = self._mask_tar[ix]

        sub_frames, sub_bkg_frame, sub_pxyz_tar = self._process_sample(sub_frames, sub_bkg_frame, sub_pxyz_tar)
        sub_loc = self._loc_perframe[ix] if self.return_loc else None
        return sub_frames, sub_detect_frame, sub_bkg_frame, sub_pxyz_tar, sub_mask_tar, sub_loc

    
    
class InferenceDataset(SMLMDataset):
    """
    A SMLM dataset without ground truth data. data should be on cpu
    This is dummy wrapper to keep the visual appearance of a seperate dataset
    """
    def __init__(self, frames:torch.Tensor, *, frame_proc:FrameRescalar=None, frame_window:int=3):
        super().__init__(frame_proc=frame_proc, bkg_frame_proc=None, tar_proc=None, frame_window=frame_window, return_loc=False)
        if frames.dim() != 3:
            raise ValueError(f"{frames.dim()} is not supported dimensions. Frames must be 3-dimentional of size (N, H, W).")
        if torch.isnan(frames).any():
            raise ValueError("NaN detected in the input frames")
        self._frames = frames
        
    def _process_sample(self, frames:torch.Tensor) -> torch.Tensor:
        """Process the frames"""
        if self.frame_proc is not None:
            frames = self.frame_proc.forward(frames)
        return frames

    def __getitem__(self, idx:int) -> torch.Tensor:
        """
        Returns:
            torch.Tensor:       sub_frame, (frame_window, H, W) the simulated / acquisition frames, and being processed by the FrameRescalar
        """
        ix = self._pad_index(idx)
        sub_frames = self._get_frames(self._frames, ix)
        sub_frames = self._process_sample(sub_frames)
        return sub_frames   
    


class SMLMLiveDataset(SMLMDataset):
    """
    A SMLM dataset where new datasets is sampleable via the simulator.Simulation.sample() method.
    __getitem__ before processing for training.
    """

    def __init__(self, simulator:Simulation, *, frame_proc:FrameRescalar=None, bkg_frame_proc:BkgFrameRescalar=None,
                 tar_proc:TargetProcess=None, frame_window:int=3, return_loc:bool=True):
        super().__init__(frame_proc=frame_proc, bkg_frame_proc=bkg_frame_proc, tar_proc=tar_proc, frame_window=frame_window, return_loc=return_loc)
        self.simulator = simulator

    def sample(self, verbose:bool=False, nan_check:bool=False):
        """Sample new acquisition, i.e. a whole dataset"""

        # Sample new dataset
        tas = time.time()
        loc, frames, detect_frames, bkg_frames, pxyz_tar, mask_tar = self.simulator.sample()
        frames, detect_frames, bkg_frames, pxyz_tar, mask_tar = hardware.to_device([frames, detect_frames, bkg_frames, pxyz_tar, mask_tar], 'cpu')
        if verbose:
            print(f"Sampled dataset in {time.time()-tas:.2f}s. {len(loc)} localizations on {frames.size(0)} frames.")

        if self.return_loc:
            self._loc_perframe = loc.split_by_frame(loose=True)
        
        self._frames = frames
        self._detect_frames = detect_frames
        self._bkg_frames = bkg_frames
        self._pxyz_tar = pxyz_tar
        self._mask_tar = mask_tar
        self._loc_tar = loc.realign_frame(frame_offset = -(self.frame_window - 1) // 2)

        if nan_check:
            self._nan_check()

    def __getitem__(self, idx:int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.BoolTensor, FrameLocSet]:
        """
        Returns:
            torch.Tensor:       sub_frame, (frame_window, H, W) the simulated / acquisition frames, and being processed by the FrameRescalar
            torch.Tensor:       sub_detect_frame, (H, W) the simulated binary frame indicating if a pixel has emitter
            torch.Tensor:       sub_bkg_frame, (H, W) the simulated background frame, and being processed by the BkgFrameRescalar
            torch.Tensor:       sub_pxyz_tar, (n_max, 4) the target pxyz values generated from the simulated loc, and being processed by the TargetProcess
            torch.BoolTensor:   sub_mask_tar, (n_max,) the activation mask for pxyz_tar
            FrameLocSet:        sub_loc, the localization set at a given index, default is None
        """
        ix = self._pad_index(idx)
        
        sub_frames = self._get_frames(self._frames, ix)
        sub_detect_frame = self._detect_frames[ix]
        sub_bkg_frame = self._bkg_frames[ix] if self._bkg_frames is not None else None
        sub_pxyz_tar = self._pxyz_tar[ix]
        sub_mask_tar = self._mask_tar[ix]
        sub_loc = self._loc_perframe[ix] if self.return_loc else None

        sub_frames, sub_bkg_frame, sub_pxyz_tar = self._process_sample(sub_frames, sub_bkg_frame, sub_pxyz_tar)
        return sub_frames, sub_detect_frame, sub_bkg_frame, sub_pxyz_tar, sub_mask_tar, sub_loc

    

class SMLMPriorDataset(SMLMDataset):
    """
    A SMLM dataset where new datasets is sampleable via the simulator.Simulation.sample() method.
    process the whole dataset before for training __getitem__.
    """

    def __init__(self, simulator:Simulation, *, frame_proc:FrameRescalar=None, bkg_frame_proc:BkgFrameRescalar=None,
                 tar_proc:TargetProcess=None, frame_window:int=3, return_loc:bool=True):
        super().__init__(frame_proc=frame_proc, bkg_frame_proc=bkg_frame_proc, tar_proc=tar_proc, frame_window=frame_window, return_loc=return_loc)
        self.simulator = simulator
             
    def sample(self, verbose:bool=False, nan_check:bool=False):
        """Sample new dataset and process them instantaneously"""
        tas = time.time()
        loc, frames, detect_frames, bkg_frames, pxyz_tar, mask_tar = self.simulator.sample()
        if verbose:
            print(f"Sampled dataset in {time.time()-tas:.2f}s. {len(loc)} emitters on {frames.size(0)} frames.")
        
        if self.return_loc:
            self._loc_perframe = loc.split_by_frame(loose=True)

        frames, bkg_frames, pxyz_tar = self._process_sample(frames, bkg_frames, pxyz_tar)
        frames, detect_frames, bkg_frames, pxyz_tar, mask_tar = hardware.to_device([frames, detect_frames, bkg_frames, pxyz_tar, mask_tar], 'cpu')

        self._frames = frames
        self._detect_frames = detect_frames
        self._bkg_frames = bkg_frames
        self._pxyz_tar = pxyz_tar
        self._mask_tar = mask_tar
        self._loc_tar = loc.realign_frame(frame_offset = -(self.frame_window - 1) // 2)

        if nan_check:
            self._nan_check()

    def __getitem__(self, idx:int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.BoolTensor, FrameLocSet]:
        """
        Returns:
            torch.Tensor:       sub_frame, (frame_window, H, W) the simulated / acquisition frames, and being processed by the FrameRescalar
            torch.Tensor:       sub_detect_frame, (H, W) the simulated binary frame indicating if a pixel has emitter
            torch.Tensor:       sub_bkg_frame, (H, W) the simulated background frame, and being processed by the BkgFrameRescalar
            torch.Tensor:       sub_pxyz_tar, (n_max, 4) the target pxyz values generated from the simulated loc, and being processed by the TargetProcess
            torch.BoolTensor:   sub_mask_tar, (n_max,) the activation mask for pxyz_tar
            FrameLocSet:        sub_loc, the localization set at a given index, default is None
        """
        ix = self._pad_index(idx)
        
        sub_frames = self._get_frames(self._frames, ix)
        sub_detect_frame = self._detect_frames[ix]
        sub_bkg_frame = self._bkg_frames[ix] if self._bkg_frames is not None else None
        sub_pxyz_tar = self._pxyz_tar[ix]
        sub_mask_tar = self._mask_tar[ix]
        sub_loc = self._loc_perframe[ix] if self.return_loc else None
        return sub_frames, sub_detect_frame, sub_bkg_frame, sub_pxyz_tar, sub_mask_tar, sub_loc
        


class SMLMLiveSampleDataset(SMLMDataset):
    """A SMLM dataset where a new sample is drawn per (training) sample."""

    def __init__(self, simulator:Simulation, ds_len:int, *, frame_proc:FrameRescalar=None, bkg_frame_proc:BkgFrameRescalar=None,
                 tar_proc:TargetProcess=None, frame_window:int=3, return_loc:bool=True):
        super().__init__(frame_proc=frame_proc, bkg_frame_proc=bkg_frame_proc, tar_proc=tar_proc, frame_window=frame_window, return_loc=return_loc)
        self.simulator = simulator
        self.ds_len = ds_len
        self._loc_tar = None

    def __len__(self):
        return self.ds_len
    
    def __getitem__(self, idx:int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.BoolTensor, FrameLocSet]:
        """
        Returns:
            torch.Tensor:       sub_frame, (frame_window, H, W) the simulated / acquisition frames, and being processed by the FrameRescalar
            torch.Tensor:       sub_detect_frame, (H, W) the simulated binary frame indicating if a pixel has emitter
            torch.Tensor:       sub_bkg_frame, (H, W) the simulated background frame, and being processed by the BkgFrameRescalar
            torch.Tensor:       sub_pxyz_tar, (n_max, 4) the target pxyz values generated from the simulated loc, and being processed by the TargetProcess
            torch.BoolTensor:   sub_mask_tar, (n_max,) the activation mask for pxyz_tar
            FrameLocSet:        sub_loc, the localization set at a given index, default is None
        """
        self.simulator.loc_sampler.ix_low = 0
        self.simulator.loc_sampler.ix_high = self.frame_window
        ix = (self.frame_window - 1) // 2 # central slice

        loc, frames, detect_frames, bkg_frames, pxyz_tar, mask_tar = self.simulator.sample()
        frames, detect_frames, bkg_frames, pxyz_tar, mask_tar = hardware.to_device([frames, detect_frames, bkg_frames, pxyz_tar, mask_tar], 'cpu')
        sub_frames, sub_bkg_frame, sub_pxyz_tar = self._process_sample(frames, bkg_frames[ix], pxyz_tar[ix])

        sub_loc = loc.get_subframe((ix, ix+1)) if self.return_loc else None    
        return sub_frames, detect_frames[ix], sub_bkg_frame, sub_pxyz_tar, mask_tar[ix], sub_loc


class SMLMRLDataset(SMLMDataset):
    """
    Dataset for RL stage: returns processed frame window, raw middle frame, and bkg target.
    """

    def __init__(self, simulator:Simulation, *, frame_proc:FrameRescalar=None, bkg_frame_proc:BkgFrameRescalar=None,
                 frame_window:int=3, cache_device:str='cpu'):
        super().__init__(frame_proc=frame_proc, bkg_frame_proc=bkg_frame_proc, tar_proc=None, frame_window=frame_window, return_loc=False)
        self.simulator = simulator
        self._bkg_frames = None
        self.cache_device = cache_device

    def sample(self, verbose:bool=False, nan_check:bool=False):
        """Sample a new dataset and cache frames on the configured device."""
        loc, frames, detect_frames, bkg_frames, pxyz_tar, mask_tar = self.simulator.sample()
        frames = hardware.to_device(frames, self.cache_device)
        bkg_frames = hardware.to_device(bkg_frames, self.cache_device) if bkg_frames is not None else None
        self._frames = frames
        self._bkg_frames = bkg_frames
        if verbose:
            print(f"Sampled RL dataset with {frames.size(0)} frames.")
        if nan_check and torch.isnan(self._frames).any():
            raise ValueError("NaN detected in simulated frames")
        if nan_check and self._bkg_frames is not None and torch.isnan(self._bkg_frames).any():
            raise ValueError("NaN detected in simulated background frames")

    def __getitem__(self, idx:int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            torch.Tensor: sub_frame, (frame_window, H, W), processed by the FrameRescalar
            torch.Tensor: raw_mid_frame, (H, W), raw camera frame without processing
            torch.Tensor: bkg_mid_frame, (H, W), background target processed by BkgFrameRescalar
        """
        ix = self._pad_index(idx)
        sub_frames = self._get_frames(self._frames, ix)
        raw_mid = self._frames[ix]
        if self.frame_proc is not None:
            sub_frames = self.frame_proc.forward(sub_frames)
        bkg_mid = None
        if self._bkg_frames is not None:
            bkg_mid = self._bkg_frames[ix]
            if self.bkg_frame_proc is not None:
                bkg_mid = self.bkg_frame_proc.forward(bkg_mid)
        return sub_frames, raw_mid, bkg_mid


class SMLMLiveSampleRLDataset(SMLMDataset):
    """Sample a new simulation per item and return processed window + raw middle frame + bkg target."""

    def __init__(self, simulator:Simulation, ds_len:int, *, frame_proc:FrameRescalar=None, bkg_frame_proc:BkgFrameRescalar=None,
                 frame_window:int=3, cache_device:str='cpu'):
        super().__init__(frame_proc=frame_proc, bkg_frame_proc=bkg_frame_proc, tar_proc=None, frame_window=frame_window, return_loc=False)
        self.simulator = simulator
        self.ds_len = ds_len
        self.cache_device = cache_device

    def __len__(self):
        return self.ds_len

    def __getitem__(self, idx:int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            torch.Tensor: sub_frame, (frame_window, H, W), processed by the FrameRescalar
            torch.Tensor: raw_mid_frame, (H, W), raw camera frame without processing
            torch.Tensor: bkg_mid_frame, (H, W), background target processed by BkgFrameRescalar
        """
        self.simulator.loc_sampler.ix_low = 0
        self.simulator.loc_sampler.ix_high = self.frame_window
        ix = (self.frame_window - 1) // 2

        loc, frames, detect_frames, bkg_frames, pxyz_tar, mask_tar = self.simulator.sample()
        frames = hardware.to_device(frames, self.cache_device)
        bkg_frames = hardware.to_device(bkg_frames, self.cache_device) if bkg_frames is not None else None
        sub_frames = frames
        raw_mid = frames[ix]
        if self.frame_proc is not None:
            sub_frames = self.frame_proc.forward(sub_frames)
        bkg_mid = None
        if bkg_frames is not None:
            bkg_mid = bkg_frames[ix]
            if self.bkg_frame_proc is not None:
                bkg_mid = self.bkg_frame_proc.forward(bkg_mid)
        return sub_frames, raw_mid, bkg_mid


def smlm_collate(batch):
    """
    Collect for dataloader that allows for None return and FrameLocSet
    Otherwise defaults to default pytorch collate
    """    

    elem = batch[0]

    if isinstance(elem, torch.Tensor):
        out = None
        if torch.utils.data.get_worker_info():
            # if in a background process, concatenate directly into a shared memory tensor to avoid extra copy
            numel = sum([x.numel() for x in batch])
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage)
        return torch.stack(batch, 0, out=out)
    
    elif isinstance(elem, collections.abc.Sequence):
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(x) == elem_size for x in it):
            raise RuntimeError('each element in list of batch should be of the same size')
        transposed = zip(*batch)
        return [smlm_collate(samples) for samples in transposed]
    
    elif isinstance(elem, FrameLocSet):
        return [loc for loc in batch]
    
    elif elem is None:
        return None
    
    else:
        return torch.utils.data.dataloader.default_collate(batch)
