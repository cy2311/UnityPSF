from typing import Union, Callable, Optional
import time
import warnings
from functools import partial
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from ..models.smlm_model import SingleUNet, DoubleUNet
from ..data.smlm_dataset import InferenceDataset
from ..data.data_factory import FrameRescalar
from ..data.feature_factory import SpatialIntegration
from ..generic.localizations import FrameLocSet
from ..utils import hardware
from ..utils.io import frames_io


class Infer:

    def __init__(self, model:Union[SingleUNet, DoubleUNet], ch_in:int,
                 frame_proc:FrameRescalar, feature_extactor:Optional[SpatialIntegration], device:Union[str, torch.device], 
                 batch_size:Union[int, str]='auto', num_workers:int=0, pin_memory:bool=False, 
                 forward_cat:Union[str, Callable]='emitter'):
        """
        Convenience class for inference.
        Args:
            model:          pytorch model
            ch_in:          number of input channels
            frame_proc:     frame pre-processing pipeline
            post_proc:      post-processing pipeline
            device:         device where to run inference
            batch_size:     batch-size or 'auto' if the batch size should be determined automatically (only use in combination with cuda)
            num_workers:    number of workers
            pin_memory:     pin memory in dataloader
            forward_cat:    method which concatenates the output batches. Can be string or Callable.
                            Use 'em' when the post-processor outputs an EmitterSet, or 'frames' when you don't use post-processing or if
                            the post-processor outputs frames.
        """

        self.model = model
        self.ch_in = ch_in
        self.frame_proc = frame_proc
        self.feature_extractor = feature_extactor
        self.device = device
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self._forward_cat_mode = forward_cat
        self.forward_cat = None

        if str(self.device) == 'cpu' and self.batch_size == 'auto':
            warnings.warn("Automatically determining the batch size does not make sense on cpu device. "
                          "Falling back to reasonable value.")
            self.batch_size = 64

    
    def forward(self, frames:torch.Tensor) -> FrameLocSet:
        """
        Forward frames through model, pre- and post-processing and output EmitterSet
        Args:
            frames:         The frames to infer
        Returns:
            FrameLocSet:    The inffered localizations
        """
        # Set up the model
        model = self.model.to(self.device)
        model.eval()

        # Set up the dataset
        ds = InferenceDataset(frames=frames, frame_proc=self.frame_proc, frame_window=self.ch_in)

        # Set up the batch size
        bs = int(0.8 * self.get_max_batch_size(model, ds[0].size(), 1, 512)) if self.batch_size == 'auto' else self.batch_size
        
        # Set up the dataloader
        dl = DataLoader(dataset=ds, batch_size=bs, shuffle=False, drop_last=False, num_workers=self.num_workers, pin_memory=self.pin_memory)
        tqdm_enum = tqdm(dl, total=len(dl), smoothing=0.)

        # Set up the concatenate function
        self.forward_cat = self._setup_forward_cat(self._forward_cat_mode, bs)

        out = []
        indf_offset = 0
        id_offset = 0
        with torch.no_grad():
            for batch_idx, sub_frames in enumerate(tqdm_enum):
                x_in = sub_frames.to(self.device)
                y_out = model(x_in)

                if self.feature_extractor is not None:
                    out.append(self.feature_extractor.forward(y_out, indf_offset, id_offset))
                    indf_offset += len(sub_frames)
                    id_offset += len(out[-1])
                else:
                    out.append(y_out.detach().cpu())
        out = self.forward_cat(out)
        return out


    def _setup_forward_cat(self, forward_cat:Union[str, Callable]) -> Callable:
        """
        Set up the concatenate method for the inferred FrameLocSet from each batch
        Args:
            forward_cat:    concatenate method to forward
            batch_size:     size of each batch
        Returns:
            Callable:       concatenate method
        """
        if forward_cat is None:
            return lambda x: x
        
        if callable(forward_cat):
            return forward_cat

        if isinstance(forward_cat, str):
            if forward_cat == 'emitter':
                return FrameLocSet.loc_cat
            elif forward_cat == 'frames':
                return partial(torch.cat, dim=0)
            else:
                raise ValueError("forward_cat can be callable, 'emitter', or 'frames'")
        
        raise TypeError(f"Specified forward cat method was wrong.")


    @staticmethod
    def get_max_batch_size(model:torch.nn.Module, frame_size:Union[tuple, torch.Size], limit_low:int, limit_high:int) -> int:
        """
        Get maximum batch size for inference. 
        The maximum available batch size will be binary attempted (via a simple callable) between limit_low and limit_high
        Args:
            model:      model on correct device
            frame_size: ((C, )H, W) size of frames (without batch dimension)
            limit_low:  lower batch size limit
            limit_high: upper batch size limit
        Returns:
            int:        maximum available batch size between limit_low and limit_high
        """
        # The simple callable
        def model_forward_no_grad(x:torch.Tensor):
            """ Helper function because we need to account for torch.no_grad() """
            with torch.no_grad():
                o = model.forward(x)
            return o
        assert next(model.parameters()).is_cuda, "Auto determining the max batch size makes only sense when running on CUDA device."
        return hardware.get_max_batch_size(model_forward_no_grad, frame_size, next(model.parameters()).device, limit_low, limit_high)



class LiveInfer(Infer):
    def __init__(self, model:Union[SingleUNet, DoubleUNet], ch_in:int, *,
                 stream, time_wait:float=5.0, safety_buffer:int=20,
                 frame_proc:FrameRescalar, feature_extractor:Optional[SpatialIntegration], device:Union[str, torch.device],
                 batch_size:Union[int, str]='auto', num_workers:int=0, pin_memory:bool=False,
                 forward_cat:Union[str, Callable]='emitter'):
        """
        Inference from memmory mapped tensor, where the mapped file is possibly live being written to.
        Args:
            model:          pytorch model
            ch_in:          number of input channels
            stream:         output stream. Will typically get emitters (along with starting and stopping index)
            time_wait:      wait if length of mapped tensor has not changed
            safety_buffer:  buffer distance to end of tensor to avoid conflicts when the file is actively being written to
            frame_proc:     frame pre-processing pipeline
            post_proc:      post-processing pipeline
            device:         device where to run inference
            batch_size:     batch-size or 'auto' if the batch size should be determined automatically (only use in combination with cuda)
            num_workers:    number of workers
            pin_memory:     pin memory in dataloader
            forward_cat:    method which concatenates the output batches. Can be string or Callable.
                            Use 'em' when the post-processor outputs an EmitterSet, or 'frames' when you don't use post-processing or if
                            the post-processor outputs frames.
        """

        super().__init__(model=model, ch_in=ch_in, frame_proc=frame_proc, feature_extactor=feature_extractor,
                         device=device, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory,
                         forward_cat=forward_cat)
        self._stream = stream
        self._time_wait = time_wait
        self._buffer_length = safety_buffer

    def forward(self, frames:Union[torch.Tensor, frames_io.TiffTensor]):

        n_fitted = 0
        n_waited = 0
        while n_waited <= 2:
            n = len(frames)

            if n_fitted == n - self._buffer_length:
                n_waited += 1
                time.sleep(self._time_wait)  # wait
                continue

            n_2fit = n - self._buffer_length
            out = super().forward(frames[n_fitted:n_2fit])
            self._stream(out, n_fitted, n_2fit)

            n_fitted = n_2fit
            n_waited = 0

        # fit remaining frames
        out = super().forward(frames[n_fitted:n])
        self._stream(out, n_fitted, n)
