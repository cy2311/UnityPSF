from abc import ABC, abstractmethod
from typing import Union, Tuple, Optional, Callable
import torch
from ..generic.localizations import FrameLocSet


class FeatureExtract(ABC):

    def __init__(self, loc_meta:FrameLocSet, *, phot_max:float=None, z_max:float=None, bkg_max:float=None):
        """
        Args:
            loc_meta:       the meta for the localization to extract
            phot_max:       maximum photon that has normalized the photons of the training target
            z_max:          maximum z that has normalized the the z positions of the training target
            bkg_max:        maximum bkg that has normalized the target background
        """
        super().__init__()
        self.loc_meta = loc_meta
        self.phot_max = phot_max
        self.z_max = z_max
        self.bkg_max = bkg_max
    
    def skip_it(self, y_out:torch.Tensor=None) -> bool:
        """
        Skip feature extraction when a certain condition is met and implementation would fail.
        i.e. too many bright pixels in the detection channel. Default implementation would return False always.
        Args:
            y_out:  network output
        Returns:
            bool:   returns true when post-processing should be skipped 
        """        
        return True
    
    def reverse_rescale(self, y_out:torch.Tensor) -> torch.Tensor:
        """
        Scale the output of the network back to the real world
        Args:
            y_out:  (N, C=10, H, W) input tensor
        Returns:
            torch.Tensor: the rescaled tensor
        """
        if y_out.dim() != 4 or y_out.size(1) != 10:
            raise ValueError(f"input shape of {list(y_out.size())} is not supported")
        
        y_ = y_out.clone()
        if self.phot_max is not None:
            y_[:, [1, 5]] *= self.phot_max
        if self.z_max is not None:
            y_[:, [4, 8]] *= self.z_max
        if self.bkg_max is not None:
            y_[:, 9] *= self.bkg_max
        return y_
    
    def _px2offset(self, *, height:int=None, width:int=None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Translate the pixels of the outputs to coordinates
        Args:
            height: spatial output height; defaults to the metadata extent
            width: spatial output width; defaults to the metadata extent
        Returns:
            torch.Tensor: xoffset of each pixel
            torch.Tensor: yoffset of each pixel
        NOTE: the extents must be left-inclusive and right-exclusive
        """
        xextent = self.loc_meta.xextent
        yextent = self.loc_meta.yextent
        if width is None:
            x_ = torch.arange(*xextent, dtype=self.loc_meta.f_type, device='cpu') + 0.5
        else:
            x0 = float(xextent[0]) + 0.5
            x_ = torch.arange(int(width), dtype=self.loc_meta.f_type, device='cpu') + x0
        if height is None:
            y_ = torch.arange(*yextent, dtype=self.loc_meta.f_type, device='cpu') + 0.5
        else:
            y0 = float(yextent[0]) + 0.5
            y_ = torch.arange(int(height), dtype=self.loc_meta.f_type, device='cpu') + y0
        nx = int(x_.numel())
        ny = int(y_.numel())
        x_offset = x_.repeat(ny).view(ny, nx)
        y_offset = y_.repeat_interleave(nx).view(ny, nx)
        return x_offset, y_offset
    
    @abstractmethod
    def forward(self, y_out:torch.Tensor) -> FrameLocSet:
        """
        Forward anything through the post-processing and return a FrameLocSet
        Args:
            y_out:      network output
        Returns:
            FrameLocSet:    extracted localizations
        """
        raise NotImplementedError
    


class NoFeatureExtract(FeatureExtract):
    """The 'No' Post processing. Will always return an empty FrameLocSet"""

    def __init__(self, loc_meta:Optional[FrameLocSet]=None):
        super().__init__(loc_meta=loc_meta, phot_max=None, z_max=None, bkg_max=None)

    def forward(self, y_out:torch.Tensor=None) -> Union[FrameLocSet, None]:
        """
        Forward anything through the post-processing and return a FrameLocSet
        Args:
            y_out:      network output
        Returns:
            FrameLocSet:    a FrameLocSet with meta only
        """
        return self.loc_meta
    


class LookUpFeature(FeatureExtract):
    """
    Simple Feature Extraction in which one thresholds the probability output (raw threshold)
    and then look-up the features in the respective channels.
    """

    def __init__(self, loc_meta:FrameLocSet, raw_th:float, *, phot_max:float=None, z_max:float=None, bkg_max:float=None):
        """
        Args:
            loc_meta:       the meta for the localization to extract
            raw_th:         initial raw threshold
            phot_max:       maximum photon that has normalized the input frames of the nn
            z_max:          maximum z that has normalized the input frames of the nn
            bkg_max:        maximum bkg that has normalized the input frames of the nn
        """
        super().__init__(loc_meta=loc_meta, phot_max=phot_max, z_max=z_max, bkg_max=bkg_max)
        self.raw_th = raw_th
    
    def _filter(self, detection:torch.Tensor) -> torch.BoolTensor:
        """Filter any input tensor with a scalar"""
        return detection >= self.raw_th
    
    @staticmethod
    def _lookup_features(features:torch.Tensor, active_px:torch.BoolTensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            features:   (N, C, H, W) features to lookup
            active_px:  (N, H, W) activation mask
        Returns:
            torch.Tensor:   indf, (M,) frame index of each active feature (px)
            torch.Tensor:   features_active, (C, M) extracted features
        """
        if features.dim() != 4:
            raise ValueError(f"the features should be of shape (N, C, H, W), but the input features.shape={features.shape}")
        if active_px.dim() != 3:
            raise ValueError(f"the active_px should be of shape (N, H, W), but the input active_px.shape={active_px.shape}")
        
        # when active_px is all False, indf.shape = (0,), features_active.shape = (C, 0)
        indf = active_px.nonzero(as_tuple=False)[:, 0]
        features_active = features.permute(1, 0, 2, 3)[:, active_px]
        return indf, features_active
    
    def forward(self, y_out:torch.Tensor, indf_offset:int=None, id_offset:int=None) -> FrameLocSet:
        """
        Forward anything through the post-processing and return a FrameLocSet
        Args:
            y_out:          (N, C=10, H, W) network output
            indf_offset:    the offset of the indf for the current batch
            id_offset:      the offset of the id for the current batch
        Returns:
            FrameLocSet:    the output FrameLocSet
        """
        x_offset, y_offset = self._px2offset(height=int(y_out.size(-2)), width=int(y_out.size(-1)))
        x_offset = x_offset.to(y_out.device)
        y_offset = y_offset.to(y_out.device)

        y_out = self.reverse_rescale(y_out)
        y_out[:, 2] += x_offset.unsqueeze(0)
        y_out[:, 3] += y_offset.unsqueeze(0)

        active_px = self._filter(y_out[:, 0])
        indf, features = self._lookup_features(y_out, active_px)
        indf = indf.type(self.loc_meta.i_type)
        features = features.type(self.loc_meta.f_type)

        if indf_offset is not None:
            indf += indf_offset
        indf = indf.type(self.loc_meta.i_type).cpu()
        id = torch.arange(len(indf), dtype=self.loc_meta.i_type, device='cpu')
        if id_offset is not None:
            id += id_offset

        prob = features[0].cpu()        
        phot = features[1].cpu()
        xyz = features[2:5].transpose(0, 1).cpu()
        
        phot_sig = features[5].cpu()
        xyz_sig = features[6:9].transpose(0, 1).cpu()

        loc_out = FrameLocSet(**self.loc_meta.meta)
        loc_out.init_load(id=id, prob=prob, indf=indf, xyz=xyz, phot=phot, xyz_sig=xyz_sig, phot_sig=phot_sig)
        return loc_out
        
        

class SpatialIntegration(LookUpFeature):
    """Feature Extration via Spatial Integration"""

    _p_aggregations = ('sum', 'norm_sum')

    def __init__(self, loc_meta:FrameLocSet, raw_th:float, split_th:float, *, phot_max:float=None, z_max:float=None, bkg_max:float=None, 
                 p_aggregation:Union[Callable, str]='norm_sum'):
        """
        Args:
            loc_meta:       the meta for the localization to extract
            raw_th:         initial raw threshold
            split_th:       secondary threshold to detect high detection but non-local maximum
            phot_max:       maximum photon that has normalized the input frames of the nn
            z_max:          maximum z that has normalized the input frames of the nn
            bkg_max:        maximum bkg that has normalized the input frames of the nn
            p_aggregation:  aggregation method to aggregate probabilities. can be 'sum', 'max', 'norm_sum'
        """
        super().__init__(loc_meta=loc_meta, raw_th=raw_th, phot_max=phot_max, z_max=z_max, bkg_max=bkg_max)
        self.split_th = split_th
        self.p_aggregation = self.set_p_aggregation(p_aggregation)

    @classmethod
    def set_p_aggregation(cls, p_aggregation:Union[Callable, str]) -> Callable:
        """
        Sets the p_aggregation by string or callable. Returns Callable
        Args:
            p_aggregation:  probability aggregation
        """
        if isinstance(p_aggregation, str):
            if p_aggregation == 'sum':
                return torch.add
            elif p_aggregation == 'max':
                return torch.max
            elif p_aggregation == 'norm_sum':
                def norm_sum(*args):
                    return torch.clamp(torch.add(*args), 0.0, 1.0)
                return norm_sum
            else:
                raise ValueError(f"{p_aggregation} is unsupported. Supported are {cls._p_aggregations}")
        else:
            return p_aggregation
        
    @staticmethod
    def _nms(p:torch.Tensor, p_aggregation:Callable, raw_th:float, split_th:float) -> torch.Tensor:
        """
        Non-Maximum Suppresion
        Args:
            p:                  (N, H, W) the detection (probability map)
            p_aggregation:      aggregation method to aggregate probabilities. can be 'sum', 'max', 'norm_sum'
            raw_th:             initial raw threshold to locate local maxima (3x3)
            split_th:           secondary threshold to find high probability but not local maximum pixels         
        """
        with torch.no_grad():
            
            # accumulate probability values from the 4 adjacent pixels
            diag = 0.0
            filt = torch.tensor([[diag, 1., diag], [1., 1., 1.], [diag, 1., diag]]).unsqueeze(0).unsqueeze(0).to(p.device)
            conv = torch.nn.functional.conv2d(p.unsqueeze(1), filt, padding=1)

            # localize the local maximum (3x3) above the raw threshold
            p_clip = torch.where(p > raw_th, p, torch.zeros_like(p))
            pool = torch.nn.functional.max_pool2d(p_clip.unsqueeze(1), kernel_size=3, stride=1, padding=1)
            max_mask1 = torch.eq(p.unsqueeze(1), pool)
            p_ps1 = max_mask1.float() * conv

            # looking for probability pixels that are not local maximum but still shows high probability (fluorophore in adjacent pixels)
            p_copy = p.unsqueeze(1).clone()
            p_copy *= 1.0 - max_mask1.float()
            max_mask2 = torch.where(p_copy > split_th, torch.ones_like(p_copy), torch.zeros_like(p_copy))
            p_ps2 = max_mask2.float() * conv

            p_ps = p_aggregation(p_ps1, p_ps2)
            return p_ps.squeeze(1)
        
    def forward(self, y_out:torch.Tensor, indf_offset:int=None, id_offset:int=None) -> FrameLocSet:
        """
        Forward anything through the post-processing and return a FrameLocSet
        Args:
            y_out:          (N, C, H, W) network output
            indf_offset:    the offset of the indf for the current batch
        Returns:
            FrameLocSet:    the output FrameLocSet
        """
        y_out[:, 0] = self._nms(y_out[:, 0], self.p_aggregation, self.raw_th, self.split_th)
        return super().forward(y_out, indf_offset, id_offset)
