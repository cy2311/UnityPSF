import warnings
from abc import ABC, abstractmethod
from typing import Tuple, Union, NamedTuple
import time

import torch

from ..generic.localizations import FrameLocSet
from ..utils.helpers import set_device, to_tensor



class MatchResult(NamedTuple):
    TP: FrameLocSet
    FP: FrameLocSet
    FN: FrameLocSet
    TP_match: FrameLocSet



class LocMatcher(ABC):

    def __init__(self, *, match_dims:int, cam_pxszs:Tuple[float, float, float], 
                 dist_tolr:float=None, dist_tolz:float=None, dist_tolv:float=None):
        """
        Args:
            match_dims:     match in 2D or 3D
            cam_pxszs:      [cam_pxszx, cam_pxszy, z_step] in nm
            dist_tolr:      lateral tolerance distance in nm
            dist_tolz:      axial tolerance distance in nm
            dist_tolv:      volumetric tolerance distance in nm
        """
        # Sanity checks
        if match_dims not in (2, 3):
            raise ValueError("Not supported match dimensionality.")

        if dist_tolr is not None and dist_tolz is not None and dist_tolv is not None:
            warnings.warn("You specified a lateral, axial and volumetric threshold. "
                          "While this is allowed; are you sure?")

        if dist_tolr is None and dist_tolz is None and dist_tolv is None:
            warnings.warn("You specified neither a lateral, axial nor volumetric threshold. Are you sure about this?")

        self.match_dims = match_dims
        self.cam_pxszs = to_tensor(cam_pxszs, dtype=torch.float32)
        self.dist_tolr = dist_tolr
        self.dist_tolz = dist_tolz
        self.dist_tolv = dist_tolv

    @abstractmethod
    def forward(self, loc_out:FrameLocSet, loc_tar:FrameLocSet) -> MatchResult:
        """
        Args:
            loc_out:    the evaluating FrameLocSet
            loc_tar:    the reference FrameLocSet
        Returns:
            NamedTuple: MatchResult
        """
        raise NotImplementedError
    


class GreedyHungarianMatching(LocMatcher):
    """Matching emitters in a greedy 'hungarian' fasion via best-first-search"""

    def __init__(self, *, match_dims:int, cam_pxszs:Tuple[float, float, float],
                 dist_tolr:float=None, dist_tolz:float=None, dist_tolv:float=None):
        super().__init__(match_dims=match_dims, cam_pxszs=cam_pxszs, dist_tolr=dist_tolr, dist_tolz=dist_tolz, dist_tolv=dist_tolv)

    def _filter(self, xyz_out:torch.Tensor, xyz_tar:torch.Tensor) -> torch.BoolTensor:
        """
        Filter kernel to rule out unwanted matches
        Args:
            xyz_out:        ((B, )N, 3) output coordinates in nm
            xyz_tar:        ((B, )M, 3) reference coordinates in nm
        Returns:
            filter_mask:    ((B, )N, M) matching map
        """
        if xyz_out.dim() == 3:
            if xyz_out.size(0) != xyz_tar.size(0):
                raise ValueError(f"number of batches mismatch: xyz_out.size(0)={xyz_out.size(0)}, xyz_tar.size(0)={xyz_tar.size(0)}")
            _to_squeeze = False
        else:
            xyz_out = xyz_out.unsqueeze(0)
            xyz_tar = xyz_tar.unsqueeze(0)
            _to_squeeze = True
        
        filter_mask = torch.full((xyz_out.size(0), xyz_out.size(1), xyz_tar.size(1)), fill_value=True)
        
        if self.dist_tolr is not None:
            dist_mat = torch.cdist(xyz_out[:, :, :2], xyz_tar[:, :, :2], p=2)
            filter_mask[dist_mat > self.dist_tolr] = False
        
        if self.dist_tolz is not None:
            dist_mat = torch.cdist(xyz_out[:, :, [2]], xyz_tar[:, :, [2]], p=2)
            filter_mask[dist_mat > self.dist_tolz] = False
        
        if self.dist_tolv is not None:
            dist_mat = torch.cdist(xyz_out, xyz_tar, p=2)
            filter_mask[dist_mat > self.dist_tolv] = False
        
        if _to_squeeze:
            filter_mask = filter_mask.squeeze(0)
        return filter_mask
    
    @staticmethod
    def _rule_out_kernel(dists:torch.Tensor) -> Tuple[torch.IntTensor, torch.IntTensor]:
        """
        Hungarian mathching
        Args:
            dists:  the distance map
        Returns:
            torch.IntTensor:    index of the target coordinates for each output coordinates
            torch.IntTensor:    index of the output coordinates for each target coordinates
        """
        if dists.dim() != 2:
            raise ValueError(f"ndim mismatch: dists.dim() should be 2, but got {dists.dim()}")
        
        if dists.numel() == 0:
            return torch.zeros((0,)).int(), torch.zeros((0,)).int()
        
        dists_ = dists.clone()
        match_list = []
        while not (dists_ == float('inf')).all():
            #idx = torch.unravel_index(dists_.argmin(), dists_.shape)
            idx = dists_.argmin() // dists.size(1), dists_.argmin() % dists.size(1)
            dists_[idx[0]] = float('inf')
            dists_[:, idx[1]] = float('inf')
            match_list.append(idx)

        if match_list.__len__() >= 1:
            assignment = torch.tensor(match_list, dtype=torch.int32)
        else:
            assignment = torch.zeros((0, 2), dtype=torch.int32)
        return assignment[:, 0], assignment[:, 1]
    
    def _match_kernel(self, xyz_out:torch.Tensor, xyz_tar:torch.Tensor, filter_mask:torch.BoolTensor) \
                    -> Tuple[torch.IntTensor, torch.IntTensor, torch.BoolTensor, torch.BoolTensor]:
        """
        Filter kernel to rule out unwanted matches. No batch implementation currently
        Args:
            xyz_out:        (N, 3) output coordinates in nm
            xyz_tar:        (M, 3) reference coordinates in nm
            filter_mask:    (N, M) matching map
        Returns:
            torch.IntTensor:    index of the target coordinates for each output coordinates
            torch.IntTensor:    index of the output coordinates for each target coordinates
            torch.BoolTensor:   (N,) bool index labeling the matched output coordinates
            torch.BoolTensor:   (M,) bool index labeling the matched target coordinates
        """

        if self.match_dims == 2:
            dist_mat = torch.cdist(xyz_out[:, :2], xyz_tar[:, :2], p=2.)
        else:
            dist_mat = torch.cdist(xyz_out, xyz_tar, p=2.)
        dist_mat[~filter_mask] = float('inf')

        tp_idx, tp_match_idx = self._rule_out_kernel(dist_mat)
        
        tp_idx_bool = torch.full(size=(xyz_out.size(0),), fill_value=False)
        tp_idx_bool[tp_idx] = True
        tp_match_idx_bool = torch.full(size=(xyz_tar.size(0),), fill_value=False,)
        tp_match_idx_bool[tp_match_idx] = True

        return tp_idx, tp_match_idx, tp_idx_bool, tp_match_idx_bool

    def forward(self, loc_out:FrameLocSet, loc_tar:FrameLocSet, device:Union[torch.device, str]=None) -> MatchResult:
        """
        Args:
            loc_out:    the evaluating FrameLocSet
            loc_tar:    the reference FrameLocSet
        Returns:
            NamedTuple: MatchResult
        """
        if not loc_out.meta_isclose(loc_tar):
            raise ValueError("the loc_out and loc_tar comes from different loc_meta")
        _f_type = loc_out.f_type
        _device = set_device(device)
        _cam_pxszs = to_tensor(self.cam_pxszs, dtype=_f_type, device=_device)

        indf_union = torch.unique(torch.cat((loc_out.indf, loc_tar.indf), dim=0))

        TP_list = []
        FP_list = []
        FN_list = []
        TP_matched_list = []

        for f in indf_union:
            _loc_out = loc_out.get_subframe((f, f+1))
            _loc_tar = loc_tar.get_subframe((f, f+1))

            if len(_loc_out) == 0 and len(_loc_tar) == 0:
                continue
            if len(_loc_out) == 0 and len(_loc_tar) > 0:
                FN_list.append(_loc_tar)
                continue
            if len(_loc_out) > 0 and len(_loc_tar) == 0:
                FP_list.append(_loc_out)
                continue
            
            xyz_out = to_tensor(_loc_out.xyz, dtype=_f_type, device=_device) * _cam_pxszs[None, :]
            xyz_tar = to_tensor(_loc_tar.xyz, dtype=_f_type, device=_device) * _cam_pxszs[None, :]
            filter_mask = self._filter(xyz_out, xyz_tar)
            tp_idx, tp_match_idx, tp_idx_bool, tp_match_idx_bool = self._match_kernel(xyz_out, xyz_tar, filter_mask)
            TP_list.append(_loc_out[tp_idx])
            TP_matched_list.append(_loc_tar[tp_match_idx])
            FP_list.append(_loc_out[~tp_idx_bool])
            FN_list.append(_loc_tar[~tp_match_idx_bool])
        
        loc_TP = FrameLocSet(**loc_out.meta) if len(TP_list) == 0 else FrameLocSet.loc_cat(TP_list)
        loc_TP_matched = FrameLocSet(**loc_tar.meta) if len(TP_matched_list) == 0 else FrameLocSet.loc_cat(TP_matched_list)
        loc_FP = FrameLocSet(**loc_out.meta) if len(FP_list) == 0 else FrameLocSet.loc_cat(FP_list)
        loc_FN = FrameLocSet(**loc_tar.meta) if len(FN_list) == 0 else FrameLocSet.loc_cat(FN_list)

        # IDs of ground truth are copied to ture positives
        loc_TP_matched.id = torch.arange(len(loc_TP_matched)).type(loc_TP_matched.i_type)
        loc_TP.id = loc_TP_matched.id.type(loc_TP.i_type)
        return MatchResult(TP=loc_TP, FP=loc_FP, FN=loc_FN, TP_match=loc_TP_matched)

