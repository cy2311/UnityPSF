from typing import Tuple, Union, NamedTuple
import warnings

import math
import torch
import numpy as np
import scipy.stats

from ..generic.localizations import FrameLocSet
from ..utils.helpers import set_device



class SegResult(NamedTuple):
    prec: float
    rec: float
    jac: float
    f1: float

class SegmentationEvaluation:
    """segementation evaluations"""
    
    def __init__(self):
        self._tp = None
        self._fp = None
        self._fn = None
        self._prec = None
        self._rec = None
        self._jac = None
        self._f1 = None

    def __str__(self):
        if self._tp is None or self._fp is None or self._fn is None:
            return "Segmentation evaluation unavailable. Run .forward(tp, fp, fn)"

        nloc_tar = len(self._tp) + len(self._fn)
        nloc_out = len(self._tp) + len(self._fp)

        str_repr = "Segmentation evaluation (cached values)\n"
        str_repr += f"Number of target localizations: {nloc_tar} predicted emitters: {nloc_out}\n"
        str_repr += f"Number of TP: {len(self._tp)} FP: {len(self._fp)} FN: {len(self._fn)}\n"
        str_repr += f"Jacquard: {self._jac:.3f}\n"
        str_repr += f"F1Score: {self._f1:.3f}\n"
        str_repr += f"Precision: {self._prec:.3f}, Recall: {self._rec:.3f}\n"
        return str_repr

    def forward(self, tp:FrameLocSet, fp:FrameLocSet, fn:FrameLocSet) -> SegResult:
        """
        Forward Localizations through evaluation.
        Args:
            tp: true positives
            fp: false positives
            fn: false negatives
        Returns:
            NamedTuple: SegResult
        """
        nTP = float(len(tp))
        nFP = float(len(fp))
        nFN = float(len(fn))

        prec = math.nan if (nTP + nFP) == 0 else nTP / (nTP + nFP)
        rec = math.nan if (nTP + nFN) == 0 else nTP / (nTP + nFN)
        jac = math.nan if (nTP + nFN + nFP) == 0 else nTP / (nTP + nFN + nFP)
        f1 = math.nan if (prec + rec) == 0 else (2.0 * prec * rec) / (prec + rec)

        # Store last result to cache
        self._tp, self._fp, self._fn = tp, fp, fn
        self._prec, self._rec, self._jac, self._f1 = prec, rec, jac, f1
        return SegResult(prec=self._prec, rec=self._rec, jac=self._jac, f1=self._f1)



class DistanceResult(NamedTuple):
    rmse_lat: float
    rmse_ax: float
    rmse_vol: float
    mad_lat: float
    mad_ax: float
    mad_vol: float

class DistanceEvaluation:
    """distance evaluations"""

    def __init__(self, cam_pxszs:Tuple[float, float, float]):
        self.cam_pxszs = cam_pxszs
        self._rmse_lat = None
        self._rmse_ax = None
        self._rmse_vol = None
        self._mad_lat = None
        self._mad_ax = None
        self._mad_vol = None

    def __str__(self):
        if self._rmse_lat is None:
            return "Distance Evaluation unavailable. Run .forward(tp, tp_match)."

        str_repr = "Distance Evaluation (cached values)\n"
        str_repr += f"RMSE: Lat. {self._rmse_lat:.3f} Axial. {self._rmse_ax:.3f} Vol. {self._rmse_vol:.3f}\n" if self.ndim == 3 else  f"RMSE: Lat. {self._rmse_lat:.3f}\n"
        str_repr += f"MAD: Lat: {self._mad_lat:.3f} Axial: {self._mad_ax:.3f} Vol: {self._mad_vol:.3f}\n" if self.ndim == 3 else  f"MAD: Lat. {self._mad_lat:.3f}\n"
        return str_repr

    @staticmethod
    def _rmse_mad_dist2D(xy_0:torch.Tensor, xy_1:torch.Tensor) -> Tuple[float, float]:
        """
        Calculates RMSE and mean absolute distance
        Args:
            xy_0:       (nlocs_0, 2) coordinates of set 0 as the running coordiantes (in nm)
            xy_1:       (nlocs_1, 2) coordinates of set 1 as the ground truth coordinates (in nm)
        Returns:
            rmse_lat:   RMSE lateral
            mad_lat:    Mean Absolute Distance lateral
        """
        rmse_lat = ((xy_0 - xy_1)**2).sum(1).mean().sqrt().cpu().item()
        mad_lat = (xy_0 - xy_1).abs().sum(1).mean(0).cpu().item()
        return rmse_lat, mad_lat
    
    @staticmethod
    def _rmse_mad_distZ(z_0:torch.Tensor, z_1:torch.Tensor) -> Tuple[float, float]:
        """
        Calculates RMSE and mean absolute distance
        Args:
            z_0:        (nlocs_0,) coordinates of set 0 as the running coordiantes (in nm)
            z_1:        (nlocs_1,) coordinates of set 1 as the ground truth coordinates (in nm)
        Returns:
            rmse_z:     RMSE axial
            mad_z:      Mean Absolute Distance axial
        """
        rmse_z = ((z_0 - z_1)**2).mean().sqrt().cpu().item()
        mad_z = (z_0 - z_1).abs().mean().cpu().item()
        return rmse_z, mad_z
        
    @staticmethod
    def _rmse_mad_dist3D(xyz_0:torch.Tensor, xyz_1:torch.Tensor) -> Tuple[float, float, float, float, float, float]:
        """
        Calculates RMSE and mean absolute distance
        Args:
            xyz_0:      (nlocs_0, 3) coordinates of set 0 as the running coordiantes (in nm)
            xyz_1:      (nlocs_1, 3) coordinates of set 1 as the ground truth coordinates (in nm)
        Returns:
            rmse_lat:   RMSE lateral
            rmse_ax:    RMSE axial
            rmse_vol:   RMSE volumetric
            mad_lat:    Mean Absolute Distance lateral
            mad_ax:     Mean Absolute Distance axial
            mad_vol:    Mean Absolute Distance volumetric
        """
        
        se_lat = ((xyz_0[:, :2] - xyz_1[:, :2])**2).sum(1)
        se_ax = ((xyz_0[:, 2] - xyz_1[:, 2])**2)
        se_vol = se_lat + se_ax
        
        rmse_lat = se_lat.mean().sqrt().cpu().item()
        rmse_ax = se_ax.mean().sqrt().cpu().item()
        rmse_vol = se_vol.mean().sqrt().cpu().item()

        ad_lat = (xyz_0[:, :2] - xyz_1[:, :2]).abs().sum(1)
        ad_ax = (xyz_0[:, 2] - xyz_1[:, 2]).abs()
        ad_vol = ad_lat + ad_ax

        mad_lat = ad_lat.mean().cpu().item()
        mad_ax = ad_ax.mean().cpu().item()
        mad_vol = ad_vol.mean().cpu().item()    

        return rmse_lat, rmse_ax, rmse_vol, mad_lat, mad_ax, mad_vol
    
    def forward(self, tp:FrameLocSet, tp_match:FrameLocSet, device:Union[torch.device, str]='cpu') -> DistanceResult:
        """
        Args:
            tp:         true positives
            tp_match:   matching ground truths
        Returns:
            NamedTuple: DistanceResult
        """
        if len(tp) == 0 or len(tp_match) == 0:
            return DistanceResult(rmse_lat=float('inf'), rmse_ax=float('inf'), rmse_vol=float('inf'),
                                  mad_lat=float('inf'), mad_ax=float('inf'), mad_vol=float('inf'))
        
        if not tp.meta_isclose(tp_match):
            raise ValueError("the input tp and tp_match are from different loc_meta")
        if len(tp) != len(tp_match):
            raise ValueError(f"Size of true positives ({len(tp)}) does not match size of reference ({len(tp_match)}).")
        
        _device = set_device(device)
        _f_type = tp.f_type
        _cam_pxszs = torch.tensor(self.cam_pxszs[:tp.ndim], dtype=_f_type, device=_device)

        xyz_0_nm = tp.xyz.to(_device) * _cam_pxszs[None, :]
        xyz_1_nm = tp_match.xyz.to(_device) * _cam_pxszs[None, :]
        if tp.ndim == 2:
            rmse_lat, mad_lat = self._rmse_mad_dist2D(xyz_0_nm, xyz_1_nm)
            # store the last result to cache
            self._rmse_lat = rmse_lat
            self._mad_lat = mad_lat
        else:
            rmse_lat, rmse_ax, rmse_vol, mad_lat, mad_ax, mad_vol = self._rmse_mad_dist3D(xyz_0_nm, xyz_1_nm)
            # store the last result to cache
            self._rmse_lat, self._rmse_ax, self._rmse_vol = rmse_lat, rmse_ax, rmse_vol
            self._mad_lat, self._mad_ax, self._mad_vol = mad_lat, mad_ax, mad_vol

        return DistanceResult(rmse_lat=self._rmse_lat, rmse_ax=self._rmse_ax, rmse_vol=self._rmse_vol,
                              mad_lat=self._mad_lat, mad_ax=self._mad_ax, mad_vol=self._mad_vol) 
    


class ErrorResult(NamedTuple):
    dxyz_mu_sig: np.ndarray     # (ndim, 2)
    dphot_mu_sig: np.ndarray    # (2,)

class ErrorEvaluation:
    """error evaluations"""
    
    _weighting_methods = ('phot', 'sig')
    _reduction_methods = ('mstd', 'gaussian')
    
    def __init__(self, cam_pxszs:Tuple[float, float, float], reduction:str='mstd', weighting:str=None):

        if reduction not in self._reduction_methods:
            raise ValueError(f"Reduction method {reduction} not implemented. Available methods are {self._reduction_methods}.")
        if weighting is not None and weighting not in self._weighting_methods:
            raise ValueError(f"Weighting method {weighting} not implemented. Available methods are {self._weighting_methods} or None")
        self.weighting = weighting
        self.reduction = reduction
        self.cam_pxszs = cam_pxszs
    
    @staticmethod
    def _norm_fit_nan(input_data:torch.Tensor) -> np.ndarray:
        """
        Fit a 1-d tensor using scipy.stats.norm.fit()
        Args:
            input_data: (N,) 1d tensor to fit
        Returns:
            np.ndarray: (2,) [mu, sig] of the fitted normal distribution
        """
        try:
            out = scipy.stats.norm.fit(input_data.cpu().numpy())
        except RuntimeError:
            warnings.warn("Non-Finite values encountered during fitting.")
            out = [float('nan')] * 2
        return np.array(out)

    def _reduce(self, dxyz:torch.Tensor, dphot:torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        """
        Reduce the weighted errors as by the specified method.
        Args:
            dxyz:       (N, 2 or 3) weighted err in xyz
            dphot:      (N,) weighted err in phot
        Returns:
            np.ndarray: (2 or 3, 2) the mean and std of dx, dy, and dz
            np.ndarray: (2,) the mean and std of dphot
        """
        if self.reduction == 'gaussian':
            dxyz_mu_sig = np.vstack([self._norm_fit_nan(dxyz[:, i]) for i in range(dxyz.shape[1])])
            dphot_mu_sig = self._norm_fit_nan(dphot)
        else:
            dxyz_mu_sig = torch.stack([dxyz.mean(0), dxyz.std(0)], dim=1).cpu().numpy()
            dphot_mu_sig = torch.stack([dphot.mean(), dphot.std()], dim=0).cpu().numpy()
        return dxyz_mu_sig, dphot_mu_sig
  
    def forward(self, tp:FrameLocSet, tp_match:FrameLocSet, device:Union[torch.device, str]='cpu') -> ErrorResult:
        """
        Args:
            tp:         true positives
            tp_match:   matching ground truth
            device:     device for computation
        Returns:
            NamedTuple: ErrorResult
        """
        if len(tp) == 0 or len(tp_match) == 0:
            dxyz_mu_sig = np.zeros((3, 2)) + float('inf')
            dphot_mu_sig = np.zeros((2,)) + float('inf')
            return ErrorResult(dxyz_mu_sig=dxyz_mu_sig, dphot_mu_sig=dphot_mu_sig)

        if not tp.meta_isclose(tp_match):
            raise ValueError("the input tp and tp_match are from different loc_meta")
        if len(tp) != len(tp_match):
            raise ValueError(f"Size of true positives ({len(tp)}) does not match size of reference ({len(tp_match)}).")
        
        _device = set_device(device)
        _f_type = tp.f_type
        _cam_pxszs = torch.tensor(self.cam_pxszs[:tp.ndim], dtype=_f_type, device=_device)

        dxyz = (tp.xyz.to(_device) - tp_match.xyz.to(_device)) * _cam_pxszs[None, :]
        dphot = tp.phot.to(_device) - tp_match.phot.to(_device)

        if self.weighting == 'phot':
            _phot = tp.phot.to(_device)
            weight_xyz = _phot.unsqueeze(1).sqrt()
            weight_phot = 1.0 / _phot.sqrt()
    
        elif self.weighting == 'sig':
            _xyz_sig = tp.xyz_sig.to(_device) * _cam_pxszs[None, :]
            _phot_sig = tp.phot_sig.to(_device)
            weight_xyz = 1.0 / _xyz_sig
            weight_phot = 1.0 / _phot_sig
        
        else: # no weighting
            weight_xyz = 1.0
            weight_phot = 1.0

        dxyz_w = dxyz * weight_xyz
        dphot_w = dphot * weight_phot
        dxyz_mu_sig, dphot_mu_sig = self._reduce(dxyz_w, dphot_w)

        return ErrorResult(dxyz_mu_sig=dxyz_mu_sig, dphot_mu_sig=dphot_mu_sig)



class SMLMEvalResult(NamedTuple):
    prec: float
    rec: float
    jac: float
    f1: float
    effcy_lat: float
    effcy_ax: float
    effcy_vol: float
    rmse_lat: float
    rmse_ax: float
    rmse_vol: float
    mad_lat: float
    mad_ax: float
    mad_vol: float
    dx_mu: float
    dx_sig: float
    dy_mu: float
    dy_sig: float
    dz_mu: float
    dz_sig: float
    dphot_mu: float
    dphot_sig: float

class SMLMEvaluation:
    """Wrapper class to combine all evaluations into one"""
    alpha_lat = 1
    alpha_ax = 0.5 
    
    def __init__(self, cam_pxszs:Tuple[float, float, float], weighting:str='sig', reduction:str='gaussian'):
        
        self.seg_eval = SegmentationEvaluation()
        self.dist_eval = DistanceEvaluation(cam_pxszs)
        self.werr_eval = ErrorEvaluation(cam_pxszs, reduction=reduction, weighting=weighting)

        self.prec = None
        self.rec = None
        self.jac = None
        self.f1 = None

        self.rmse_vol = None
        self.rmse_lat = None
        self.rmse_ax = None
        self.mad_vol = None
        self.mad_lat = None
        self.mad_ax = None

    @property
    def effcy_lat(self):
        return (100 - math.sqrt((100 * (1.0 - self.jac))**2 + self.alpha_lat**2 * self.rmse_lat**2)) / 100

    @property
    def effcy_ax(self):
        return (100 - math.sqrt((100 * (1.0 - self.jac))**2 + self.alpha_ax**2 * self.rmse_ax**2)) / 100

    @property
    def effcy_vol(self):
        return 0.5 * (self.effcy_lat + self.effcy_ax)

    def __str__(self):
        repstr = "------------------------ Evaluation Set ------------------------\n"
        repstr += f"Precision {self.prec}\n"
        repstr += f"Recall {self.rec}\n"
        repstr += f"Jaccard {self.jac}\n"
        repstr += f"F1Score {self.f1}\n"
        repstr += f"RMSE lat. {self.rmse_lat}\n"
        repstr += f"RMSE ax. {self.rmse_ax}\n"
        repstr += f"RMSE vol. {self.rmse_vol}\n"
        repstr += f"MAD lat. {self.mad_lat}\n"
        repstr += f"MAD ax. {self.mad_ax}\n"
        repstr += f"MAD vol. {self.mad_vol}\n"
        repstr += f"Efficiency lat. {self.effcy_lat}\n"
        repstr += f"Efficiency ax. {self.effcy_ax}\n"
        repstr += f"Efficiency vol. {self.effcy_vol}\n"
        repstr += "-----------------------------------------------------------------"
        return repstr

    def forward(self, tp:FrameLocSet, fp:FrameLocSet, fn:FrameLocSet, tp_match:FrameLocSet) -> SMLMEvalResult:
        """
        Evaluate sets of emitters by all available metrics.
        Args:
            tp:         true positives
            fp:         false positives
            fn:         false negatives
            tp_match:   matching ground truths
        Returns:
            NamedTuple: SMLMEvalResult
        """
        seg_out = self.seg_eval.forward(tp, fp, fn)
        dist_out = self.dist_eval.forward(tp, tp_match)
        werr_out = self.werr_eval.forward(tp, tp_match)

        self.prec, self.rec, self.jac, self.f1 = seg_out.prec, seg_out.rec, seg_out.jac, seg_out.f1

        self.rmse_lat = dist_out.rmse_lat
        self.rmse_ax = dist_out.rmse_ax
        self.rmse_vol = dist_out.rmse_vol

        self.mad_lat = dist_out.mad_lat
        self.mad_ax = dist_out.mad_ax
        self.mad_vol = dist_out.mad_vol

        dx_mu_sig = werr_out.dxyz_mu_sig[0].tolist()
        dy_mu_sig = werr_out.dxyz_mu_sig[1].tolist()
        dz_mu_sig = werr_out.dxyz_mu_sig[2].tolist() if tp.ndim == 3 else [None] * 2

        return SMLMEvalResult(prec=self.prec, rec=self.rec, jac=self.jac, f1=self.f1,
                              effcy_lat=self.effcy_lat, effcy_ax=self.effcy_ax, effcy_vol=self.effcy_vol,
                              rmse_lat=self.rmse_lat, rmse_ax=self.rmse_ax, rmse_vol=self.rmse_vol,
                              mad_lat=self.mad_lat, mad_ax=self.mad_ax, mad_vol=self.mad_vol,
                              dx_mu=dx_mu_sig[0], dx_sig=dx_mu_sig[1],
                              dy_mu=dy_mu_sig[0], dy_sig=dy_mu_sig[1],
                              dz_mu=dz_mu_sig[0], dz_sig=dz_mu_sig[1],
                              dphot_mu=werr_out.dphot_mu_sig[0].item(), dphot_sig=werr_out.dphot_mu_sig[1].item())