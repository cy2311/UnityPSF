from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Tuple, Union

import math
import torch
import numpy as np
import scipy.stats

from matplotlib import pyplot as plt

from ..utils.helpers import to_tensor



class ProgressCheck(ABC):

    def __call__(self, *args, **kwargs) -> bool:
        return self.check_progress(*args, **kwargs)
    
    @abstractmethod
    def check_progress(self, *args, **kwargs) -> bool:
        """Returns ture when covergence seems okay"""
        raise NotImplementedError
    


class NoCheck(ProgressCheck):

    def check_progress(self) -> bool:
        return True
    


class GMMHeuristicCheck(ProgressCheck):

    def __init__(self, nLocs_avg:float, threshold:float=100.0, ref_epoch:int=1):
        """
        Validates the training progress by some heuristics.
        Args:
            nlocs_avg:  Expected number of localizations per frame
            threshold:  maximum loss per loc after reference epoch
            ref_epoch:  reference epoch
        """
        super().__init__()
        self.nlocs_avg = nLocs_avg
        self.threshold = threshold
        self.ref_epoch = ref_epoch
        self._prev_converged = None

    def check_progress(self, loss:float, epoch:int) -> bool:

        if self._prev_converged is not None:
            return self._prev_converged
        
        if epoch != self.ref_epoch:
            return True
        else:
            if loss / self.nlocs_avg >= self.threshold:
                self._prev_converged = False
                return False
            else:
                self._prev_converged = True
                return True
            


class MetricMeter:
    """Computes and stores the average and current metric values"""

    def __init__(self, vals:torch.Tensor=torch.zeros((0,)), reduce_nan:bool=True):
        self.val = None
        self.vals = vals
        self.reduce_nan = reduce_nan

    @property
    def count(self):
        return self.vals.numel()
    
    @property
    def mean(self):
        return self.vals.mean().item() if isinstance(self.vals, torch.Tensor) else to_tensor(self.vals).mean().item()
    
    @property
    def std(self):
        return self.vals.std().item() if isinstance(self.vals, torch.Tensor) else to_tensor(self.vals).std().item()
    
    def __str__(self):
        if self.count >= 2:
            return f'{self.mean:.3f} +/- {self.std:.3f}'
        else:
            return f'{self.mean:.3f}'

    def __neg__(self):
        m = MetricMeter(reduce_nan=self.reduce_nan)
        m.vals = -self.vals
        return m
    
    def __add__(self, other:Union[MetricMeter, torch.Tensor]) -> MetricMeter:
        m = MetricMeter(reduce_nan=self.reduce_nan)
        if isinstance(other, MetricMeter):
            m.vals = self.vals + other.vals
        else:
            m.vals = self.vals + other
        return m
    
    def __radd__(self, other:Union[MetricMeter, torch.Tensor]) -> MetricMeter:
        return self + other
    
    def __sub__(self, other:Union[MetricMeter, torch.Tensor]) -> MetricMeter:
        return -other + self
    
    def __rsub__(self, other:Union[MetricMeter, torch.Tensor]) -> MetricMeter:
        return -self + other
    
    def __mul__(self, other:Union[MetricMeter, torch.Tensor]) -> MetricMeter:
        m = MetricMeter(reduce_nan=self.reduce_nan)
        if isinstance(other, MetricMeter):
            m.vals = self.vals * other.vals
        else:
            m.vals = self.vals * other
        return m
    
    def __rmul__(self, other:Union[MetricMeter, torch.Tensor]) -> MetricMeter:
        return self * other
    
    def __truediv__(self, other:Union[MetricMeter, torch.Tensor]) -> MetricMeter:
        m = MetricMeter(reduce_nan=self.reduce_nan)
        if isinstance(other, MetricMeter):
            m.vals = self.vals / other.vals
        else:
            m.vals = self.vals / other
        return m
    
    def __pow__(self, other:Union[MetricMeter, torch.Tensor]) -> MetricMeter:
        m = MetricMeter(reduce_nan=self.reduce_nan)
        if isinstance(other, MetricMeter):
            raise ValueError('Power not implemented for both operands being MetricMeter')
        else:
            m.vals = self.vals ** other
        return m

    def reset(self):
        """Resets the instance"""
        self.val = None
        self.vals = torch.zeros((0,))

    def update(self, val:Union[float, torch.Tensor]):
        """Updates the instance"""
        if isinstance(val, torch.Tensor) and val.numel() == 1:
            val = val.detach().cpu().item()
        val = float(val)
        if math.isnan(val) and self.reduce_nan:
            return
        self.val = val
        self.vals = torch.cat((self.vals, to_tensor([val], dtype=self.vals.dtype, device=self.vals.device)), dim=0)
    
    def mmhist(self, range_hist:Tuple[float, float]=None, nbins:int=50, fit=scipy.stats.norm):
        """
        Args:
            range_hist:     bin_range for histogram
            nbins:          number of bins
            fit:            fitting method
        Returns:
            matplotlib.figure
        """
        if self.vals.numel() <= 10:
            fig = plt.figure(figsize=(10, 8))
            return fig
        elif self.vals.unique().numel() == 1:
            fig = plt.figure(figsize=(10, 8))
            plt.vlines(self.vals[0], 0, 1)
            return fig
        
        if range_hist is not None:
            bins_ = np.linspace(*range_hist, num=nbins+1, endpoint=True)
        else:
            range_hist = np.quantile(self.vals, [0.01, 0.99])
            bins_ = np.linspace(*range_hist, num=nbins+1, endpoint=True)
        binw = bins_[1] - bins_[0]
        binc = 0.5 * (bins_[:-1] + bins_[1:])

        vals = self.vals.view(-1).numpy()
        vals_hist = np.histogram(vals, bins=bins_)[0]
        vals_mu, vals_sig = scipy.stats.norm.fit(vals_hist)
        vals_histfit = scipy.stats.norm.pdf(binc, loc=vals_mu, scale=vals_sig)
        
        fig, (ax_box, ax_hist) = plt.subplots(nrows=2, sharex=True, gridspec_kw={'height_ratios':(0.1, 0.9)}, figsize=(8.5, 11))
        ax_box.boxplot(vals, orientation='horizontal')
        ax_box.tick_params(left=False, labelleft=False, labelbottom=False)
        ax_box.tick_params(axis='x', which='both', direction='in')
        ax_hist.bar(bins_[:-1], vals_hist, width=binw, align='edge')
        ax_hist.plot(binc, vals_histfit)
        ax_hist.tick_params(axis='both', which='both', direction='in')

        return fig