from typing import Tuple, Union, Callable

import numpy as np
import torch

import matplotlib.figure as mfig
import matplotlib.axes as maxes
import matplotlib.figure as mfig
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as maxes3d
from matplotlib.collections import PathCollection

from scipy.stats import gaussian_kde
from scipy.linalg import cholesky

from ..generic.localizations import FrameLocSet, EmitterLocSet



class PlotFrame:
    """Plots a frame"""
    
    def __init__(self, frame:Union[torch.Tensor, np.ndarray], clim:Tuple[float, float]=None, plot_colorbar:bool=False):
        """
        Args:
            frame:          frame to be plotted
            clim:           clim values
            plot_colorbar:  whether to plot the colorbar
        """
        self.frame = frame.detach().squeeze().numpy() if torch.is_tensor(frame) else frame.squeeze()
        self.clim = clim
        self.plot_colorbar = plot_colorbar

    def plot(self, fig:mfig.Figure, ax:maxes.Axes) -> None:
        """
        plot the frame onto the matplotlib figure and axis
        Args:
            fig:    the matplotlib.figure.Figure object to plot the image on
            ax:     the matplotlib.axes.Axes object (associated with fig) to plot the image on
        """
        im = ax.imshow(self.frame, cmap='gray')
        
        if self.clim is not None:
            im.set_clim(*self.clim)
        if self.plot_colorbar:
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel('x')
        ax.set_ylabel('y')



class PlotCoordinates:
    
    tar_dict = {'marker':'o', 'mec':'tab:red', 'mfc':'none', 'cmap':'winter', 'alpha':0.5, 'label':'Target'}
    out_dict = {'marker':'x', 'mec':'tab:blue', 'mfc':'tab:blue', 'cmap':'viridis', 'alpha':0.5, 'label':'Output'}
    ini_dict = {'marker':'+', 'mec':'tab:green', 'mfc':'tab:green', 'cmap':'copper', 'alpha':0.5, 'label':'Init'}

    def __init__(self, *, loc_tar:Union[FrameLocSet, EmitterLocSet]=None, loc_out:Union[FrameLocSet, EmitterLocSet]=None, 
                 loc_ini:Union[FrameLocSet, EmitterLocSet]=None, match_lines:bool=False, adjust_to_frame:bool=False, cmap_photon:bool=True):
        """
        Plots points in 2D projection.
        Args:
            loc_tar:            the FrameLocSet of the target
            loc_out:            the FrameLocSet of extracted from the output
            loc_ini:            the FrameLocSet of the initialization
            match_lines:        whether to connect the matched point between the output and the target
            adjust_to_frame:    whether to deduct 0.5 to make the ploting adjust to the frame for imshow
            cmap_photon:        whether to use photon as the cmap 
        """
        self.match_lines = match_lines
        self.adjust_to_frame = adjust_to_frame
        self.cmap_photon = cmap_photon

        self.xyz_tar, self.phot_tar = self._adjust_coord_to_frame(loc_tar)
        self.xyz_out, self.phot_out = self._adjust_coord_to_frame(loc_out)
        self.xyz_ini, self.phot_ini = self._adjust_coord_to_frame(loc_ini)
        
    def _adjust_coord_to_frame(self, loc:Union[FrameLocSet, EmitterLocSet]) -> Tuple[np.ndarray, np.ndarray]:
        """Adjusts the xyz coordinates to plot on the corresponding frame"""
        
        if loc is None:
            return np.zeros((0, 2), dtype=np.float32), np.zeros((0,), dtype=np.float32)
        
        xyz = loc.xyz.clone().detach().numpy()
        phot = loc.phot.clone().detach().numpy()
        if self.adjust_to_frame:
            xyz[:, 0] -= loc.xextent[0] + 0.5
            xyz[:, 1] -= loc.yextent[0] + 0.5
        return xyz, phot
        
    def _plot(self, ax:maxes.Axes, xyz:np.ndarray, phot:np.ndarray, config:dict):
        """
        Plots 2D projection of the pos in colors encoded by photons
        Args:
            ax:                 axis to plot on
            xyz:                (N, 2 or 3) xyz positions
            phot:               (N,) the photons for each pos
            adjust_to_frame:    whether to deduct 0.5 to make the ploting adjust to the frame for imshow
            config:             dictionaty must contains keys 'cmap' or 'color', 'marker', and 'label'
        """
        if not self.cmap_photon or phot is None:
            ax.plot(xyz[:, 0], xyz[:, 1], ls='none', marker=config['marker'], mec=config['mec'], mfc=config['mfc'], label=config['label'])
        else:
            ax.scatter(xyz[:, 0], xyz[:, 1], c=phot, 
                       marker=config['marker'], cmap=config['cmap'], alpha=config['alpha'], label=config['label'])

    def plot(self, ax:maxes.Axes):
        """Plot on the given matplotlib.axes.Axes"""
        
        self._plot(ax, self.xyz_tar, self.phot_tar, self.tar_dict)    
        self._plot(ax, self.xyz_out, self.phot_out, self.out_dict)    
        self._plot(ax, self.xyz_ini, self.phot_ini, self.ini_dict)
                
        if self.match_lines:
            plot_connect_point(ax, self.xyz_tar, self.xyz_out, ndim=2)

        ax.set_aspect('equal', adjustable='box')
        if self.adjust_to_frame:
            ax_ylimits = ax.get_ylim()
            if ax_ylimits[0] <= ax_ylimits[1]:
                ax.set_ylim(ax_ylimits[::-1])  # invert the axis
    
        ax.set_xlabel('x')
        ax.set_ylabel('y') 



class PlotFrameCoord(PlotCoordinates, PlotFrame):

    def __init__(self, frame:Union[torch.Tensor, np.ndarray], loc_tar:Union[FrameLocSet, EmitterLocSet]=None, 
                 loc_out:Union[FrameLocSet, EmitterLocSet]=None, loc_ini:Union[FrameLocSet, EmitterLocSet]=None,
                 clim:Tuple[float, float]=None, match_lines:bool=False, plot_colorbar:bool=False, cmap_photon:bool=False):

        self.coord_plotter = PlotCoordinates(loc_tar=loc_tar, loc_out=loc_out, loc_ini=loc_ini, 
                                             match_lines=match_lines, adjust_to_frame=True, cmap_photon=cmap_photon)
        self.frame_plotter = PlotFrame(frame=frame, clim=clim, plot_colorbar=plot_colorbar)

    def plot(self, fig:mfig.Figure, ax:maxes.Axes):
        self.frame_plotter.plot(fig=fig, ax=ax)
        self.coord_plotter.plot(ax=ax)



class PlotCoordinates3D:
    
    _labels_default = ('Target', 'Output')

    def __init__(self, *, loc_tar:Union[FrameLocSet, EmitterLocSet]=None, loc_out:Union[FrameLocSet, EmitterLocSet]=None, 
                 match_lines:bool=False, labels:Tuple[str, str]=None):
        """
        Plots points in 3D projection.
        Args:
            loc_tar:            the FrameLocSet of the target
            loc_out:            the FrameLocSet of extracted from the output
            match_lines:        whether to connect the matched point between the output and the target
            labels:             labels of the plots 
        """
        self.match_lines = match_lines
        self.labels = labels if labels is not None else self._labels_default

        self.xyz_tar = loc_tar.xyz.numpy() if loc_tar is not None else np.zeros((0, 3), dtype=np.float32)
        self.xyz_out = loc_out.xyz.numpy() if loc_out is not None else np.zeros((0, 3), dtype=np.float32)
        

    def plot(self, ax:maxes3d.Axes3D):
        ax.scatter(self.xyz_tar[:, 0], self.xyz_tar[:, 1], self.xyz_tar[:, 2], c='tab:red', marker='o', label=self.labels[0])
        ax.scatter(self.xyz_out[:, 0], self.xyz_out[:, 1], self.xyz_out[:, 2], c='tab:blue', marker='^', label=self.labels[1])
        if self.match_lines:
            plot_connect_point(ax, self.xyz_tar, self.xyz_out, ndim=3)



class ErrorScatterPlot:
    """Scatter plot the errors"""
    
    def __init__(self, cam_pxszs:Tuple[float, float, float], residual:bool=False, sub_sample:int=1000, nan_inf_okay:bool=True, band_width:Union[str, float, Callable]=None):
        """
        Args:
            am_pxszs:      [cam_pxszx, cam_pxszy, z_step] in nm
            residulas:      Ture to plot the residuals
            sub_sample:     number of plots to display kde
            nan_inf_okay:   False to raise Errors when nan or inf is encountered
            band_width:     band_width option as in scipy.stats.gaussian_kde  
        """
        self.cam_pxszs = np.array(cam_pxszs, dtype=np.float32)
        self.residual = residual
        self.sub_sample = max(sub_sample, 1)
        self.nan_inf_okay = nan_inf_okay
        self.band_width = band_width

    def _kde_plot(self, ax:maxes.Axes, x:np.ndarray, y:np.ndarray) -> PathCollection:
        """
        plot the x, y on the given Axes with gaussian_kde colored scatters
        Args:
            ax:             matplotlib axes to plot on
            x:              (N,) the x-coordinates
            y:              (N,) the y-coordinates            
        Returns:
            PathCollection: scatter object associated with the input ax
        """
        
        if len(x) != len(y):
            raise ValueError(f"length mismatch: x.shape={x.shape}, y.shape={y.shape}")
        if len(x) <= 1:
            scatters = ax.scatter(x, y, c='k', s=10, edgecolors='none')
            return scatters
        
        sub_sample = max(min(self.sub_sample, len(x)), 1)
        
        rng = np.random.default_rng()
        idx_sample = np.arange(len(x))
        rng.shuffle(idx_sample)
        idx_in = idx_sample[:sub_sample]
        idx_out = idx_sample[sub_sample:]
        
        x_in, y_in = x[idx_in], y[idx_in]
        x_out, y_out = x[idx_out], y[idx_out]

        if sub_sample < len(x):
            ax.scatter(x_out, y_out, c='k', s=10, edgecolors='none')

        xy_in = np.concatenate((x_in[None, :], y_in[None, :]), axis=0)
        
        try:
            _kde = GaussianKde(xy_in, bw_method=self.band_width) if self.band_width is not None else GaussianKde(xy_in)
            p = _kde(xy_in)
            norm = plt.Normalize(vmin=p.min(), vmax=p.max())
            scatters = ax.scatter(x_in, y_in, c=p, s=10, edgecolors='none', cmap='RdBu_r', norm=norm) 
        except Exception as e:
            print(f"Error in gaussian_kde {e}")
            scatters = ax.scatter(x_in, y_in, c='r', s=10, edgecolors='none')
        return scatters

    def _plot_core(self, ax:maxes.Axes, x_gt:np.ndarray, x:np.ndarray) -> PathCollection:
        """
        plot the prediction or residuals against the groundtruth
        Args:
            ax:             matplotlib axes to plot on
            x_gt:           (N,) the ground truth on x-axis
            x:              (N,) the prediction on y-axis
        Returns:
            PathCollection: scatter object associated with the input ax
        """
        x, y = nan_inf_filter(x_gt, x, self.nan_inf_okay)
        if self.residual:
            y -= x
        scatters = self._kde_plot(ax, x, y)

        if self.residual:
            if len(x) > 1:
                ax.plot([x.min(), x.max()], [0, 0], 'green')
            ax.set_ylabel('residuals')
        else:
            if len(x) > 1:
                ax.plot([x.min(), x.max()], [x.min(), x.max()], 'green')
            ax.set_ylabel('prediction')
        ax.set_xlabel('ground truth')
        return scatters
    
    def plot_errors(self, tp:Union[FrameLocSet, EmitterLocSet], tp_match:Union[FrameLocSet, EmitterLocSet]) -> Tuple[mfig.Figure, maxes.Axes]:
        """
        plot the errors between the tp FrameLocSet and tp_match FrameLocSet
        Args:
            tp:         the true postives
            tp_match:   the groud truth of the tp
        Returns:
            fig:        matplotlib figure object
            axes:       array of matplotlib Axes
        """
        fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(7, 5))
        tp_match_xyz = tp_match.xyz.detach().cpu().numpy() * self.cam_pxszs[None, :tp_match.ndim]
        tp_xyz = tp.xyz.detach().cpu().numpy() * self.cam_pxszs[None, :tp.ndim]
        for i in range(tp.ndim):
            scatters_ = self._plot_core(axes[0, i], tp_match_xyz[:, i], tp_xyz[:, i])
        
        tp_match_phot = tp_match.phot.detach().cpu().numpy()
        tp_phot = tp.phot.detach().cpu().numpy()
        scatters_ = self._plot_core(axes[1, 0], tp_match_phot, tp_phot)
        return fig, axes



class ErrorHistPlot:
    """Histogram plot the errors"""
    
    def __init__(self, cam_pxszs:Tuple[float, float, float], SampleRate:int=100, nan_inf_okay:bool=True):
        """
        Args:
            cam_pxszs:      [cam_pxszx, cam_pxszy, z_step] in nm
            SampleRate:     number of bins to plot the histogram
            nan_inf_okay:   False to raise Errors when nan or inf is encountered 
        """
        self.cam_pxszs = np.array(cam_pxszs, dtype=np.float32)
        self.nbins = SampleRate
        self.nan_inf_okay = nan_inf_okay
        
    def _plot_core(self, ax:maxes.Axes, x_gt:np.ndarray, x:np.ndarray):
        """
        plot the histogram for the residuals
        Args:
            ax:             matplotlib axes to plot on
            x_gt:           (N,) the ground truth on x-axis
            x:              (N,) the prediction on y-axis
        """
        x, y = nan_inf_filter(x_gt, x, self.nan_inf_okay)
        res = y - x
        res = np.clip(res, a_min=np.quantile(res, 0.05), a_max=np.quantile(res, 0.95))
        ax.hist(res, bins=self.nbins, histtype='stepfilled')
        ax.set_xlabel('residuals')
  
    def plot_errors(self, tp:Union[FrameLocSet, EmitterLocSet], tp_match:Union[FrameLocSet, EmitterLocSet]):
        """
        plot the errors between the tp FrameLocSet and tp_match FrameLocSet
        Args:
            tp:         the true postives
            tp_match:   the groud truth of the tp
        """
        fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(7, 5))
        tp_match_xyz = tp_match.xyz.detach().cpu().numpy() * self.cam_pxszs[None, :tp_match.ndim]
        tp_xyz = tp.xyz.detach().cpu().numpy() * self.cam_pxszs[None, :tp.ndim]
        for i in range(tp.ndim):
            self._plot_core(axes[0, i], tp_match_xyz[:, i], tp_xyz[:, i])
        
        tp_match_phot = tp_match.phot.detach().cpu().numpy()
        tp_phot = tp.phot.detach().cpu().numpy()
        self._plot_core(axes[1, 0], tp_match_phot, tp_phot)
        return fig, axes
    


def plot_connect_point(ax:Union[maxes.Axes, maxes3d.Axes3D], xyz0:Union[torch.Tensor, np.ndarray], xyz1:Union[torch.Tensor, np.ndarray], ndim:int=2) -> None:
    """
    Plots the connecting lines between the set0 and set1 in 2D on a given matplotlib Axes.
    Args:
        ax:     axis where to plot
        xyz0:   (N, 2/3) coordinates of set0
        xyz1:   (N, 2/3) coordinates of set1
        ndim:   dimensions
    """
    if xyz0.shape != xyz1.shape:
        raise ValueError(f"set0 and set1 should be of the same shape, whereas set0.shape={xyz0.shape}, set1.shape={xyz1.shape}")
    if ndim not in {2, 3}:
        raise ValueError("only supports ndim = 2 or 3")
    if ndim > xyz0.shape[1]:
        raise ValueError(f"set0.shape[1]={xyz0.shape[1]}, does not support ndim={ndim}")
    
    if xyz0.shape[1] > 0:
        if ndim == 3:
            for i in range(xyz0.shape[0]):
                ax.plot3D([xyz0[i, 0], xyz1[i, 0]], [xyz0[i, 1], xyz1[i, 1]], [xyz0[i, 2], xyz1[i, 2]], 'orange')
        else:
            for i in range(xyz0.shape[0]):
                ax.plot([xyz0[i, 0], xyz1[i, 0]], [xyz0[i, 1], xyz1[i, 1]], 'orange')



def nan_inf_filter(x_gt:np.ndarray, x:np.ndarray, nan_inf_okay:bool=False) -> Tuple[np.ndarray, np.ndarray]:
    """
    filter out the nan and inf in the input data
    Args:
        x:              (N,) the predicted coordinates
        x_gt:           (N,) the groundtruth coordinates
        nan_inf_okay:   False to raise Errors when nan or inf is encountered
    returns:
        np.ndarray:     filtered x_gt
        np.ndarray:     filtered x 
    """
    if len(x_gt) != len(x):
        raise ValueError(f"length mismatch: x_gt.shape={x_gt.shape}, x.shape={x.shape}")
    
    if (np.isnan(x_gt) | np.isinf(x_gt) | np.isnan(x) | np.isinf(x_gt)).any():
        if not nan_inf_okay:
            raise ValueError("Some of the data encountered NaN or Inf")
        idx = (~np.isnan(x_gt)) & (~np.isinf(x_gt)) & (~np.isnan(x)) & (~np.isinf(x))
        return x_gt[idx], x[idx]
    return x_gt, x



def plot_crosshair(ax:maxes.Axes, x:float, y:float, color:Union[str, tuple]='r') -> None:
    """
    Plot Crosshair and expands to axes limits
    Args:
        ax:     axis where to put crosshair
        x:      x coordinate
        y:      y coordinate
        color:  colour as specified by matplotlib
    """
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    ax.hlines(y, *xlim, colors=color)
    ax.vlines(x, *ylim, colors=color)



class GaussianKde(gaussian_kde):
    """
    Drop-in replacement for gaussian_kde that adds the class attribute EPSILON
    to the covmat eigenvalues, to prevent exceptions due to numerical error.
    """

    EPSILON = 1e-10

    def _compute_covariance(self):
        """ Computes the covariance matrix for each Gaussian kernel using covariance_factor() """
        
        # Cache covariance and Cholesky decomp of covariance
        self.factor = self.covariance_factor()
        
        if not hasattr(self, '_data_cho_cov'):
            self._data_covariance = np.atleast_2d(np.cov(self.dataset, rowvar=True, bias=False, aweights=self.weights))
            self._data_covariance += self.EPSILON * np.eye(len(self._data_covariance))
            self._data_cho_cov = cholesky(self._data_covariance, lower=True)

        self.covariance = self._data_covariance * self.factor**2
        self.cho_cov = (self._data_cho_cov * self.factor).astype(np.float64)
        self.log_det = 2 * np.log(np.diag(self.cho_cov * np.sqrt(2 * np.pi))).sum()