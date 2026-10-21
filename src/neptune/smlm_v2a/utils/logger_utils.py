from typing import List
import time

import torch
import torch.utils.tensorboard
import matplotlib.pyplot as plt

from ..generic.localizations import FrameLocSet

from .plotter import PlotFrameCoord, PlotCoordinates3D, ErrorScatterPlot, ErrorHistPlot



class SummaryWriter(torch.utils.tensorboard.SummaryWriter):
    """A simple customized tensorboard SummaryWritter"""
    
    def __init__(self, filter_keys=List[str], *args, **kwargs):
        """
        Args:
            filter_keys:    keys to be filtered in add_scalar_dict method
            *args:          args for tensorboard.SummaryWriter
            **kwargs:       kwargs for tensorboard.SummaryWritter
        """
        super().__init__(*args, **kwargs)
        self.filter_keys = filter_keys

    def add_scalar_dict(self, prefix:str, scalar_dict:dict, global_step:int, walltime=None):
        """
        Adds a couple of scalars that are in a dictionary (scalar_dict) to the summary.
        NOTE: this is different from 'add_scalars'
        """
        for name, value in scalar_dict.items():
            if name not in self.filter_keys:
                self.add_scalar(tag=prefix+name, scalar_value=value, global_step=global_step, walltime=walltime)



class NoLog(SummaryWriter):
    """The hardcoded No-Op of the tensorboard SummaryWritter."""

    def __init__(self, *args, **kwargs):
        return

    def add_scalar(self, *args, **kwargs):
        return

    def add_scalars(self, *args, **kwargs):
        return

    def add_scalar_dict(self, *args, **kwargs):
        return

    def add_histogram(self, *args, **kwargs):
        return

    def add_figure(self, tag, figure, *args, **kwargs):
        plt.close(figure)
        return

    def add_figures(self, *args, **kwargs):
        return

    def add_image(self, *args, **kwargs):
        return

    def add_images(self, *args, **kwargs):
        return

    def add_video(self, *args, **kwargs):
        return

    def add_audio(self, *args, **kwargs):
        return

    def add_text(self, *args, **kwargs):
        return

    def add_graph(self, *args, **kwargs):
        return

    def add_embedding(self, *args, **kwargs):
        return

    def add_pr_curve(self, *args, **kwargs):
        return

    def add_custom_scalars(self, *args, **kwargs):
        return

    def add_mesh(self, *args, **kwargs):
        return

    def add_hparams(self, *args, **kwargs):
        return
    


class DictLogger(NoLog):
    """
    Simple logger that logs scalars to a dictionary
    It inherits from NoLog so that it has the same method as the SummaryWriter has so that can be called via a multi_logger (see below)
    """
    def __init__(self):
        super().__init__()
        self.log_dict = {}

    def add_scalar_dict(self, prefix:str, scalar_dict:dict, global_step:int, walltime=None):
        for name, value in scalar_dict.items():
            self.add_scalar(prefix + name, value, global_step=global_step, walltime=walltime)

    def add_scalar(self, tag:str, scalar_value:float, global_step:int, walltime=None):
        if walltime is None:
            walltime = time.time()
        
        if tag not in self.log_dict:
            self.log_dict[tag] = {'scalar': [], 'step': [], 'walltime': []}
        
        self.log_dict[tag]['scalar'].append(scalar_value)
        self.log_dict[tag]['step'].append(global_step)
        self.log_dict[tag]['walltime'].append(walltime)



class MultiLogger:
    """
    A 'Meta-Logger' forwards method calls to all loggers in its logger list, in the order they are provided during initiation.
    Example:
        >>> multi_logger = MultiLogger([logger_0, logger_1,...]) 
        >>> multi_logger.add_scalar_dict(*args, **kwargs) will sequentially execute: logger_0.add_scalar_dict(*args, **kwargs), logger_1.add_scalar_dict(*args, **kwargs),...
    NOTE: all component loggers are assumed to have the same methods.
    """

    def __init__(self, loggers:List):
        """
        Args:
            loggers:    a list of loggers (e.g. [SummaryWritter, DictLogger])
        """
        self.loggers = loggers
        methods = [m for m in dir(self.loggers[0]) if callable(getattr(self.loggers[0], m)) and not m.startswith('__')]
        for method_name in methods:
            setattr(self, method_name, self._create_wrapper(method_name))

    def _create_wrapper(self, method_name:str):
        """Creates method that calls all loggers with given method name"""
        def wrapper(*args, **kwargs):
            return [getattr(logger, method_name)(*args, **kwargs) for logger in self.loggers]
        return wrapper



def log_frames(x:torch.Tensor, y_out:torch.Tensor, loc_out:FrameLocSet, loc_tar:FrameLocSet, loc_tp:FrameLocSet, loc_tp_match:FrameLocSet, 
               logger:MultiLogger, step:int, colorbar:bool=True):
    """
    Args:
        x:              (N, C=channels_in, H, W) the input frames
        y_out:          (N, C=10, H, W) the output from the nn 
        loc_out:        the output FrameLocSet
        loc_tar:        the target FrameLocSet
        loc_tp:         the true positive FrameLocSet
        loc_tp_match:   the ground truth matches to the ground truth
        logger:         the logger instance log values to both the tensorboard and a dictionary
        step:           the epoch counter
        colorbar:       bool, True to add colorbar
    """
    
    if x.dim() != 4:
        raise ValueError(f"The dimension of the input should be 4 but got x.dim={x.dim()}")
    r_ix = torch.randint(0, len(x), (1,)).int().item()
    
    x = x[r_ix]
    y_out = y_out[r_ix]
    
    loc_tar = loc_tar.get_subframe((r_ix, r_ix+1))
    loc_out = loc_out.get_subframe((r_ix, r_ix+1))
    loc_tp = loc_tp.get_subframe((r_ix, r_ix+1))
    loc_tp_match = loc_tp_match.get_subframe((r_ix, r_ix+1))

    # loop over all input channels
    for i, xc in enumerate(x):
        f_input, ax_input = plt.subplots()
        PlotFrameCoord(xc, loc_tar=loc_tar, plot_colorbar=colorbar).plot(fig=f_input, ax=ax_input)
        logger.add_figure(f'input/raw_input_ch_{i}', f_input, step)

    # loop over all output channels
    for i, yc in enumerate(y_out):
        f_output, ax_output = plt.subplots()
        PlotFrameCoord(yc, plot_colorbar=colorbar).plot(fig=f_output, ax=ax_output)
        logger.add_figure(f'output/raw_output_ch_{i}', f_output, step)

    # record tar / output emitters
    tar_ch = (x.size(0) - 1) // 2

    f_loc_out, ax_loc_out = plt.subplots(figsize=(10, 8))
    PlotFrameCoord(x[tar_ch], loc_tar=loc_tar, loc_out=loc_out).plot(fig=f_loc_out, ax=ax_loc_out)
    logger.add_figure('loc_out/loc_out_tar', f_loc_out, step)

    if loc_tar.ndim == 3:
        f_loc_out_3d = plt.figure(figsize=(10, 8))
        ax_loc_out_3d = plt.axes(projection='3d')
        PlotCoordinates3D(loc_tar=loc_tar, loc_out=loc_out, match_lines=False).plot(ax=ax_loc_out_3d)
        logger.add_figure('loc_out/loc_out_tar_3d', f_loc_out_3d, step)

    f_match, ax_match = plt.subplots(figsize=(10, 8))
    PlotFrameCoord(x[tar_ch], loc_tar=loc_tp_match, loc_out=loc_tp, match_lines=True).plot(fig=f_match, ax=ax_match)
    logger.add_figure('loc_out/loc_match', f_match, step)

    if loc_tp.ndim == 3:
        f_match_3d = plt.figure(figsize=(10, 8))
        ax_match_3d = plt.axes(projection='3d')
        PlotCoordinates3D(loc_tar=loc_tp_match, loc_out=loc_tp, match_lines=True, labels=('TP match', 'TP')).plot(ax_match_3d)
        logger.add_figure('loc_out/loc_match_3d', f_match_3d, step)


def log_loss(loss_cmp_perbatch:torch.Tensor, source:str, logger:MultiLogger, step:int):
    """
    Logs the loss components per batch at each epoch
    Args:
        loss_cmp_perbatch:  (n_batches, 3) the loss components [[loss_count, loss_loc, loss_bkg],...] of each batch at the given step
        source:             'test' or 'train' to specify where the loss comes from
        logger:
        step:               the logging epoch
    """
    if loss_cmp_perbatch.dim() != 2:
        raise ValueError("unsupported loss_cmp_perbatch shape, should be (n_batches, 3)")
    
    logger.add_scalar(f'learning/loss_all_{source}', loss_cmp_perbatch.sum(-1).mean(), step)
    for i in range(loss_cmp_perbatch.size(1)):
        logger.add_scalar(f'learning/loss_ch{i}_{source}', loss_cmp_perbatch[:, i].mean(), step)
        

def log_err(tp:FrameLocSet, tp_match:FrameLocSet, logger:MultiLogger, step:int, cam_pxszs):
    """
    Logs the errors between the true positives and the ground truth matching the true positives
    Args:
        tp:             true positives
        tp_match:       ground truth mathiching the true positives
        logger:
        step:           the step / epoch
    """
    if len(tp) == 0 or len(tp_match) == 0:
        return
    err_scatter = ErrorScatterPlot(cam_pxszs=cam_pxszs, residual=True, sub_sample=1000)
    fig_res, axes_res = err_scatter.plot_errors(tp, tp_match)
    logger.add_figure('residuals/pxyz_res', fig_res, step)

    err_hist = ErrorHistPlot(cam_pxszs=cam_pxszs, SampleRate=50)
    fig_hist, axes_hist = err_hist.plot_errors(tp, tp_match)
    logger.add_figure('errhist/pxyz_err', fig_hist, step)
