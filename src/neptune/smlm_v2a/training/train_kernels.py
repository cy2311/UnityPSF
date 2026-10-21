import warnings
from typing import Tuple, Union
from collections import namedtuple
from collections.abc import Sequence
from tqdm import tqdm
import time

import torch
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.jit import TracerWarning

from .progress import MetricMeter
from .losses import GMMLoss
from ..models.smlm_model import SingleUNet, DoubleUNet

from ..generic.localizations import FrameLocSet
from ..evaluation.matcher import GreedyHungarianMatching, MatchResult
from ..evaluation.evaluation import SMLMEvaluation, SMLMEvalResult

from ..utils import hardware


def _primary_input_tensor(x):
    if isinstance(x, Sequence) and not isinstance(x, torch.Tensor):
        if len(x) == 0:
            raise ValueError("Empty model input sequence.")
        return _primary_input_tensor(x[0])
    return x


def train_forward(model:Union[SingleUNet, DoubleUNet], optimizer:Optimizer, loss:GMMLoss, dataloader:DataLoader, 
                  grad_mod:bool, epoch:int, device:torch.device) -> torch.Tensor:
    """
    The core kernel of the training
    Args:
        model:      the nn Model for training
        optimizer:  the torch optimizer
        loss:       the loss instance that defines how loss is calculated
        dataloader: the torch utils data DataLoader to access the SMLMDataSet batchwise
        grad_mod:   Ture to L2-normalize the gradients each epoch, with the max_norm set to 0.03
        epoch:      the epoch
        logger:     a MultiLogger instance to log the loss at the current epoch
        device:     torch.device
    Returns:
        torch.Tensor: (n_batches, 3) the loss components [[loss_count, loss_loc, loss_bkg],...] of each batch at the given epoch
    """
    
    loss_cmp_epoch = []
    loss_vals_log = MetricMeter()

    model.train()
    
    tqdm_enum = tqdm(dataloader, total=len(dataloader), smoothing=0.)
    t0 = time.time()
    
    for batch_idx, (x, detect_tar, bkg_tar, pxyz_tar, mask_tar, frame_loc) in enumerate(tqdm_enum): 
        
        t_data = time.time() - t0
        x, detect_tar, bkg_tar, pxyz_tar, mask_tar = hardware.to_device([x, detect_tar, bkg_tar, pxyz_tar, mask_tar], device)
        
        optimizer.zero_grad()
        y_out = model(x)
        
        loss_val = loss.forward(y_out, detect_tar, pxyz_tar, mask_tar, bkg_tar) # -> (batch_size, 3)
        loss_mean = loss_val.sum(-1).mean()
        try:
            loss_mean.backward()    
        except:
            for name, param in model.named_parameters():
                if not torch.isfinite(param.grad).all():
                    nan_detected = True
            if nan_detected:
                print(f"loss_mean = {loss_mean}")
                for name, param in model.named_parameters():
                    _finite = torch.isfinite(param.grad).all()
                    _norm = param.grad.norm()
                    print(f"{name}: finite={_finite}, norm={_norm:.6f}")
            raise ValueError
        
        if grad_mod:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.03, norm_type=2)
        optimizer.step()

        t_batch = time.time() - t0
        loss_vals_log.update(loss_mean)
        tqdm_enum.set_description(f"(Train) E: {epoch} - t: {t_batch:.2} - t_dat: {t_data:.2} - L: {loss_vals_log.mean:.3}")
        
        loss_cmp_epoch.append(loss_val.mean(dim=0).detach().cpu())
        del loss_val
        t0 = time.time()
    
    loss_cmp_epoch = torch.stack(loss_cmp_epoch, dim=0)
    return loss_cmp_epoch



_val_return = namedtuple("test_data", ["x", "y_out"])
def test_forward(model:Union[SingleUNet, DoubleUNet], loss:GMMLoss, dataloader:DataLoader, epoch:int, device:torch.device) -> Tuple[torch.Tensor, _val_return]:
    """
    The core kernel of the training
    Args:
        model:          the nn Model for training
        loss:           the loss instance that defines how loss is calculated
        dataloader:     the torch utils data DataLoader to access the SMLMDataSet batchwise
        epoch:          the epoch
        device:         torch.device
    Returns:
        torch.Tensor:   (n_batches, 3) the loss components [[loss_count, loss_loc, loss_bkg],...] of each batch at the given epoch
        _val_return:    x:      the input frames (nbatch * batchsize, C=channels_in, H, W)
                        y_out:  the output frames (nbatch * batchsize, C=10, H, W)   
    """
    x_ep, y_out_ep, loss_cmp_epoch = [], [], []  # store things epoche wise
    loss_vals_log = MetricMeter()

    model.eval()
    tqdm_enum = tqdm(dataloader, total=len(dataloader), smoothing=0.) 
    t0 = time.time()

    with torch.no_grad():
        for batch_idx, (x, detect_tar, bkg_tar, pxyz_tar, mask_tar, frame_loc) in enumerate(tqdm_enum):

            x, detect_tar, bkg_tar, pxyz_tar, mask_tar = hardware.to_device([x, detect_tar, bkg_tar, pxyz_tar, mask_tar], device)
            
            y_out = model(x)
            loss_val = loss.forward(y_out, detect_tar, pxyz_tar, mask_tar, bkg_tar) # -> (batch_size, 3)
            
            t_batch = time.time() - t0
            loss_vals_log.update(loss_val.sum(-1).mean())
            tqdm_enum.set_description(f"( Test) E: {epoch} - t: {t_batch:.2} - L: {loss_vals_log.mean:.3}")

            loss_cmp_epoch.append(loss_val.mean(dim=0).detach().cpu())
            del loss_val
            x_ep.append(_primary_input_tensor(x).detach().cpu())
            y_out_ep.append(y_out.detach().cpu())

    # Batch-Wise concatenate
    loss_cmp_epoch = torch.stack(loss_cmp_epoch, dim=0)
    x_ep = torch.cat(x_ep, dim=0)
    y_out_ep = torch.cat(y_out_ep, dim=0)

    return loss_cmp_epoch, _val_return(x=x_ep, y_out=y_out_ep)



def eval_forward(cam_pxszs:Tuple[float, float, float], *, matcher:GreedyHungarianMatching, 
                 loc_out:FrameLocSet, loc_tar:FrameLocSet) -> Tuple[MatchResult, SMLMEvalResult]:
    """
    Extracts the localizations from the output of the neural network, and perform evaluation against the target localization
    Args:
        cam_pxszs:      pixel sizes for [x, y, z] axes in nanometer
        matcher:        the matcher between the extracted and targeted localization set
        loc_out:        the output localization set extracted from the output of the neural network
        loc_tar:        the target localization set
    Returns:
        MatchResult:    the matching NamedTuple containing the matching localizations
        SMLMEvalResult: the evaluation NamedTuple containing the evaluation metrics
    """
    loc_sets = matcher.forward(loc_out, loc_tar)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        evaluator = SMLMEvaluation(cam_pxszs=cam_pxszs, weighting='sig', reduction='gaussian')
        eval_set = evaluator.forward(loc_sets.TP, loc_sets.FP, loc_sets.FN, loc_sets.TP_match)
    
    return loc_sets, eval_set
