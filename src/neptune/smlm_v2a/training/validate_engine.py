from win32api import GetSystemMetrics
from typing import Union, Tuple
import sys
import os
import json
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from .losses import GMMLoss
from . import train_kernels

from ..calibration.psf import CSplinePSF
from ..calibration.camera import PerfectCamera, EMCCD, sCMOS95, sCMOS82

from ..generic.localizations import FrameLocSet

from ..simulation.loc_generator import EmitterSamplerBlinking
from ..simulation.bkg_generator import UniformBackground
from ..simulation.simulator import Simulation

from ..data.smlm_dataset import SMLMPriorDataset, smlm_collate
from ..data.data_factory import FrameRescalar, BkgFrameRescalar, TargetProcess
from ..data.feature_factory import NoFeatureExtract, LookUpFeature, SpatialIntegration

from ..models.smlm_model import SingleUNet, DoubleUNet

from ..evaluation.matcher import GreedyHungarianMatching

from ..utils import hardware
from ..utils.io import param_io, model_io
from ..utils.param_types import RecursiveNamespace
from ..utils.plotter import PlotFrameCoord, ErrorHistPlot


def parse_args():
    p = argparse.ArgumentParser(description='Validation Args')

    p.add_argument('-p', '--param_file', required=True, help='Specify your parameter file (.yml or .json).')

    args = p.parse_args()
    return args


def setup_device(param:RecursiveNamespace, device_overwrite:str=None) -> str:
    """
    Set up the hardware and server stuff
    Args:
        param:                  configured parameters
        device_overwrite:       the device to overwrite the param.Hardware.device in param
    Returns:
        device:                 str, the device
    """
    if device_overwrite is not None:
        device = device_overwrite
        param.Hardware.device_simulation = device_overwrite
    else:
        device = param.Hardware.device

    if torch.cuda.is_available():
        _, device_ix = hardware.specific_device_by_str(device)
        if device_ix is not None:
            torch.cuda.set_device(device)
    elif not torch.cuda.is_available():
        device = 'cpu'

    if param.Hardware.torch_multiprocessing_sharing_strategy is not None:
        torch.multiprocessing.set_sharing_strategy(param.Hardware.torch_multiprocessing_sharing_strategy)

    if sys.platform in ('linux', 'darwin'):
        os.nice(param.Hardware.unix_niceness)
    elif param.Hardware.unix_niceness is not None:
        print(f"Cannot set niceness on platform {sys.platform}. You probably do not need to worry.")

    torch.set_num_threads(param.Hardware.torch_threads)    
    return device


def setup_loc_meta(param:RecursiveNamespace, calibration:RecursiveNamespace) -> FrameLocSet:
    """
    Setup the loc meta (as an empty FrameLocSet) to define the simulation and feature extractor accordingly
    Args:
        param:          configured parameters
        calibration:    calibration for training simulation
    Returns:
        FrameLocSet:    A meta LocSet to define the simulation and feature extractor
    """
    loc_meta = {'ndim':param.Simulation.ndim,
                'xextent':param.Simulation.emitter_extent[0],
                'yextent':param.Simulation.emitter_extent[1],
                'zextent':param.Simulation.emitter_extent[2],
                'nchannels':param.Simulation.nchannels,
                'R2Ts':calibration.ChannelTrans.R2Ts,
                'Ts2R':calibration.ChannelTrans.Ts2R,
                'i_type':param.DataType.i_type,
                'f_type':param.DataType.f_type}
    return FrameLocSet(**loc_meta)


def setup_random_simulation(loc_meta:FrameLocSet, param:RecursiveNamespace, calibration:RecursiveNamespace) -> Simulation:
    """
    Setup the actual simulation
    Args:
        loc_meta:       A meta FrameLoSet that defines the simulation
        param:          configured parameters
        calibration:    calibration for training simulation
    Returns:
        Simulation:     simulator for the training dataset, run simulator.sample() to get the train dataset
        Simulation:     simulator for the testing dataset, run simulator.sample() to get the test dataset
    """
    # Cameras and PSF
    cameras_available = {'PerfectCamera': PerfectCamera, 'EMCCD': EMCCD, 'sCMOS95':sCMOS95, 'sCMOS82':sCMOS82}
    cam = cameras_available[param.Camera.cam_type].parse(calibration)
    psf = CSplinePSF.parse(calibration)

    # Simulators
    frame_rng_test = (0, param.Simulation.test_ds_size)
    loc_sampler_test = EmitterSamplerBlinking.parse(frame_rng_test, loc_meta, param) 
    bkg_sampler = UniformBackground(param.Simulation.bkg_uniform)
    return Simulation.parse(cam, psf, loc_sampler_test, bkg_sampler, param)


def setup_dataset(param:RecursiveNamespace, simulator_test:Simulation) -> SMLMPriorDataset:
    """
    Set up the training and testing dataset
    Args:
        param:              configured parameters
        simulator_test:     the simulator for training dataset, must run simulator_train.sample() to have data
    Returns:
        SMLMPriorDataset:   the testing dataset
    """
    frame_proc = FrameRescalar(scale=param.Scaling.input_scale, offset=param.Scaling.input_offset)
    bkg_proc = BkgFrameRescalar(scale=param.Scaling.bkg_max)
    target_proc = TargetProcess(disable_attr=param.DataFilter.disabled_attributes, phot_max=param.Scaling.phot_max, z_max=param.Scaling.z_max)
    
    test_dataset = SMLMPriorDataset(simulator_test, frame_proc=frame_proc, bkg_frame_proc=bkg_proc, tar_proc=target_proc,
                                    frame_window=param.HyperParameter.channels_in, return_loc=False)
    test_dataset.sample(verbose=True, nan_check=True)
    return test_dataset


def setup_dataloader(param:RecursiveNamespace, test_dataset:SMLMPriorDataset) -> Tuple[DataLoader, DataLoader]:
    """
    Set up the DataLoader
    Args:
        param:              configured parameters
        test_dataset:       the testing dataset
    Returns:
        DataLoader:         the dataloader for testing
    """
    return DataLoader(dataset=test_dataset, batch_size=param.HyperParameter.batch_size, 
                      drop_last=False, shuffle=False, num_workers=param.Hardware.num_worker_train,
                      pin_memory=False, collate_fn=smlm_collate) if test_dataset is not None else None


def setup_model(param:RecursiveNamespace, model_in:Union[Path, str], device:Union[torch.device, str]) -> Tuple[
                  Union[SingleUNet, DoubleUNet], model_io.LoadSaveModel]:
    """
    Set up the neural fitter
    Args:
        param:              configured parameters
        model_out:          the path to save the model
        device:             device for training
    Returns:
        model:              Union[SingleUNet, DoubleUNet], the model instance
        model_ls:           model_io.LoadSaveModel, the snl instance to save the model
    """
    # Instantiates a model from constructor and a model snl instance to save the models
    models_available = {'SingleUNet': SingleUNet, 'DoubleUNet': DoubleUNet}
    model = models_available[param.HyperParameter.architecture].parse(param)
    
    model_ls = model_io.LoadSaveModel(model, output_file=None, input_file=model_in)
    model = model_ls.load_init(device)
    
    return model


def setup_criterion(loc_meta:FrameLocSet, param:RecursiveNamespace) -> GMMLoss:
    """
    Set up the loss instance
    Args:
        loc_meta:   A meta FrameLoSet that defines the feature extractor
        param:      configured parameters
    Returns:
        GMMLoss:    The Gaussian Mixture Modeled Loss instance
    """
    xyoffset = (loc_meta.xextent[0], loc_meta.yextent[0])
    return GMMLoss(xyoffset=xyoffset, ch_weight=param.HyperParameter.loss_weight)


def setup_postprocess(loc_meta:FrameLocSet, param:RecursiveNamespace) -> Tuple[SpatialIntegration, GreedyHungarianMatching]:
    """
    Set up the feature extractor and matcher
    Args:
        loc_meta:           A meta FrameLoSet that defines the feature extractor
        param:              configured parameters
        calibration:        calibration for training simulation
    Returns:
        feature_extractor:  Union[NoFeatureExtract, LookUpFeature, SpatialIntegration] the feature extractor
        matcher:            GreedyHungarianMatching, the matcher
    """
    # Sets up the feature extractor
    if param.PostProcessing is None:
        feature_extractor = NoFeatureExtract(loc_meta)
    elif param.PostProcessing == 'LookUp':
        feature_extractor = LookUpFeature(loc_meta, raw_th=param.PostProcessingParam.raw_th, phot_max=param.Scaling.phot_max,
                                          z_max=param.Scaling.z_max, bkg_max=param.Scaling.bkg_max)
    elif param.PostProcessing == 'SpatialIntegration':
        feature_extractor = SpatialIntegration(loc_meta, raw_th=param.PostProcessingParam.raw_th, split_th=param.PostProcessingParam.split_th,
                                               phot_max=param.Scaling.phot_max, z_max=param.Scaling.z_max, bkg_max=param.Scaling.bkg_max)
    else:
        raise NotImplementedError

    # Sets up the matcher
    matcher = GreedyHungarianMatching(match_dims=param.Evaluation.match_dims, cam_pxszs=param.Camera.cam_pxszs, 
                                      dist_tolr=param.Evaluation.dist_tolr, dist_tolz=param.Evaluation.dist_tolz, dist_tolv=param.Evaluation.dist_tolvol)
    return feature_extractor, matcher


def validate(model_in:Union[str, Path]):
    """ Setting up the validation with a trained model """
    
    scrWidth, scrHeight = GetSystemMetrics(0)//72, GetSystemMetrics(1)//72

    model_in = Path(model_in) if isinstance(model_in, str) else model_in
    fpath = model_in.parent
    param_file = fpath / Path('param_run.yaml')
    param, calibration = param_io.load_params(param_file)

    device = setup_device(param)
    loc_meta = setup_loc_meta(param, calibration)
    model = setup_model(param, model_in, device)

    criterion = setup_criterion(loc_meta, param)

    _sim = setup_random_simulation(loc_meta, param, calibration)
    _ds = setup_dataset(param, _sim)
    _dl = setup_dataloader(param, _ds)
    post_processor, matcher = setup_postprocess(loc_meta, param)

    loss_cmp_test, test_data = train_kernels.test_forward(model=model, loss=criterion, dataloader=_dl, epoch=0, device=device)
    print(f"loss_batch_mean = {loss_cmp_test.mean(0)}, loss_batch_std = {loss_cmp_test.std(0)}")
    loc_out = post_processor.forward(test_data.y_out)
    loc_tar = _ds._loc_tar
    
    matching_locsets, eval_sets = train_kernels.eval_forward(cam_pxszs=param.Camera.cam_pxszs, matcher=matcher, loc_out=loc_out, loc_tar=loc_tar)
    loc_tp = matching_locsets.TP
    loc_tp_match = matching_locsets.TP_match
    eval_result = eval_sets._asdict()
    jsonObj = json.dumps(eval_result, indent=4)
    with open(fpath / 'evaluation.json', "w") as jsfid:
        jsfid.write(jsonObj)
    for k, w in eval_result.items():
        print(f"{k}: {w:.3f}")
    
    r_ix = torch.randint(0, len(test_data.x), (1,)).int().item()
    x = test_data.x[r_ix]
    y_out = test_data.y_out[r_ix]
    sub_loc_out = loc_out.get_subframe([r_ix, r_ix+1])
    sub_loc_tar = loc_tar.get_subframe([r_ix, r_ix+1])
    sub_loc_tp = loc_tp.get_subframe([r_ix, r_ix+1])
    sub_loc_tp_match = loc_tp_match.get_subframe([r_ix, r_ix+1])

    f_input, ax_input = plt.subplots(nrows=1, ncols=len(x), figsize=(scrWidth, scrHeight))
    for i, xc in enumerate(x):
        PlotFrameCoord(xc, loc_tar=sub_loc_tar, plot_colorbar=True).plot(fig=f_input, ax=ax_input[i])
    f_input.savefig(fpath / 'inputs.png', format='PNG', dpi=600)

    f_output, ax_output = plt.subplots(nrows=3, ncols=4, figsize=(scrWidth, scrHeight))
    for i, yc in enumerate(y_out):
        PlotFrameCoord(yc, plot_colorbar=True).plot(fig=f_output, ax=ax_output[i//4][i%4])
    f_output.savefig(fpath / 'outputs.png', format='PNG', dpi=600)

    tar_ch = (x.size(0) - 1) // 2
    f_loc_out, ax_loc_out = plt.subplots(nrows=1, ncols=2, figsize=(scrWidth, scrHeight))
    PlotFrameCoord(x[tar_ch], loc_tar=sub_loc_tar, loc_out=sub_loc_out).plot(fig=f_loc_out, ax=ax_loc_out[0])
    PlotFrameCoord(x[tar_ch], loc_tar=sub_loc_tp_match, loc_out=sub_loc_tp, match_lines=True).plot(fig=f_loc_out, ax=ax_loc_out[1])
    f_loc_out.savefig(fpath / 'matching.png', format='PNG', dpi=600)

    f_errhist, _ = ErrorHistPlot(cam_pxszs=param.Camera.cam_pxszs).plot_errors(loc_tp, loc_tp_match)
    f_errhist.savefig(fpath / 'err_hist.png', format='PNG', dpi=600)


if __name__ == '__main__':
    args = parse_args()
    validate(args.param_file)