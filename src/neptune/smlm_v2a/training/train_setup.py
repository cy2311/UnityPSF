from typing import Union, Tuple
import sys
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .losses import GMMLoss

from ..calibration.psf import CSplinePSF
from ..calibration.camera import PerfectCamera, EMCCD, sCMOS95, sCMOS82

from ..generic.localizations import FrameLocSet

from ..simulation.loc_generator import EmitterSamplerBlinking
from ..simulation.bkg_generator import UniformBackground
from ..simulation.simulator import Simulation

from ..data.smlm_dataset import SMLMLiveDataset, SMLMLiveSampleDataset, SMLMPriorDataset, smlm_collate
from ..data.data_factory import FrameRescalar, BkgFrameRescalar, TargetProcess
from ..data.feature_factory import NoFeatureExtract, LookUpFeature, SpatialIntegration

from ..models.smlm_model import SingleUNet, DoubleUNet

from ..evaluation.matcher import GreedyHungarianMatching

from ..utils import hardware
from ..utils.io import model_io
from ..utils.param_types import RecursiveNamespace
from ..utils.logger_utils import SummaryWriter, NoLog, DictLogger, MultiLogger



def setup_device(param:RecursiveNamespace, device_overwrite:str=None, num_worker_overwrite:int=None) -> str:
    """
    Set up the hardware and server stuff
    Args:
        param:                  configured parameters
        device_overwrite:       the device to overwrite the param.Hardware.device in param
        num_worker_overwrite:   the number of workers to overwrite the param.Hardware.num_worker_train in param
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

    if num_worker_overwrite is not None:
        param.Hardware.num_worker_train = num_worker_overwrite

    if param.Hardware.torch_multiprocessing_sharing_strategy is not None:
        torch.multiprocessing.set_sharing_strategy(param.Hardware.torch_multiprocessing_sharing_strategy)

    if sys.platform in ('linux', 'darwin'):
        os.nice(param.Hardware.unix_niceness)
    elif param.Hardware.unix_niceness is not None:
        print(f"Cannot set niceness on platform {sys.platform}. You probably do not need to worry.")

    torch.set_num_threads(param.Hardware.torch_threads)    
    return device


def setup_logger(no_log:bool, log_folder:Union[Path, str], experiment_id:Union[Path, str]) -> MultiLogger:
    """
    Args:
        no_log:         True to disable logging
        log_folder:     directory for logging (where tensorboard puts its stuff)
        experiment_id:  the experiment ID
    Returns:
        MultiLogger:    the logger that logs stuff to both the SumaryWriter and a Dictionary
    """
    _filter_keys = ['dx_mu', 'dx_sig', 'dy_mu', 'dy_sig', 'dz_mu', 'dz_sig', 'dphot_mu', 'dphot_sig']
    if no_log:
        return NoLog()
    log_folder = Path(log_folder)  / experiment_id
    return MultiLogger([SummaryWriter(log_dir=log_folder, filter_keys=_filter_keys), DictLogger()])


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


def setup_random_simulation(loc_meta:FrameLocSet, param:RecursiveNamespace, calibration:RecursiveNamespace) -> Tuple[Simulation, Simulation]:
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

    # Frame ranges
    if param.Simulation.mode in ('acquisition', 'apriori'):
        frame_rng_train = (0, param.Simulation.train_ds_size)
    elif param.Simulation.mode == 'samples':
        frame_rng_train = (0, param.HyperParameter.channels_in)
    else:
        raise ValueError("invalid Simulation.mode, only 'acquisition', 'apriori', and 'samples' are supported")
    frame_rng_test = (0, param.Simulation.test_ds_size)

    # Loc samplers
    loc_sampler_train = EmitterSamplerBlinking.parse(frame_rng_train, loc_meta, param)  
    loc_sampler_test = EmitterSamplerBlinking.parse(frame_rng_test, loc_meta, param) 
    
    # Background
    bkg_sampler = UniformBackground(param.Simulation.bkg_uniform)

    # Main simulator
    simu_train = Simulation.parse(cam, psf, loc_sampler_train, bkg_sampler, param)
    simu_test = Simulation.parse(cam, psf, loc_sampler_test, bkg_sampler, param)
    return simu_train, simu_test


def setup_dataset(param:RecursiveNamespace, simulator_train:Simulation, simulator_test:Simulation) -> Tuple[SMLMLiveDataset, SMLMPriorDataset]:
    """
    Set up the training and testing dataset
    Args:
        param:              configured parameters
        simulator_train:    the simulator for training dataset, must run simulator_train.sample() to have data
        simulator_test:     the simulator for training dataset, must run simulator_train.sample() to have data
    Returns:
        SMLMLiveDataset:    the training dataset
        SMLMPriorDataset:   the testing dataset
    """
    frame_proc = FrameRescalar(scale=param.Scaling.input_scale, offset=param.Scaling.input_offset)
    bkg_proc = BkgFrameRescalar(scale=param.Scaling.bkg_max)
    target_proc = TargetProcess(disable_attr=param.DataFilter.disabled_attributes, phot_max=param.Scaling.phot_max, z_max=param.Scaling.z_max)
    
    if param.Simulation.mode == 'acquisition':
        train_dataset = SMLMLiveDataset(simulator_train, frame_proc=frame_proc, bkg_frame_proc=bkg_proc, tar_proc=target_proc, 
                                        frame_window=param.HyperParameter.channels_in, return_loc=False)
        train_dataset.sample(verbose=True, nan_check=True)
    elif param.Simulation.mode == 'samples':
        train_dataset = SMLMLiveSampleDataset(simulator_train, param.Simulation.train_ds_size, frame_proc=frame_proc, bkg_frame_proc=bkg_proc,
                                              tar_proc=target_proc, frame_window=param.HyperParameter.channels_in, return_loc=False)
    
    test_dataset = SMLMPriorDataset(simulator_test, frame_proc=frame_proc, bkg_frame_proc=bkg_proc, tar_proc=target_proc,
                                    frame_window=param.HyperParameter.channels_in, return_loc=False)
    test_dataset.sample(verbose=True, nan_check=True)
    return train_dataset, test_dataset


def setup_dataloader(param:RecursiveNamespace, train_dataset:Union[SMLMLiveDataset, SMLMLiveSampleDataset], 
                     test_dataset:SMLMPriorDataset=None) -> Tuple[DataLoader, DataLoader]:
    """
    Set up the DataLoader
    Args:
        param:              configured parameters
        train_dataset:      the training dataset
        test_dataset:       the testing dataset
    Returns:
        DataLoader:         the dataloader for training
        DataLoader:         the dataloader for testing
    """
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=param.HyperParameter.batch_size,
                                  drop_last=True, shuffle=True, num_workers=param.Hardware.num_worker_train,
                                  pin_memory=True, collate_fn=smlm_collate)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=param.HyperParameter.batch_size,
                                 drop_last=False, shuffle=False, num_workers=param.Hardware.num_worker_train,
                                 pin_memory=False, collate_fn=smlm_collate) if test_dataset is not None else None
    return train_dataloader, test_dataloader


def setup_fitter(param:RecursiveNamespace, model_out:Union[Path, str], device:Union[torch.device, str]) -> Tuple[
                  Union[SingleUNet, DoubleUNet], model_io.LoadSaveModel,
                  Union[torch.optim.Adam, torch.optim.AdamW],
                  Union[torch.optim.lr_scheduler.ReduceLROnPlateau, torch.optim.lr_scheduler.StepLR]]:
    """
    Set up the neural fitter
    Args:
        param:              configured parameters
        model_out:          the path to save the model
        device:             device for training
    Returns:
        model:              Union[SingleUNet, DoubleUNet], the model instance
        model_ls:           model_io.LoadSaveModel, the snl instance to save the model
        optimizer:          Union[torch.optim.Adam, torch.optim.AdamW] the optimizer
        lr_scheduler:       Union[ReduceLROnPlateau, StepLR] the learning rate instance     
    """
    # Instantiates a model from constructor and a model snl instance to save the models
    models_available = {'SingleUNet': SingleUNet, 'DoubleUNet': DoubleUNet}
    model = models_available[param.HyperParameter.architecture].parse(param)
    
    checkpoint_init = None
    if hasattr(param, "ExpInOut") and hasattr(param.ExpInOut, "checkpoint_init"):
        checkpoint_init = param.ExpInOut.checkpoint_init

    model_ls = model_io.LoadSaveModel(
        model,
        output_file=model_out,
        input_file=checkpoint_init,
    )
    model = model_ls.load_init(device)

    # Sets up the optimizers
    optimizer_available = {'Adam': torch.optim.Adam, 'AdamW': torch.optim.AdamW}
    optimizer = optimizer_available[param.HyperParameter.optimizer]
    optimizer = optimizer(model.parameters(), **param.HyperParameter.opt_param)

    # Sets up the leraning rate
    lr_scheduler_available = {'ReduceLROnPlateau':torch.optim.lr_scheduler.ReduceLROnPlateau,
                              'StepLR':torch.optim.lr_scheduler.StepLR}
    lr_scheduler = lr_scheduler_available[param.HyperParameter.learning_rate_scheduler]
    lr_scheduler = lr_scheduler(optimizer, **param.HyperParameter.learning_rate_scheduler_param)
    
    return model, model_ls, optimizer, lr_scheduler


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
