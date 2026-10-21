from typing import Union, Tuple
import sys
import os
import datetime
import socket
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ..models.smlm_model import SingleUNet, DoubleUNet

from ..calibration.camera import PerfectCamera, EMCCD, sCMOS95, sCMOS82

from ..data.smlm_dataset import InferenceDataset, smlm_collate
from ..data.data_factory import FrameRescalar
from ..data.feature_factory import SpatialIntegration

from ..utils import hardware
from ..utils.io import model_io, frames_io
from ..utils.io.param_io import RecursiveNamespace
from ..utils.processing import TransformSequence


def setup_experiment_ID(param:RecursiveNamespace, debug:bool, log_comment:str=None) -> str:
    """
    Args:
        debug:          active debug mode (i.e. less samples) for fast testing
        param:          configured parameters
        log_comment:    comment to the experiment
    Returns:
        str:            experiment ID
    """
    if debug:
        return Path('debug')
    if param.InOut.checkpoint_init is not None:
        return Path(param.InOut.checkpoint_init).parent.name
    experiment_id = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + '_' + socket.gethostname()
    return experiment_id + '_' + log_comment if log_comment is not None else experiment_id
        

def setup_device(param:RecursiveNamespace, device_overwrite:str=None, num_worker_override:int=None) -> str:
    """
    Set up the hardware and server stuff
    Args:
        param:                  configured parameters
        device_overwrite:       the device to overwrite the param.Hardware.device in param
        num_worker_override:    the number of workers to overwrite the param.Hardware.num_worker_train in param
    Returns:
        device:                 str, the device
    """
    
    device = device_overwrite if device_overwrite is not None else param.Hardware.device

    if num_worker_override is not None:
        param.Hardware.num_worker_train = num_worker_override

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



def setup_dataset(param:RecursiveNamespace) -> InferenceDataset:
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
    target_proc = TargetProcess(disable_attr=param.HyperParameter.disabled_attributes, phot_max=param.Scaling.phot_max, z_max=param.Scaling.z_max)
    
    if param.Simulation.mode == 'acquisition':
        train_dataset = SMLMLiveDataset(simulator_train, frame_proc=frame_proc, bkg_frame_proc=bkg_proc, tar_proc=target_proc, 
                                        frame_window=param.HyperParameter.channels_in)
        train_dataset.sample(verbose=True)
    elif param.Simulation.mode == 'samples':
        train_dataset = SMLMLiveSampleDataset(simulator_train, param.Simulation.train_ds_size, frame_proc=frame_proc, bkg_frame_proc=bkg_proc,
                                              tar_proc=target_proc, frame_window=param.HyperParameter.channels_in)
    
    test_dataset = SMLMPriorDataset(simulator_test, frame_proc=frame_proc, bkg_frame_proc=bkg_proc, tar_proc=target_proc,
                                    frame_window=param.HyperParameter.channels_in)
    test_dataset.sample(verbose=True)
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
                                 drop_last=False, shuffle=False, num_workers=param.Hardware.num_work_train,
                                 pin_memory=False, collate_fn=smlm_collate) if test_dataset is not None else None
    return train_dataloader, test_dataloader


def setup_trainer(param:RecursiveNamespace, logger:MultiLogger, model_out:Union[Path, str], ckpt_path:Union[Path, str], device:Union[torch.device, str]):
    """
    Args:
        param:              configured parameters
        logger:             the Logging instance to log the training progress
        model_out:          the path to save the model
        ckpt:               the path to save the checkpoint
        device:             device for training
    Returns:
        model:              Union[SingleUNet, DoubleUNet], the model instance
        model_ls:           model_io.LoadSaveModel, the snl instance to save the model
        optimizer:          Union[torch.optim.Adam, torch.optim.AdamW] the optimizer
        criterion:          GMMLoss, the loss instance
        lr_scheduler:       Union[ReduceLROnPlateau, StepLR] the learning rate instance
        grad_mod:           bool, whether to implement the grad_mod
        checkpoint:         CheckPoint instance to save the state_dict of the model, optimizer, lr_scheduler,...etc      
    """
    
    # Instantiates a model from constructor and a model snl instance to save the models
    models_available = {'SingleUNet': SingleUNet, 'DoubleUNet': DoubleUNet}
    model = models_available[param.HyperParameter.architecture].parse(param)
    
    model_ls = model_io.LoadSaveModel(model, output_file=model_out)
    model = model_ls.load_init()
    model = model.to(device)

    # Sets up the optimizers
    optimizer_available = {'Adam': torch.optim.Adam, 'AdamW': torch.optim.AdamW}
    optimizer = optimizer_available[param.HyperParameter.optimizer]
    optimizer = optimizer(model.parameters(), **param.HyperParameter.opt_param)

    # Sets up the Loss function
    xyoffset = (param.Simulation.emitter_extent[0][0], param.Simulation.emitter_extent[1][0])
    criterion = GMMLoss(xyoffset=xyoffset, ch_weight=param.HyperParameter.chweight_stat)

    # Sets up the leraning rate
    lr_scheduler_available = {'ReduceLROnPlateau':torch.optim.lr_scheduler.ReduceLROnPlateau,
                              'StepLR':torch.optim.lr_scheduler.StepLR}
    lr_scheduler = lr_scheduler_available[param.HyperParameter.learning_rate_scheduler]
    lr_scheduler = lr_scheduler(optimizer, **param.HyperParameter.learning_rate_scheduler_param)

    # Sets up CheckPoint
    checkpoint = CheckPoint(path=ckpt_path)

    # Sets up gradient modification
    grad_mod = param.HyperParameter.grad_mod

    # Log the model
    try:
        imwidth = param.Simulation.emitter_extent[0][1] - param.Simulation.emitter_extent[0][0]
        imheight = param.Simulation.emitter_extent[1][1] - param.Simulation.emitter_extent[1][0]
        dummy = torch.rand((2, param.HyperParameter.channels_in, imheight, imwidth), device=device)
        logger.add_graph(model, dummy)
    except:
        print("Did not log graph")

    return model, model_ls, optimizer, criterion, lr_scheduler, grad_mod, checkpoint


def setup_postprocess(param:RecursiveNamespace) -> Tuple[SpatialIntegration, GreedyHungarianMatching]:
    """
    Set up the feature extractor and matcher
    Args:
        param:              configured parameters
    Returns:
        feature_extractor:  Union[NoFeatureExtract, LookUpFeature, SpatialIntegration] the feature extractor
        matcher:            GreedyHungarianMatching, the matcher
    """
    
    # Sets up the feature extractor
    if param.PostProcessing is None:
        feature_extractor = NoFeatureExtract()
    elif param.PostProcessing == 'LookUp':
        feature_extractor = LookUpFeature(raw_th=param.PostProcessingParam.raw_th, phot_max=param.Scaling.phot_max,
                                          z_max=param.Scaling.z_max, bkg_max=param.Scaling.bkg_max)
    elif param.PostProcessing == 'SpatialIntegration':
        feature_extractor = SpatialIntegration(raw_th=param.PostProcessingParam.raw_th, split_th=param.PostProcessingParam.split_th,
                                               phot_max=param.Scaling.phot_max, z_max=param.Scaling.z_max, bkg_max=param.Scaling.nlg_max)
    else:
        raise NotImplementedError

    # Sets up the matcher
    matcher = GreedyHungarianMatching(match_dims=param.Evaluation.match_dims, cam_pxszs=param.Calibration.cam_pxszs, 
                                      dist_tolr=param.Evaluation.dist_tolr, dist_tolz=param.Evaluation.dist_tolz, dist_tolv=param.Evaluation.dist_tolvol)
    return feature_extractor, matcher