from typing import Union
import json
import shutil
from pathlib import Path
import argparse
import torch

from .checkpoint import CheckPoint
from .progress import GMMHeuristicCheck
from . import train_setup
from . import train_kernels

from ..utils.io import param_io
from ..utils.logger_utils import log_loss, log_frames, log_err
from ..utils.helpers import current_info



def _load_existing_best_state(experiment_path: Union[Path, str]) -> tuple[float, int]:
    experiment_path = Path(experiment_path)
    best_meta_path = experiment_path / Path('best_jaccard.json')
    if not best_meta_path.is_file():
        return float('-inf'), -1

    try:
        payload = json.loads(best_meta_path.read_text())
        return float(payload['jaccard']), int(payload['epoch'])
    except Exception:
        return float('-inf'), -1


def parse_args():
    p = argparse.ArgumentParser(description='Training Args')

    p.add_argument('-p', '--param_file', required=True, help='Specify your parameter file (.yml or .json).')
    p.add_argument('-d', '--debug', default=False, action='store_true', help='Debug the specified parameter file')
    p.add_argument('-c', '--experiment_comment', default=None, help='Add a experiment_comment to the run.')
    p.add_argument('-i', '--device', default=None, help='Specify the device stringand overwrite param.', type=str, required=False)
    p.add_argument('-w', '--num_worker_override', default=None, help='Override the number of workers for the dataloaders.', type=int)
    p.add_argument('-n', '--no_log', default=False, action='store_true', help='Set no logging')
    p.add_argument('-l', '--log_folder', default='runs', help='Specify the (parent) folder you want to log to')
    p.add_argument('--best_epoch_limit', default=300, type=int, help='Save best Jaccard within the first N epochs (<=0 to disable).')
    p.add_argument('--epochs_override', default=None, type=int, help='Override total training epochs for this run.')
    p.add_argument('--run_info_path', default=None, help='Optional path to write run metadata (experiment_id/path).')
    
    args = p.parse_args()
    return args


def live_engine_setup(param_file:Union[Path, str], debug:bool=False, experiment_comment:str=None, 
                      device_overwrite:str=None, num_worker_override:int=None, no_log:bool=False, log_folder:str='runs',
                      best_epoch_limit:int=300, epochs_override:int=None, run_info_path:Union[Path, str]=None):
    """
    Sets up the engine to train. Includes sample simulation and the actual training
    Args:
        param_file:             parameter file (.json, .yml or .yaml) path
        debug:                  active debug mode (i.e. less samples) for fast testing
        experiment_comment:     comment to the experiment
        device_overwrite:       overwrite cuda index specified by param_file
        num_worker_override:    overwrite number of workers for dataloader
        no_log:                 disable logging
        log_folder:             directory for logging (where tensorboard puts its stuff)
        best_epoch_limit:       save best Jaccard checkpoint within first N epochs (<=0 to disable)
        epochs_override:        override total training epochs for this run
        run_info_path:          optional path to write run metadata
    """
    ##### Initialize the parameters. param can be different from param_file due to autofill
    param_file = Path(param_file) if isinstance(param_file, str) else param_file
    param, calibration = param_io.load_params(param_file)

    if epochs_override is not None and epochs_override > 0:
        param.HyperParameter.epochs = epochs_override

    ##### Set up Experiment ID and Paths
    if debug or param.ExpInOut.checkpoint_init is None:
        from_ckpt = False
        if not Path(param.ExpInOut.experiment_out).exists():
            Path(param.ExpInOut.experiment_out).mkdir()
        experiment_id = 'debug' if debug else current_info(experiment_comment)
        experiment_path = Path(param.ExpInOut.experiment_out) / Path(experiment_id)
        experiment_path.mkdir(exist_ok = True if debug else False)
    else:
        from_ckpt = True
        if not Path(param.ExpInOut.checkpoint_init).is_file():
            raise FileNotFoundError(f"could not find the initialization checkpoint file in {param.ExpInOut.checkpoint_init}")
        experiment_id = Path(param.ExpInOut.checkpoint_init).parent.name
        experiment_path = Path(param.ExpInOut.checkpoint_init).parent 
    model_out = experiment_path / Path('model.pt')
    if run_info_path is not None:
        run_info_path = Path(run_info_path)
        run_info_path.parent.mkdir(parents=True, exist_ok=True)
        with run_info_path.open('w') as f:
            json.dump({'experiment_id': experiment_id, 'experiment_path': str(experiment_path)}, f, indent=2)
    
    ##### Hardware and Server stuff. Overriding parameters can also make param different from param_file
    device = train_setup.setup_device(param, device_overwrite, num_worker_override)
    
    ##### Backup the parameter file under the network output path with the experiments ID
    param_backup_in = experiment_path / Path('param_run_in').with_suffix(param_file.suffix)
    shutil.copy(param_file, param_backup_in)

    param_backup = experiment_path / Path('param_run').with_suffix(param_file.suffix)
    if debug:
        param = param_io.convert_param_debug(param)
    param_io.save_params(param_backup, param) 
    
    ##### Set up the logger
    logger = train_setup.setup_logger(no_log, log_folder, experiment_id)

    ##### Set up the meta FrameLocSet for simulation and feature extraction
    loc_meta = train_setup.setup_loc_meta(param, calibration)
    
    ##### Set up the simulator, dataset, dataloader, fitters, criterion, and postprocessors
    sim_train, sim_test = train_setup.setup_random_simulation(loc_meta, param, calibration)
    ds_train, ds_test = train_setup.setup_dataset(param, sim_train, sim_test)
    dl_train, dl_test = train_setup.setup_dataloader(param, ds_train, ds_test)
    model, model_ls, optimizer, lr_scheduler = train_setup.setup_fitter(param, model_out, device)
    criterion = train_setup.setup_criterion(loc_meta, param)
    post_processor, matcher = train_setup.setup_postprocess(loc_meta, param)
    
    ##### Display the model architecture if logger (tensorboard.SummarWritter) available
    try:
        imwidth = param.Simulation.emitter_extent[0][1] - param.Simulation.emitter_extent[0][0]
        imheight = param.Simulation.emitter_extent[1][1] - param.Simulation.emitter_extent[1][0]
        dummy = torch.rand((2, param.HyperParameter.channels_in, imheight, imwidth), device=device)
        logger.add_graph(model, dummy)
    except:
        print("Did not log model graph")

    ##### Training
    if from_ckpt:
        ckpt = CheckPoint.load(param.ExpInOut.checkpoint_init)
        model.load_state_dict(ckpt.model_state)
        optimizer.load_state_dict(ckpt.optimizer_state)
        lr_scheduler.load_state_dict(ckpt.lr_sched_state)
        first_epoch = ckpt.step + 1
        print(f'Resuming training from checkpoint ' + experiment_id)
    else:
        ckpt = CheckPoint(path = experiment_path / Path('ckpt.pt'))
        first_epoch = 0

    best_epoch_limit = best_epoch_limit or 0
    best_model_path = experiment_path / Path('model_best_jaccard.pt')
    best_meta_path = experiment_path / Path('best_jaccard.json')
    if from_ckpt:
        best_jaccard, best_epoch = _load_existing_best_state(experiment_path)
    else:
        best_jaccard = float('-inf')
        best_epoch = -1

    converges = False
    n = 0
    n_max = param.HyperParameter.auto_restart_param.num_restarts
    while not converges and n < n_max:
        
        n += 1
        conv_check = GMMHeuristicCheck(nLocs_avg=sim_train.loc_sampler.em_avg, threshold=param.HyperParameter.auto_restart_param.restart_treshold, ref_epoch=1)
        for i in range(first_epoch, param.HyperParameter.epochs):
            logger.add_scalar('learning/learning_rate', optimizer.param_groups[0]['lr'], i)

            if i >= 1:
                loss_cmp_train = train_kernels.train_forward(model=model, optimizer=optimizer, loss=criterion, dataloader=dl_train, 
                                                             grad_mod=param.HyperParameter.grad_mod, epoch=i, device=device)
                log_loss(loss_cmp_perbatch=loss_cmp_train, source='train', logger=logger, step=i)
            
            loss_cmp_test, test_data = train_kernels.test_forward(model=model, loss=criterion, dataloader=dl_test, epoch=i, device=device)
            log_loss(loss_cmp_perbatch=loss_cmp_test, source='test', logger=logger, step=i)

            if not conv_check(loss_cmp_test[:, :2].sum(1).mean(), i):
                print(f"The model will be reinitialized and retrained due to a pathological loss."
                      f"The max. allowed loss per loc is {conv_check.threshold:.1f} vs."
                      f" {(loss_cmp_test[:, :2].sum(1).mean() / conv_check.nlocs_avg):.1f} (observed).")

                # reset the neural fitter and the logger
                model, model_ls, optimizer, lr_scheduler = train_setup.setup_fitter(param, model_out, device)
                logger = train_setup.setup_logger(no_log, log_folder, experiment_id)
                converges = False
                break
            else:
                converges = True
            
            # Post-Process and Evaluate
            loc_out = post_processor.forward(test_data.y_out)
            loc_tar = ds_test._loc_tar
            matching_locsets, eval_sets = train_kernels.eval_forward(cam_pxszs=param.Camera.cam_pxszs, matcher=matcher, loc_out=loc_out, loc_tar=loc_tar)
            loc_tp = matching_locsets.TP
            loc_tp_match = matching_locsets.TP_match
            
            eval_dict = eval_sets._asdict()
            print(f"rec: {eval_dict['rec']:.3f}, prec: {eval_dict['prec']:.3f}, jacc: {eval_dict['jac']:.3f}, rmse_r: {eval_dict['rmse_lat']:.3f}, rmse_z: {eval_dict['rmse_ax']:.3f}")
            logger.add_scalar_dict('eval/', eval_dict, global_step=i)
            if best_epoch_limit > 0 and i < best_epoch_limit:
                jac = float(eval_dict['jac'])
                if jac > best_jaccard:
                    best_jaccard = jac
                    best_epoch = i
                    torch.save(model.state_dict(), best_model_path)
                    with best_meta_path.open('w') as f:
                        json.dump({'epoch': best_epoch, 'jaccard': best_jaccard}, f, indent=2)
                    logger.add_scalar('eval/best_jaccard', best_jaccard, i)
                    print(f"Saved best Jaccard checkpoint at epoch {best_epoch} (jac={best_jaccard:.4f}).")

            log_frames(x=test_data.x, y_out=test_data.y_out, loc_out=loc_out, loc_tar=loc_tar, loc_tp=loc_tp, loc_tp_match=loc_tp_match, logger=logger, step=i)
            log_err(loc_tp, loc_tp_match, logger, step=i, cam_pxszs=param.Camera.cam_pxszs)
            
            if i >= 1:
                if isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    lr_scheduler.step(loss_cmp_test.sum(1).mean())
                else:
                    lr_scheduler.step()

            model_ls.save(model, metric_val=None)
            if no_log:
                ckpt.dump(i, model.state_dict(), optimizer.state_dict(), lr_scheduler.state_dict())
            else:
                ckpt.dump(i, model.state_dict(), optimizer.state_dict(), lr_scheduler.state_dict(), logger.loggers[1].log_dict)

            # Draw new samples
            if param.Simulation.mode == 'acquisition':
                ds_train.sample(verbose=True)
            elif param.Simulation.mode != 'samples':
                raise ValueError

    if converges:
        print("Training finished after reaching maximum number of epochs.")
    else:
        raise ValueError(f"Training aborted after {n_max} restarts.")
    

if __name__ == '__main__':
    args = parse_args()
    live_engine_setup(args.param_file, args.debug, args.experiment_comment, args.device, args.num_worker_override,
                      args.no_log, args.log_folder, args.best_epoch_limit, args.epochs_override, args.run_info_path)
