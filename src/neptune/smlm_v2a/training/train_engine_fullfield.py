from typing import Union
import argparse
import shutil
from pathlib import Path

import torch

from .checkpoint import CheckPoint
from .progress import GMMHeuristicCheck
from . import train_setup
from . import train_kernels

from ..utils.io import param_io
from ..utils.logger_utils import log_loss, log_frames, log_err
from ..utils.helpers import current_info
from ..evaluation.full_field_eval import prepare_full_field_eval, evaluate_full_field


def parse_args():
    p = argparse.ArgumentParser(description="Training Args (with full-field eval)")

    p.add_argument("-p", "--param_file", required=True, help="Specify your parameter file (.yml or .json).")
    p.add_argument("-d", "--debug", default=False, action="store_true", help="Debug the specified parameter file")
    p.add_argument("-c", "--experiment_comment", default=None, help="Add a experiment_comment to the run.")
    p.add_argument("-i", "--device", default=None, help="Specify the device stringand overwrite param.", type=str, required=False)
    p.add_argument("-w", "--num_worker_override", default=None, help="Override the number of workers for the dataloaders.", type=int)
    p.add_argument("-n", "--no_log", default=False, action="store_true", help="Set no logging")
    p.add_argument("-l", "--log_folder", default="runs", help="Specify the (parent) folder you want to log to")
    p.add_argument("--eval_stride", default=10, type=int, help="Run eval every N epochs.")

    p.add_argument(
        "--phase_config",
        default="/home/guest/Others/main/config/pipeline_pointset_min.json",
        help="Phase-retrieval pipeline config for full-field eval.",
    )
    p.add_argument(
        "--phase_tiff",
        default="/home/guest/Others/main/phase_retrieval/beads_data/spool_100mW_30ms_3D_1_2_MMStack_Default.ome.tif",
        help="Phase-retrieval TIFF path for full-field eval.",
    )
    p.add_argument(
        "--full_field_frames",
        default=150,
        type=int,
        help="Frames per full-field ROI evaluation sample.",
    )
    p.add_argument(
        "--full_field_output",
        default="/home/guest/Others/main/smlm/output/full_field_eval",
        help="Output directory for full-field eval plots.",
    )
    p.add_argument(
        "--full_field_cache",
        default="/home/guest/Others/main/smlm/output/full_field_cache",
        help="Cache directory for full-field eval datasets.",
    )

    args = p.parse_args()
    return args


def live_engine_setup(
    param_file: Union[Path, str],
    debug: bool = False,
    experiment_comment: str = None,
    device_overwrite: str = None,
    num_worker_override: int = None,
    no_log: bool = False,
    log_folder: str = "runs",
    eval_stride: int = 10,
    phase_config: Union[Path, str] = None,
    phase_tiff: Union[Path, str] = None,
    full_field_frames: int = 150,
    full_field_output: Union[Path, str] = None,
    full_field_cache: Union[Path, str] = None,
):
    """
    Sets up the engine to train, with periodic full-field eval.
    """
    param_file = Path(param_file) if isinstance(param_file, str) else param_file
    param, calibration = param_io.load_params(param_file)

    # Experiment output
    if debug or param.ExpInOut.checkpoint_init is None:
        from_ckpt = False
        if not Path(param.ExpInOut.experiment_out).exists():
            Path(param.ExpInOut.experiment_out).mkdir()
        experiment_id = "debug" if debug else current_info(experiment_comment)
        experiment_path = Path(param.ExpInOut.experiment_out) / Path(experiment_id)
        experiment_path.mkdir(exist_ok=True if debug else False)
    else:
        from_ckpt = True
        if not Path(param.ExpInOut.checkpoint_init).is_file():
            raise FileNotFoundError(
                f"could not find the initialization checkpoint file in {param.ExpInOut.checkpoint_init}"
            )
        experiment_id = Path(param.ExpInOut.checkpoint_init).parent.name
        experiment_path = Path(param.ExpInOut.checkpoint_init).parent
    model_out = experiment_path / Path("model.pt")

    device = train_setup.setup_device(param, device_overwrite, num_worker_override)

    # Backup parameter file
    param_backup_in = experiment_path / Path("param_run_in").with_suffix(param_file.suffix)
    shutil.copy(param_file, param_backup_in)

    param_backup = experiment_path / Path("param_run").with_suffix(param_file.suffix)
    if debug:
        param = param_io.convert_param_debug(param)
    param_io.save_params(param_backup, param)

    logger = train_setup.setup_logger(no_log, log_folder, experiment_id)

    loc_meta = train_setup.setup_loc_meta(param, calibration)

    sim_train, sim_test = train_setup.setup_random_simulation(loc_meta, param, calibration)
    ds_train, ds_test = train_setup.setup_dataset(param, sim_train, sim_test)
    dl_train, dl_test = train_setup.setup_dataloader(param, ds_train, ds_test)
    model, model_ls, optimizer, lr_scheduler = train_setup.setup_fitter(param, model_out, device)
    criterion = train_setup.setup_criterion(loc_meta, param)
    post_processor, matcher = train_setup.setup_postprocess(loc_meta, param)

    try:
        imwidth = param.Simulation.emitter_extent[0][1] - param.Simulation.emitter_extent[0][0]
        imheight = param.Simulation.emitter_extent[1][1] - param.Simulation.emitter_extent[1][0]
        dummy = torch.rand((2, param.HyperParameter.channels_in, imheight, imwidth), device=device)
        logger.add_graph(model, dummy)
    except:
        print("Did not log model graph")

    # Full-field eval preparation
    phase_config = Path(phase_config) if phase_config is not None else None
    phase_tiff = Path(phase_tiff) if phase_tiff is not None else None
    full_field_output = Path(full_field_output) if full_field_output is not None else None
    full_field_cache = Path(full_field_cache) if full_field_cache is not None else None

    full_field_data = prepare_full_field_eval(
        param=param,
        calibration=calibration,
        phase_config_path=phase_config,
        tiff_path=phase_tiff,
        frames_per_sample=int(full_field_frames),
        output_dir=full_field_output,
        cache_dir=full_field_cache,
    )

    if from_ckpt:
        ckpt = CheckPoint.load(param.ExpInOut.checkpoint_init)
        model.load_state_dict(ckpt.model_state)
        optimizer.load_state_dict(ckpt.optimizer_state)
        lr_scheduler.load_state_dict(ckpt.lr_sched_state)
        first_epoch = ckpt.step + 1
        print(f"Resuming training from checkpoint {experiment_id}")
    else:
        ckpt = CheckPoint(path=experiment_path / Path("ckpt.pt"))
        first_epoch = 0

    converges = False
    n = 0
    n_max = param.HyperParameter.auto_restart_param.num_restarts
    while not converges and n < n_max:

        n += 1
        conv_check = GMMHeuristicCheck(
            nLocs_avg=sim_train.loc_sampler.em_avg,
            threshold=param.HyperParameter.auto_restart_param.restart_treshold,
            ref_epoch=1,
        )
        for i in range(first_epoch, param.HyperParameter.epochs):
            logger.add_scalar("learning/learning_rate", optimizer.param_groups[0]["lr"], i)

            if i >= 1:
                loss_cmp_train = train_kernels.train_forward(
                    model=model,
                    optimizer=optimizer,
                    loss=criterion,
                    dataloader=dl_train,
                    grad_mod=param.HyperParameter.grad_mod,
                    epoch=i,
                    device=device,
                )
                log_loss(loss_cmp_perbatch=loss_cmp_train, source="train", logger=logger, step=i)

            loss_cmp_test, test_data = train_kernels.test_forward(
                model=model, loss=criterion, dataloader=dl_test, epoch=i, device=device
            )
            log_loss(loss_cmp_perbatch=loss_cmp_test, source="test", logger=logger, step=i)

            if not conv_check(loss_cmp_test[:, :2].sum(1).mean(), i):
                print(
                    f"The model will be reinitialized and retrained due to a pathological loss."
                    f"The max. allowed loss per loc is {conv_check.threshold:.1f} vs."
                    f" {(loss_cmp_test[:, :2].sum(1).mean() / conv_check.nlocs_avg):.1f} (observed)."
                )

                model, model_ls, optimizer, lr_scheduler = train_setup.setup_fitter(param, model_out, device)
                logger = train_setup.setup_logger(no_log, log_folder, experiment_id)
                converges = False
                break
            else:
                converges = True

            do_eval = (i % max(1, int(eval_stride)) == 0)

            if do_eval:
                loc_out = post_processor.forward(test_data.y_out)
                loc_tar = ds_test._loc_tar
                matching_locsets, eval_sets = train_kernels.eval_forward(
                    cam_pxszs=param.Camera.cam_pxszs,
                    matcher=matcher,
                    loc_out=loc_out,
                    loc_tar=loc_tar,
                )
                loc_tp = matching_locsets.TP
                loc_tp_match = matching_locsets.TP_match

                eval_dict = eval_sets._asdict()
                print(
                    f"rec: {eval_dict['rec']:.3f}, prec: {eval_dict['prec']:.3f}, "
                    f"jacc: {eval_dict['jac']:.3f}, rmse_r: {eval_dict['rmse_lat']:.3f}, "
                    f"rmse_z: {eval_dict['rmse_ax']:.3f}"
                )
                logger.add_scalar_dict("eval/", eval_dict, global_step=i)

                log_frames(
                    x=test_data.x,
                    y_out=test_data.y_out,
                    loc_out=loc_out,
                    loc_tar=loc_tar,
                    loc_tp=loc_tp,
                    loc_tp_match=loc_tp_match,
                    logger=logger,
                    step=i,
                )
                log_err(loc_tp, loc_tp_match, logger, step=i, cam_pxszs=param.Camera.cam_pxszs)

                ff_mean_lat, ff_mean_ax = evaluate_full_field(
                    model=model,
                    eval_data=full_field_data,
                    param=param,
                    calibration=calibration,
                    device=device,
                    batch_size=param.HyperParameter.batch_size,
                    output_dir=full_field_output,
                    epoch=i,
                )
                print(
                    f"[full-field] rmse_lat_mean={ff_mean_lat:.3f}, rmse_ax_mean={ff_mean_ax:.3f}"
                )

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

            if param.Simulation.mode == "acquisition":
                ds_train.sample(verbose=True)
            elif param.Simulation.mode != "samples":
                raise ValueError

    if converges:
        print("Training finished after reaching maximum number of epochs.")
    else:
        raise ValueError(f"Training aborted after {n_max} restarts.")


if __name__ == "__main__":
    args = parse_args()
    live_engine_setup(
        param_file=args.param_file,
        debug=args.debug,
        experiment_comment=args.experiment_comment,
        device_overwrite=args.device,
        num_worker_override=args.num_worker_override,
        no_log=args.no_log,
        log_folder=args.log_folder,
        eval_stride=args.eval_stride,
        phase_config=args.phase_config,
        phase_tiff=args.phase_tiff,
        full_field_frames=args.full_field_frames,
        full_field_output=args.full_field_output,
        full_field_cache=args.full_field_cache,
    )
