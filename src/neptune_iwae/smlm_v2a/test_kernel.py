import sys
from pathlib import Path

import torch
import matplotlib.pyplot as plt

from .models.smlm_model import DoubleUNet
from .training import train_kernels

from .utils.io import param_io
from .utils.io.model_io import LoadSaveModel
from .training import train_setup
from .utils.plotter import PlotFrameCoord
from .utils import hardware


def _parse():

    fpath = Path('D:/Data/simulations/20250911_test/fullview_settings')
    param, calibration = param_io.load_params(fpath / Path('param_test.yaml'))
    loc_meta = train_setup.setup_loc_meta(param, calibration)
    return loc_meta, param, calibration    

def simulation_test():
    
    loc_meta, param, calibration = _parse()
    train_sim, test_sim = train_setup.setup_random_simulation(loc_meta, param, calibration)
    _testing = test_sim

    loc, frames, detect_frames, bkg_frames, pxyz_tar, mask_tar = _testing.sample()
    frames, detect_frames, bkg_frames, pxyz_tar, mask_tar = hardware.to_device([frames, detect_frames, bkg_frames, pxyz_tar, mask_tar], 'cpu') 
    print(f"len(loc)={len(loc)}, frame_rng={(loc.indf.min(), loc.indf.max())}")
    print(f"frames.shape={frames.shape}, bkg_frames.shape={bkg_frames.shape}, detect_frames.shape={detect_frames.shape}")
    print(f"pxyz_tar.shape={pxyz_tar.shape}, mask_tar.shape={mask_tar.shape}")
    print(f"nlocs per frame={mask_tar.sum(-1)}")
    print(f"device: frames.{frames.device}, detect_frames.{detect_frames.device}, bkg_frames.{bkg_frames.device}, pxyz_tar.{pxyz_tar.device}, mask_tar.{mask_tar.device}")

    nfrm = param.Simulation.train_ds_size if _testing == train_sim else param.Simulation.test_ds_size
    ix = torch.randint(low=0, high=nfrm, size=(1,))[0]
    sub_loc = loc.get_subframe((ix, ix+1))
    sub_frame = frames[ix]
    sub_detect = detect_frames[ix]
    fig, ax = plt.subplots(nrows=1, ncols=2)
    PlotFrameCoord(sub_frame, loc_tar=sub_loc).plot(fig, ax[0])
    PlotFrameCoord(sub_detect, loc_tar=sub_loc).plot(fig, ax[1])
    plt.show()
    

def ds_test():
    
    loc_meta, param, calibration = _parse()
    train_sim, test_sim = train_setup.setup_random_simulation(loc_meta, param, calibration)
    train_ds, test_ds = train_setup.setup_dataset(param, train_sim, test_sim)
    _testing = test_ds
    for attr in dir(_testing):
        if not (attr.startswith('__') and attr.endswith('__')):
            if attr in ('_frames', '_detect_frames', '_bkg_frames', '_pxyz_tar', '_mask_tar'):
                item = getattr(_testing, attr)
                print(f"{attr}.shape = {item.shape}")
            elif attr in ('_loc_tar', '_loc_perframe'):
                print(f"{attr}.indf_min = {_testing._loc_tar.indf.min()}, {attr}.indf_max = {_testing._loc_tar.indf.max()}")
            else:
                print(attr)
    
    nfrm = param.Simulation.train_ds_size if _testing == train_ds else param.Simulation.test_ds_size
    ix = torch.randint(low=0, high=nfrm, size=(1,))[0]
    
    sub_loc = _testing._loc_perframe[ix]
    sub_frame = _testing._frames[ix]
    sub_detect = _testing._detect_frames[ix]
    sub_pxyz_tar = _testing._pxyz_tar[ix][_testing._mask_tar[ix]]
    
    fig, ax = plt.subplots(nrows=2, ncols=2)
    PlotFrameCoord(sub_frame, loc_tar=sub_loc).plot(fig, ax[0][0])
    PlotFrameCoord(sub_detect, loc_tar=sub_loc).plot(fig, ax[0][1])
    ax[1][0].imshow(sub_frame, cmap='gray')
    ax[1][0].plot(sub_pxyz_tar[:, 1]-0.5, sub_pxyz_tar[:, 2]-0.5, '+', mec='tab:red')
    ax[1][0].set_aspect('equal', adjustable='box')
    ax_ylimits = ax[1][0].get_ylim()
    ax[1][0].set_ylim(ax_ylimits[::-1])
    ax[1][1].imshow(sub_detect, cmap='gray')
    ax[1][1].plot(sub_pxyz_tar[:, 1]-0.5, sub_pxyz_tar[:, 2]-0.5, '+', mec='tab:red')
    ax[1][1].set_aspect('equal', adjustable='box')
    ax_ylimits = ax[1][1].get_ylim()
    ax[1][1].set_ylim(ax_ylimits[::-1])
    plt.show()


def dl_test():
    
    loc_meta, param, calibration = _parse()
    train_sim, test_sim = train_setup.setup_random_simulation(loc_meta, param, calibration)
    train_ds, test_ds = train_setup.setup_dataset(param, train_sim, test_sim)
    train_dl, test_dl = train_setup.setup_dataloader(param, train_ds, test_ds)
    _testing = test_dl

    print(len(train_dl))
    for batch_idx, (x, detect_tar, bkg_tar, pxyz_tar, mask_tar, frame_loc) in enumerate(_testing):

        x, detect_tar, bkg_tar, pxyz_tar, mask_tar = hardware.to_device([x, detect_tar, bkg_tar, pxyz_tar, mask_tar], param.Hardware.device_simulation)
        print(f"batch_idx={batch_idx}")
        print(f"\tx.shape={x.shape}")
        print(f"\tbkg_tar.shape={bkg_tar.shape}")
        print(f"\tpxyz_tar.shape={pxyz_tar.shape}")
        print(f"\tmask_tar.shape={mask_tar.shape}")


def kernel_test():
    
    fpath = Path('D:/Data/simulations/20250911_test/fullview_settings')
    param, calibration = param_io.load_params(fpath / Path('param_test.yaml'))
    loc_meta = train_setup.setup_loc_meta(param, calibration)
    logger = train_setup.setup_logger(False, './runs', 'kernel_test')

    model_instance = DoubleUNet.parse(param)
    model_ls = LoadSaveModel(model_instance, output_file=fpath/'model_test.pt', input_file=fpath/Path('2025-10-22_10-44-48_DESKTOP-7H4DP7S')/'model_0.pt')
    model = model_ls.load_init('cuda:0')

    criterion = train_setup.setup_criterion(loc_meta, param)

    train_sim, test_sim = train_setup.setup_random_simulation(loc_meta, param, calibration)
    train_ds, test_ds = train_setup.setup_dataset(param, train_sim, test_sim)
    train_dl, test_dl = train_setup.setup_dataloader(param, train_ds, test_ds)
    
    post_processor, matcher = train_setup.setup_postprocess(loc_meta, param)

    loss_cmp_test, test_data = train_kernels.test_forward(model=model, loss=criterion, dataloader=test_dl, epoch=0, device=param.Hardware.device_simulation)
    print(f"loss_batch_mean = {loss_cmp_test.mean(0)}")
    print(f"loss_batch_std = {loss_cmp_test.std(0)}")
    loc_out = post_processor.forward(test_data.y_out)
    loc_tar = test_ds._loc_tar
    
    matching_locsets, eval_sets = train_kernels.eval_forward(cam_pxszs=param.Camera.cam_pxszs, matcher=matcher, loc_out=loc_out, loc_tar=loc_tar)
    loc_tp = matching_locsets.TP
    loc_tp_match = matching_locsets.TP_match
    eval_result = eval_sets._asdict()
    for k, w in eval_result.items():
        print(f"{k}: {w:.3f}")
    
    r_ix = torch.randint(0, len(test_data.x), (1,)).int().item()
    x = test_data.x[r_ix]
    y_out = test_data.y_out[r_ix]
    sub_loc_out = loc_out.get_subframe([r_ix, r_ix+1])
    sub_loc_tar = loc_tar.get_subframe([r_ix, r_ix+1])
    sub_loc_tp = loc_tp.get_subframe([r_ix, r_ix+1])
    sub_loc_tp_match = loc_tp_match.get_subframe([r_ix, r_ix+1])

    f_input, ax_input = plt.subplots(nrows=1, ncols=len(x))
    for i, xc in enumerate(x):
        PlotFrameCoord(xc, loc_tar=sub_loc_tar, plot_colorbar=False).plot(fig=f_input, ax=ax_input[i])
    plt.show()

    f_output, ax_output = plt.subplots(nrows=3, ncols=4)
    for i, yc in enumerate(y_out):
        PlotFrameCoord(yc, plot_colorbar=False).plot(fig=f_output, ax=ax_output[i//4][i%4])
    plt.show()

    tar_ch = (x.size(0) - 1) // 2
    f_loc_out, ax_loc_out = plt.subplots(nrows=1, ncols=2)
    PlotFrameCoord(x[tar_ch], loc_tar=sub_loc_tar, loc_out=sub_loc_out).plot(fig=f_loc_out, ax=ax_loc_out[0])
    PlotFrameCoord(x[tar_ch], loc_tar=sub_loc_tp_match, loc_out=sub_loc_tp, match_lines=True).plot(fig=f_loc_out, ax=ax_loc_out[1])
    plt.show()
    



if __name__ == '__main__':
    if sys.argv[1] == 'simu':
        simulation_test()   
    elif sys.argv[1] == 'ds':
        ds_test()
    elif sys.argv[1] == 'dl':
        dl_test()
    elif sys.argv[1] == 'kernel':
        kernel_test()