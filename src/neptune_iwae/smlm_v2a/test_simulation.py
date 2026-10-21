import sys
from pathlib import Path
from tqdm import tqdm

import torch
import matplotlib.pyplot as plt

from .utils.io import param_io
from .training.train_setup import setup_random_simulation, setup_dataset, setup_dataloader
from .utils.plotter import PlotFrameCoord
from .utils import hardware


def simulation_test():
    
    fpath = Path('D:/Data/simulations/20250911_test/fullview_settings')
    param, calibration = param_io.load_params(fpath / Path('param_test.yaml'))
    sim_train, sim_test = setup_random_simulation(param, calibration)
    _testing = sim_train

    loc, frames, bkg_frames, pxyz_tar, mask_tar = _testing.sample() 
    print(f"len(loc)={len(loc)}, frame_rng={(loc.indf.min(), loc.indf.max())}")
    print(f"frames.shape={frames.shape}")
    print(f"bkg_frames.shape={bkg_frames.shape}")
    print(f"pxyz_tar.shape={pxyz_tar.shape}")
    print(f"mask_tar.shape={mask_tar.shape}")
    print(f"nlocs per frame={mask_tar.sum(-1)}")
    print(f"device: frames.{frames.device}, bkg_frames.{bkg_frames.device}, pxyz_tar.{pxyz_tar.device}, mask_tar.{mask_tar.device}")


def ds_test():
    fpath = Path('D:/Data/simulations/20250911_test/fullview_settings')
    param, calibration = param_io.load_params(fpath / Path('param_test.yaml'))
    
    sim_train, sim_test = setup_random_simulation(param, calibration)
    train_dataset, test_dataset = setup_dataset(param, sim_train, sim_test)
    _testing = test_dataset
    for attr in dir(_testing):
        if not (attr.startswith('__') and attr.endswith('__')):
            if attr in ('_frames', '_pxyz_tar', '_bkg_frames', '_mask_tar'):
                item = getattr(_testing, attr)
                print(attr, item.shape)
            elif attr == '_loc':
                print(attr, len(_testing._loc))
            else:
                print(attr)
    
    nfrm = param.Simulation.train_ds_size if _testing == train_dataset else param.Simulation.test_size
    ix = torch.randint(low=0, high=nfrm, size=(1,))[0]
    
    sub_loc = _testing._loc_perframe[ix]
    xextent = sub_loc.xextent
    yextent = sub_loc.yextent
    print(f"xextent={xextent}, yextent={yextent}")
    
    sub_frame = _testing._frames[ix]
    print(f"frm_shape={sub_frame.shape}, frame_max={sub_frame.max()}, frm_min={sub_frame.min()}")
    sub_bkg_frame = train_dataset._bkg_frames[ix]
    print(f"bkg_shape={sub_bkg_frame.shape}, bkg_max={sub_bkg_frame.max()}, bkg_min={sub_bkg_frame.min()}")
    
    sub_pxyz_tar = _testing._pxyz_tar[ix]
    sub_mask_tar = _testing._mask_tar[ix]
    sub_pxyz_tar = sub_pxyz_tar[sub_mask_tar]
    print(sub_mask_tar.shape, sub_pxyz_tar.shape)
    print(f"phot_min={sub_pxyz_tar[:, 0].min()}, phot_max={sub_pxyz_tar[:, 0].max()}")
    print(f"z_min={sub_pxyz_tar[:, 3].min()}, z_max={sub_pxyz_tar[:, 3].max()}")
    print(sub_pxyz_tar)

    fig, ax = plt.subplots()
    PlotFrameCoord(sub_frame, sub_loc).plot(fig, ax)
    plt.show()


def dl_test():
    fpath = Path('D:/Data/simulations/20250911_test/fullview_settings')
    param, calibration = param_io.load_params(fpath / Path('param_test.yaml'))
    
    sim_train, sim_test = setup_random_simulation(param, calibration)
    train_dataset, test_dataset = setup_dataset(param, sim_train, sim_test)
    train_dataset
    train_dataloader, test_dataloader = setup_dataloader(param, train_dataset, test_dataset)
    _testing = train_dataloader

    print(len(train_dataset))
    tqdm_enum = tqdm(_testing, total=len(_testing), smoothing=0.)  # progress bar enumeration
    for batch_idx, (x, bkg_tar, pxyz_tar, mask_tar) in enumerate(tqdm_enum): 

        x, bkg_tar, pxyz_tar, mask_tar = hardware.to_device([x, bkg_tar, pxyz_tar, mask_tar], param.Hardware.device_simulation)
        print(f"batch_idx={batch_idx}")
        print(f"\tx.shape={x.shape}")
        print(f"\tbkg_tar.shape={bkg_tar.shape}")
        print(f"\tpxyz_tar.shape={pxyz_tar.shape}")
        print(f"\tmask_tar.shape={mask_tar.shape}")

    



if __name__ == '__main__':
    if sys.argv[1] == 'simu':
        simulation_test()   
    elif sys.argv[1] == 'ds':
        ds_test()
    elif sys.argv[1] == 'dl':
        dl_test()