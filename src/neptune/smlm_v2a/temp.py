from pathlib import Path
import pickle
import math
import torch

def dualview_setting():
    fpath_beads = Path('D:/Data/beads/XY_20231215_beads_3D')
    fpath_save = Path('D:/Data/simulations/20250911_test/dualview_settings')

    cam_path = fpath_beads / Path('Camera_settings.pkl')
    csplinePSF_path = fpath_beads / Path('PSFCalibration_unlinked.pkl')
    chnlT_path = fpath_beads / Path('chnlwarp_coeff.pkl')

    with open(cam_path, 'rb') as fid:
        cam = pickle.load(fid)
    cam_chipszx = cam.pop('chipszw')
    cam_chipszy = cam.pop('chipszh')
    cam_pxszx = cam.pop('cam_pxszx')
    cam_pxszy = cam.pop('cam_pxszy')
    _ = cam.pop('cam_pth')
    cam['chipszx'] = cam_chipszx
    cam['chipszy'] = cam_chipszy
    cam['cam_pxszs'] = [cam_pxszx, cam_pxszy]
    cam['QE'] = 0.95
    cam['spur_noise'] = 0.0002
    print(cam.keys())
    print(f"chipszx={cam['chipszx']}, chipszy={cam['chipszy']}, view_type={cam['view_type']}")
    with open(fpath_save / Path('camera.pkl'), 'wb') as fid:
        pickle.dump(cam, fid, pickle.HIGHEST_PROTOCOL)

    with open(csplinePSF_path, 'rb') as fid:
        psf = pickle.load(fid)
    nchannels = psf['nchannels']
    csplinePSF = {'modality':psf['modality'], 'nchannels':nchannels, 'zFocus':psf['zFocus'], 'zstepsz_nm':psf['zstepsz_nm']}
    coeff_sizes = [psf['splineszx'], psf['splineszx'], psf['splineszz']]
    csplinePSF['coeff_sizes'] = coeff_sizes
    csplinePSF['coeff'] = psf['coeff'].reshape((nchannels, math.prod(coeff_sizes), -1))
    print(f"modality={csplinePSF['modality']}, nchannels={csplinePSF['nchannels']}")
    print(f"zFocus={csplinePSF['zFocus']}")
    print(f"coeff_sizes={csplinePSF['coeff_sizes']}")
    print(f"coeff_shape={csplinePSF['coeff'].shape}")
    with open(fpath_save / Path('csplinePSF.pkl'), 'wb') as fid:
        pickle.dump(csplinePSF, fid, pickle.HIGHEST_PROTOCOL)

    with open(chnlT_path, 'rb') as fid:
        chnlT = pickle.load(fid)
    R2Ts = chnlT['R2Ts'].transpose((0, 2, 1))
    Ts2R = chnlT['Ts2R'].transpose((0, 2, 1))
    print(f"R2T.shapes={R2Ts.shape}, Ts2R.shape={Ts2R.shape}")
    with open(fpath_save / Path('chnlT.pkl'), 'wb') as fid:
        pickle.dump({'R2Ts':R2Ts, 'Ts2R':Ts2R}, fid, pickle.HIGHEST_PROTOCOL)


def fullview_setting():
    fpath_beads = Path('D:/Data/beads/XY_20231215_beads_3D')
    fpath_save = Path('D:/Data/simulations/20250911_test/fullview_settings')

    cam_path = fpath_beads / Path('Camera_settings.pkl')
    csplinePSF_path = fpath_beads / Path('PSFCalibration_unlinked.pkl')

    with open(cam_path, 'rb') as fid:
        cam = pickle.load(fid)
    cam_chipszx = cam.pop('chipszw')
    cam_chipszy = cam.pop('chipszh')
    cam_pxszx = cam.pop('cam_pxszx')
    cam_pxszy = cam.pop('cam_pxszy')
    _ = cam.pop('cam_pth')
    cam['chipszx'] = cam_chipszx
    cam['chipszy'] = cam_chipszy
    cam['cam_pxszs'] = [cam_pxszx, cam_pxszy]
    cam['QE'] = 0.95
    cam['spur_noise'] = 0.0002

    cam['view_type'] = 'fullview'
    print(cam.keys())
    print(f"chipszx={cam['chipszx']}, chipszy={cam['chipszy']}, view_type={cam['view_type']}")
    with open(fpath_save / Path('camera.pkl'), 'wb') as fid:
        pickle.dump(cam, fid, pickle.HIGHEST_PROTOCOL)

    with open(csplinePSF_path, 'rb') as fid:
        psf = pickle.load(fid)
    nchannels = 1
    csplinePSF = {'modality':psf['modality'], 'nchannels':nchannels, 'zFocus':psf['zFocus'], 'zstepsz_nm':psf['zstepsz_nm']}
    coeff_sizes = [psf['splineszx'], psf['splineszx'], psf['splineszz']]
    csplinePSF['coeff_sizes'] = coeff_sizes
    csplinePSF['coeff'] = psf['coeff'][[0]].reshape((nchannels, math.prod(coeff_sizes), -1))
    print(f"modality={csplinePSF['modality']}, nchannels={csplinePSF['nchannels']}")
    print(f"zFocus={csplinePSF['zFocus']}")
    print(f"coeff_sizes={csplinePSF['coeff_sizes']}")
    print(f"coeff_shape={csplinePSF['coeff'].shape}")
    with open(fpath_save / Path('csplinePSF.pkl'), 'wb') as fid:
        pickle.dump(csplinePSF, fid, pickle.HIGHEST_PROTOCOL)

    with open(fpath_save / Path('chnlT.pkl'), 'wb') as fid:
        pickle.dump({'R2Ts':None, 'Ts2R':None}, fid, pickle.HIGHEST_PROTOCOL)


def simple_test():
    x = torch.tensor([0, 0, 0, 2, 2, 2, 2, 2, 5, 5, 6, 7, 7, 7, 10, 10, 11], dtype=torch.int32)
    print(f"x = {x}")
    x_new = torch.empty(size=(len(x)+1,), dtype=torch.int32)
    x_new[:-1] = x
    x_new[-1] = x[-1] + 1
    
    y = torch.unique_consecutive(x_new)
    print(f"y = {y}")

    z = torch.searchsorted(x_new, y)
    print(f"z = {z}")


if __name__ == '__main__':

    #fullview_setting()
    simple_test()



