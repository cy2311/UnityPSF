import sys
from pathlib import Path
import pickle
from PIL import Image
import matplotlib.pyplot as plt

import numpy as np
import torch

from calibration.channelT import ChannelTrans
from utils.helpers import read_roi, to_tensor


class ChnlTransTest:

    def __init__(self):

        fpath = Path('D:/Data/beads/XY_20231215_beads_3D')
        with open(fpath / 'chnlwarp_coeff.pkl', 'rb') as fid:
            chnlT = pickle.load(fid)
        with open(fpath / 'Camera_settings.pkl', 'rb') as fid:
            cam = pickle.load(fid)
            cam_pxszs = to_tensor([cam['cam_pxszx'], cam['cam_pxszy'], 10.0], dtype=torch.float32, device='cuda')

        nchannels = len(chnlT['R2Ts']) + 1
        R2Ts = np.transpose(chnlT['R2Ts'], (0, 2, 1))
        Ts2R = np.transpose(chnlT['Ts2R'], (0, 2, 1))
        self.chnl_trans = ChannelTrans(nchannels, R2Ts=R2Ts, Ts2R=Ts2R)

        fpath = Path('D:/Data/samples/HYF_G1A50_20240111_SAMPLES_MICRO_647M_TOM20_660CR_samples-3d/spool_1000MW_3D_50ms_5_2')
        _, roi = read_roi(fpath / '0605-0301.roi')
        self.xextent_ch0 = to_tensor([roi[1], roi[3]], dtype=torch.int32, device='cuda')
        self.yextent_ch0 = to_tensor([roi[0], roi[2]], dtype=torch.int32, device='cuda')

        with open(fpath / 'smlm_result_mulpa' / 'spool_1000MW_3D_50ms_5_2_roi_0_locsnm_noconsecg.pkl', 'rb') as fid:
            smlm_data = pickle.load(fid)
            self.indf = to_tensor(smlm_data['indf'], dtype=torch.int32, device='cuda')
            self.xyz = to_tensor(smlm_data['xvec'][:, :3], dtype=torch.float32, device='cuda')
            self.xyz /= cam_pxszs[None, :]
            self.phot = to_tensor(smlm_data['xvec'][:, 3], dtype=torch.float32, device='cuda')
            fracN = to_tensor(smlm_data['xvec'][:, [4]], dtype=torch.float32, device='cuda')
            self.fracN = torch.cat((fracN, 1.0-fracN), dim=1)

        with Image.open(fpath / 'spool_1000MW_3D_50ms_5_2_MMStack_Default_0.ome.tif') as ImObj:
            ImObj.seek(0)
            dum_camim = torch.as_tensor(np.asarray(ImObj, dtype=np.float32), dtype=torch.float32, device='cuda')
            dum_camim = (dum_camim - cam['offset']) / cam['A2D'] / cam['EMGain']
            self.ims = torch.cat((dum_camim[None, :, :600], dum_camim[None, :, 600:1200]), dim=0)

    
    def test(self, arg:str):
        offset_ch0 = to_tensor([self.xextent_ch0[0], self.yextent_ch0[0]], dtype=torch.int32, device='cuda')
        if arg == 'coor':   
            xyz_ch1 = self.chnl_trans.coor_polytrans(self.xyz, 1, 'R2T', offset_src=offset_ch0, offset_dst=offset_ch0)
            xyz_ch0 = self.chnl_trans.coor_polytrans(xyz_ch1, 1, 'T2R', offset_src=offset_ch0, offset_dst=offset_ch0)
            dlta = xyz_ch0 - self.xyz
            print(f"min = {dlta.min(0).values}")
            print(f"max = {dlta.max(0).values}")
        
        elif arg == 'extent':
            xextent_dst, yextent_dst = self.chnl_trans.extent_polytrans(xextent_src=self.xextent_ch0, yextent_src=self.yextent_ch0, chID=1, mode='R2T')
            print(self.xextent_ch0)
            print(xextent_dst)
            print(self.yextent_ch0)
            print(yextent_dst)
        

if __name__  == '__main__':

    chnlT = ChnlTransTest()
    chnlT.test(sys.argv[1])