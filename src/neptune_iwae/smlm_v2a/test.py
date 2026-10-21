from typing import Tuple, Union, Callable

import numpy as np
import torch

import matplotlib.figure as mfig
import matplotlib.axes as maxes
import matplotlib.figure as mfig
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as maxes3d
from matplotlib.collections import PathCollection

from scipy.stats import gaussian_kde
from scipy.linalg import cholesky

from .generic.localizations import FrameLocSet, EmitterLocSet


class PlotCoordinates:
    
    tar_dict = {'color':'tab:red', 'marker':'o', 'cmap':'winter', 'mfc':'none', 'label':'Target'}
    out_dict = {'color':'tab:blue', 'marker':'x', 'cmap':'viridis', 'mfc':'none', 'label':'Output'}
    ini_dict = {'color':'tab:green', 'marker':'+', 'cmap':'copper', 'mfc':'none', 'label':'Init'}

    def __init__(self, *, loc_tar:Union[FrameLocSet, EmitterLocSet]=None, adjust_to_frame:bool=True):
        """
        Plots points in 2D projection.
        Args:
            loc_tar:            the FrameLocSet of the target
            loc_out:            the FrameLocSet of extracted from the output
            loc_ini:            the FrameLocSet of the initialization
            match_lines:        whether to connect the matched point between the output and the target
            adjust_to_frame:    whether to deduct 0.5 to make the ploting adjust to the frame for imshow
            cmap_photon:        whether to use photon as the cmap 
        """
        self.adjust_to_frame = adjust_to_frame
        self.xyz_tar, self.phot_tar = self._adjust_coord_to_frame(loc_tar)
        
    def _adjust_coord_to_frame(self, loc:Union[FrameLocSet, EmitterLocSet]) -> Tuple[np.ndarray, np.ndarray]:
        """Adjusts the xyz coordinates to plot on the corresponding frame"""
        
        if loc is None:
            return np.zeros((0, 2), dtype=np.float32), np.zeros((0,), dtype=np.float32)
        
        xyz = loc.xyz.clone().detach().numpy()
        phot = loc.phot.clone().detach().numpy()
        if self.adjust_to_frame:
            xyz[:, 0] -= loc.xextent[0] + 0.5
            xyz[:, 1] -= loc.yextent[0] + 0.5
        return xyz, phot
        
    def printf(self):
        print(torch.cat([torch.from_numpy(self.xyz_tar), loc_tar.xyz], dim=1))



if __name__ == '__main__':

    meta_dict = {'ndim': 3, 'xextent': (0, 40), 'yextent': (0, 40), 'zextent': (0, 121),
                 'nchannels': 1, 'R2Ts': None, 'Ts2R': None,
                 'i_type': torch.int32, 'f_type': torch.float32}
    data_dict = {'xyz': torch.stack([torch.ones((10,), dtype=torch.float32), 2.0 + torch.ones((10,), dtype=torch.float32),
                                     5.0 + torch.ones((10,), dtype=torch.float32)], dim=1),
                 'phot': torch.rand((10,), dtype=torch.float32)}
    loc_dict = {}
    loc_dict.update(meta_dict)
    loc_dict.update(data_dict)
    loc_tar = FrameLocSet(**loc_dict)

    PlotCoordinates(loc_tar=loc_tar).printf()
    PlotCoordinates(loc_tar=loc_tar).printf()
    PlotCoordinates(loc_tar=loc_tar).printf()
