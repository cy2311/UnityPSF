import sys
from typing import Tuple

import torch

from .generic.localizations import FrameLocSet
from .utils.io import localization_io


class LocSetTest:

    def __init__(self):
        meta = self.set_meta(ndim=3, nchannels=2)
        data_a = self.set_data(meta, nloc_perfrm=3, frm_rng=(10, 12), phot_mu_sig=(20000, 1000))
        data_b = self.set_data(meta, nloc_perfrm=2, frm_rng=(20, 24), phot_mu_sig=(5000, 250))

        loc_dict_a = {}
        loc_dict_a.update(meta)
        loc_dict_a.update(data_a)
        self.loc_a = FrameLocSet(**loc_dict_a)

        self.loc_b = FrameLocSet(**meta)
        self.loc_b.init_load(**data_b)
        print("Initializing two LocSet for further testing")
        print(self.loc_a)
        print(self.loc_b)

    @staticmethod
    def set_meta(ndim:int, nchannels:int) -> dict:
        
        i_type = torch.int32
        f_type = torch.float32

        xextent = (10, 60)
        yextent = (10, 60)
        zextent = (10, 60)
        
        R2Ts = torch.zeros((1, 4, 2))
        R2Ts[0, 1, 0] = 1.0
        R2Ts[0, 2, 1] = 1.0
        Ts2R = R2Ts.detach().clone()
        
        return {'ndim':ndim, 'xextent':xextent, 'yextent':yextent, 'zextent':zextent,
                'nchannels':nchannels, 'R2Ts':R2Ts, 'Ts2R':Ts2R, 'i_type':i_type, 'f_type':f_type} 
    
    @staticmethod
    def set_data(meta:dict, nloc_perfrm:int, frm_rng:Tuple[int, int], phot_mu_sig:Tuple[float, float]):
        
        nfrm = frm_rng[1] - frm_rng[0]
        nlocs = nfrm * nloc_perfrm
        indf = torch.arange(frm_rng[0], frm_rng[1]).repeat_interleave(nloc_perfrm)
        x = meta['xextent'][0] + (meta['xextent'][1] - meta['xextent'][0]) * torch.rand((nlocs, 1))
        y = meta['yextent'][0] + (meta['yextent'][1] - meta['yextent'][0]) * torch.rand((nlocs, 1))
        z = meta['zextent'][0] + (meta['zextent'][1] - meta['zextent'][0]) * torch.rand((nlocs, 1))
        xyz = torch.cat([x, y, z], dim=1)
        phot = phot_mu_sig[0] + phot_mu_sig[1] * torch.randn((nlocs,))
        data_dict = {'indf':indf, 'xyz':xyz, 'phot':phot}
        
        if meta['nchannels'] > 1:
            fracN = torch.rand((nlocs, meta['nchannels']))
            data_dict['fracN'] = fracN / fracN.sum(dim=1)[:, None]
        
        return data_dict

    def test(self, request:str):
        if request in {'add', 'cat', 'iadd'}:
            if request == 'add':
                loc_c = self.loc_a + self.loc_b 
            elif request == 'cat': 
                loc_c = FrameLocSet.loc_cat([self.loc_a, self.loc_b])
            else:
                loc_c = self.loc_a.clone()
                loc_c += self.loc_b
            print(loc_c)
            print(loc_c.indf)
            localization_io.save_xlsx('test_locset.xlsx', loc_c)
            
        elif request in {'sort_frm', 'isort_frm', 'split_by_frame', 'get_sub_frame'}:
            loc_c = self.loc_b.clone()
            loc_c += self.loc_a
            if request == 'sort_frm':
                loc_d = loc_c.sort_by_frame()
                print(f"loc_c.indf = {loc_c.indf}")
                print(f"loc_d.indf = {loc_d.indf}")
            
            elif request == 'isort_frm':
                print(f"loc_c.indf(before) = {loc_c.indf}")
                loc_c.sort_by_frame_()
                print(f"loc_c.indf(after) = {loc_c.indf}")

            elif request == 'split_by_frame':
                loc_subsets = loc_c.split_by_frame(loose=True)
                for i, _loc in enumerate(loc_subsets):
                    print(f"the {i}-th subloc:")
                    print(_loc)
                    print(_loc.active_frames)

            elif request == 'get_sub_frame':
                loc_d = loc_c.get_subframe((3, 7))
                print(loc_d)
                print(loc_d.indf)
        
    


if __name__ == '__main__':
    
    loc_test = LocSetTest()
    loc_test.test(sys.argv[1])