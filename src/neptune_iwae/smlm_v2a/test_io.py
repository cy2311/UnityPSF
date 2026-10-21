import sys
from pathlib import Path
import pickle
import pandas as pd
import ast
import yaml

import torch
from .utils.io import param_io, calibr_io, model_io, frames_io


def test_param_io():
    # dualview
    fpath = 'D:/Data/simulations/20250911_test/dualview_settings'
    param_path = fpath / Path('param_test.yaml')
    params, calibrations = param_io.load_params(param_path)
    
    param_io.save_params(fpath / Path('param_test_run.yaml'), params)
    calibr_io.save_calibrations(fpath / Path('calibrations.pkl'), calibrations)

    with open(fpath / Path('calibrations.pkl'), 'rb') as fid:
        calibration_dict = pickle.load(fid)
    print(calibration_dict.keys())
    print(calibration_dict['Camera'].keys())
    print(calibration_dict['CSplinePSF'].keys())
    print(calibration_dict['ChannelTrans'].keys())
    print(calibration_dict['ChannelTrans']['R2Ts'])
    print(calibration_dict['ChannelTrans']['Ts2R'])
    
    # fullview
    fpath = 'D:/Data/simulations/20250911_test/fullview_settings'
    param_path = fpath / Path('param_test.yaml')
    params, calibrations = param_io.load_params(param_path)
    
    param_io.save_params(fpath / Path('param_test_run.yaml'), params)
    calibr_io.save_calibrations(fpath / Path('calibrations.pkl'), calibrations)

    with open(fpath / Path('calibrations.pkl'), 'rb') as fid:
        calibration_dict = pickle.load(fid)
    print(calibration_dict.keys())
    print(calibration_dict['Camera'].keys())
    print(calibration_dict['CSplinePSF'].keys())
    print(calibration_dict['ChannelTrans'].keys())
    print(calibration_dict['ChannelTrans']['R2Ts'])
    print(calibration_dict['ChannelTrans']['Ts2R'])


def test_load_xlsx():
    dtype_mapping = {'torch.int16':torch.int16, 'torch.int32':torch.int32, 'torch.int64':torch.int64,
                     'torch.float16':torch.float16, 'torch.float32':torch.float32, 'torch.float64':torch.float64}
    xyz_attrs = ['x', 'y', 'z']
    xyz_sig_attrs = ['x_sig', 'y_sig', 'z_sig']
    
    def _df2torch(df:pd.DataFrame, dtype:torch.dtype) -> torch.Tensor:
        return torch.from_numpy(df.to_numpy()).type(dtype)
    
    path = 'test_locset.xlsx'
    with pd.ExcelFile(path) as xlsx:
        
        meta_df = pd.read_excel(xlsx, 'meta')
        meta_keys = meta_df['meta_key'].tolist()
        meta_vals = meta_df['meta_val'].tolist()
        meta_dict = {}
        for k, v in zip(meta_keys, meta_vals):
            if k in ('xextent', 'yextent'):
                meta_dict[k] = ast.literal_eval(v)
            elif k == 'zextent' and isinstance(v, str):
                meta_dict[k] = ast.literal_eval(v)
            elif k in ('i_type', 'f_type'):
                meta_dict[k] = dtype_mapping[v]    
            else:
                meta_dict[k] = v
        
        if meta_dict['R2Ts'] is not None and meta_dict['R2Ts']:
            R2Ts_df = pd.read_excel(xlsx, 'R2Ts')
            R2Ts = torch.stack([_df2torch(R2Ts_df[f'R2T{i+1}'], meta_dict['f_type']).view(-1, 2) for i in range(meta_dict['nchannels'] - 1)], dim=0)
            meta_dict['R2Ts'] = R2Ts

        if meta_dict['Ts2R'] is not None and meta_dict['Ts2R']:
            Ts2R_df = pd.read_excel(xlsx, 'Ts2R')
            Ts2R = torch.stack([_df2torch(Ts2R_df[f'T{i+1}2R'], meta_dict['f_type']).view(-1, 2) for i in range(meta_dict['nchannels'] - 1)], dim=0)
            meta_dict['Ts2R'] = Ts2R

        data_df = pd.read_excel(xlsx, 'data')
        active_attrs = set(data_df.columns.to_list())
        data_dict = {}
        if 'x' in active_attrs:
            xyz = torch.stack([_df2torch(data_df[xyz_attrs[i]], meta_dict['f_type']) for i in range(meta_dict['ndim'])], dim=1)
            active_attrs -= {xyz_attrs[i] for i in range(meta_dict['ndim'])}
            data_dict['xyz'] = xyz
        if 'x_sig' in active_attrs:
            xyz_sig = torch.stack([_df2torch(data_df[xyz_sig_attrs[i]], meta_dict['f_type']) for i in range(meta_dict['ndim'])], dim=1)
            active_attrs -= {xyz_attrs[i] for i in range(meta_dict['ndim'])}
            data_dict['xyz_sig'] = xyz_sig
        if 'f_ch0' in active_attrs:
            fracN = torch.stack([_df2torch(data_df[f'f_ch{i}'], meta_dict['f_type']) for i in range(meta_dict['nchannels'])], dim=1)
            active_attrs -= {f'f_ch{i}' for i in range(meta_dict['nchannels'])}
            data_dict['fracN'] = fracN
        if 'f_ch0_sig' in active_attrs:
            fracN_sig = torch.stack([_df2torch(data_df[f'f_ch{i}_sig'], meta_dict['f_type']) for i in range(meta_dict['nchannels'])], dim=1)
            active_attrs -= {f'f_ch{i}_sig' for i in range(meta_dict['nchannels'])}
            data_dict['fracN_sig'] = fracN_sig
        active_attrs = list(active_attrs)
        data_dict.update({k: _df2torch(data_df[k], meta_dict['i_type'] if k in {'id', 'indf'} else meta_dict['f_type']) for k in active_attrs})
        
        
        
        print(meta_dict)
        print(meta_dict['R2Ts'].shape)
        print(meta_dict['Ts2R'].shape)
        
        print(data_dict)
        
         





if __name__ == '__main__':
    
    test_load_xlsx()

