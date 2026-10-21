from typing import Union
import pathlib
import copy
import h5py
import pickle
import ast
import pandas as pd
import torch

from ...generic.localizations import FrameLocSet, EmitterLocSet



class LocalizationFile:
    _suffix = {'.pt', '.pkl', '.h5', '.hdf5', '.xlsx'}
    
    def __init__(self, path:Union[str, pathlib.Path]):
        """ Localization io. """
        self._path = path if isinstance(path, pathlib.Path) else pathlib.Path(path)
        if self._path.suffix not in self._suffix:
            raise ValueError(f"suffix {self._path.suffix} is not supported, supported are {self._suffix}")

    def save(self, loc:Union[FrameLocSet, EmitterLocSet]) -> None:
        if self._path.suffix == '.pt':
            save_torch(self._path, loc)
        elif self._path.suffix in ('.h5', '.hdf5'):
            save_h5(self._path, loc)
        elif self._path.suffix == '.xlsx':
            save_xlsx(self._path, loc)
        else:
            save_pickle(self._path, loc)
        
    def load(self):
        if self._path.suffix == '.pt':
            return load_torch(self._path)
        elif self._path.suffix in ('.h5', '.hdf5'):
            return load_h5(self._path)
        elif self._path.suffix == '.xlsx':
            return load_xlsx(self._path)
        else:
            return load_pickle(self._path)
        


def save_h5(path:Union[str, pathlib.Path], loc:Union[FrameLocSet, EmitterLocSet]) -> None:
    """ Save a LocSet to hd5f file """
    
    with h5py.File(path, 'w') as f:
        meta = f.create_group('meta')
        meta_dict = {}
        for k, v in loc.meta.items():
            if k == 'zextent':
                meta_dict[k] = h5py.Empty("i") if v is None else v
            elif k in ('R2Ts', 'Ts2R'):
                meta_dict[k] = h5py.Empty("f") if v is None else v.numpy()
            elif k in ('i_type', 'f_type'):
                meta_dict[k] = str(v)
            else:
                meta_dict[k] = v
        meta.attrs.update(meta_dict)

        data = f.create_group('data')
        data_dict = {}
        for k, v in loc.data:
            data_dict[k] = v.numpy()
        data.attrs.update(data_dict)

        
def load_h5(path:Union[str, pathlib.Path]) -> Union[FrameLocSet, EmitterLocSet]:
    """ Loads a hdf5 file and returns a LocSet """
    
    dtype_mapping = {'torch.int16':torch.int16, 'torch.int32':torch.int32, 'torch.int64':torch.int64,
                     'torch.float16':torch.float16, 'torch.float32':torch.float32, 'torch.float64':torch.float64}
    
    with h5py.File(path, 'r') as h5:
        meta_dict = {}
        for k, v in h5['meta'].items():
            if k == 'zextent':
                meta_dict[k] = None if isinstance(v, h5py.Empty) else v
            elif k in ('R2Ts', 'Ts2R'):
                meta_dict[k] = None if isinstance(v, h5py.Empty) else torch.from_numpy(v)
            elif k in ('i_type', 'f_type'):
                meta_dict[k] = dtype_mapping[v]
            else:
                meta_dict[k] = v

        if 'indf' in h5['data'].keys():
            LocType = FrameLocSet
        elif 't0' in h5['data'].keys():
            LocType = EmitterLocSet
        else:
            raise TypeError(f"cannot identify if {path.name} is FrameLocSet or EmitterLocSet")
        data_dict = {k: torch.from_numpy(v) for k, v in h5['data'].items()}

    loc_dict = dict(meta_dict, **data_dict)
    return LocType(**loc_dict)


def save_torch(path:Union[str, pathlib.Path], loc:FrameLocSet) -> None:
    """ Save a LocSet to pt file """
    torch.save({'data': loc.data, 'meta': loc.meta}, path)


def load_torch(path:Union[str, pathlib.Path]) -> Union[FrameLocSet, EmitterLocSet]:
    """ Loads a torch saved LocSet and returns LocSet. """
    out = torch.load(path)
    if 'indf' in out['data'].keys():
        LocType = FrameLocSet
    elif 't0' in out['data'].keys():
        LocType = EmitterLocSet
    else:
        raise TypeError(f"cannot identify if {path.name} is FrameLocSet or EmitterLocSet")
    loc_dict = dict(out['meta'], **out['data'])
    return LocType(**loc_dict)


def save_pickle(path:Union[str, pathlib.Path], loc:FrameLocSet) -> None:
    """ Save a LocSet to pkl file """
    with open(path, 'wb') as f:
        pickle.dump({'meta':loc.meta, 'data':loc.data}, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(path:Union[str, pathlib.Path]) -> Union[FrameLocSet, EmitterLocSet]:
    """ Loads a pickle saved LocSet and returns LocSet. """
    with open(path, 'rb') as f:
        out = pickle.load(f)
        if 'indf' in out['duata'].keys():
            LocType = FrameLocSet
        elif 't0' in out['data'].keys():
            LocType = EmitterLocSet
        else:
            raise TypeError(f"cannot identify if {path.name} is FrameLocSet or EmitterLocSet")
    loc_dict = dict(out['meta'], **out['data'])
    return LocType(**loc_dict)


def save_xlsx(path:Union[str, pathlib.Path], loc:Union[FrameLocSet, EmitterLocSet]) -> None:
    """ Save a LocSet to Excel xlsx file in which the data and meta are in different sheets"""
    
    xyz_attr = ['x', 'y', 'z']
    xyz_sig_attr = ['x_sig', 'y_sig', 'z_sig']

    def _torch2numpy(data:dict) -> dict:
        """Convert all torch tensors in dict to numpy."""
        for k, v in data.items():
            if isinstance(v, torch.Tensor):
                data[k] = v.numpy()
        return data

    _chnltrans = False
    meta_rows = []
    for k, v in loc.meta.items():
        if k in ('R2Ts', 'Ts2R') and v is not None:
            _chnltrans = True
            meta_rows.append({'meta_key':k, 'meta_val':True})
        elif k in ('i_type', 'f_type'):
            meta_rows.append({'meta_key':k, 'meta_val':str(v)})
        else:
            meta_rows.append({'meta_key':k, 'meta_val':v})
    meta_df = pd.DataFrame(meta_rows)

    if _chnltrans:
        R2Ts_dict1d = {f'R2T{i+1}': loc.R2Ts[i].view(-1).numpy() for i in range(loc.nchannels - 1)}
        Ts2R_dict1d = {f'T{i+1}2R': loc.Ts2R[i].view(-1).numpy() for i in range(loc.nchannels - 1)}
        R2Ts_df = pd.DataFrame(R2Ts_dict1d)
        Ts2R_df = pd.DataFrame(Ts2R_dict1d)

    data_dict = _torch2numpy(loc.data)
    active_attr = loc.get_activeattr()
    data_dict1d = {}
    for k in active_attr:
        if k == 'xyz':
            xyz = data_dict.pop('xyz')
            data_dict1d.update({xyz_attr[i]: xyz[:, i] for i in range(loc.ndim)})
        elif k == 'xyz_sig':
            xyz_sig = data_dict.pop('xyz_sig')
            data_dict1d.update({xyz_sig_attr[i]: xyz_sig[:, i] for i in range(loc.ndim)})
        elif k == 'fracN':
            fracN = data_dict.pop('fracN')
            data_dict1d.update({f'f_ch{i}': fracN[:, i] for i in range(loc.nchannels)})
        elif k == 'fracN_sig':
            fracN_sig = data_dict.pop('fracN_sig')
            data_dict1d.update({f'f_ch{i}_sig': fracN_sig[:, i] for i in range(loc.nchannels)})
        else:
            data_dict1d.update({k: data_dict[k]})
    data_df = pd.DataFrame.from_dict(data_dict1d)

    with pd.ExcelWriter(path, engine='xlsxwriter') as writer:
        data_df.to_excel(writer, sheet_name='data', index=False)
        meta_df.to_excel(writer, sheet_name='meta', index=False)
        if _chnltrans:
            R2Ts_df.to_excel(writer, sheet_name='R2Ts', index=False)
            Ts2R_df.to_excel(writer, sheet_name='Ts2R', index=False)


def load_xlsx(path:Union[str, pathlib.Path]) -> Union[FrameLocSet, EmitterLocSet]:
    """ Loads a pickle saved LocSet and returns LocSet. """

    dtype_mapping = {'torch.int16':torch.int16, 'torch.int32':torch.int32, 'torch.int64':torch.int64,
                     'torch.float16':torch.float16, 'torch.float32':torch.float32, 'torch.float64':torch.float64}
    xyz_attr = ['x', 'y', 'z']
    xyz_sig_attr = ['x_sig', 'y_sig', 'z_sig']
    
    def _df2torch(df:pd.DataFrame, dtype:torch.dtype) -> torch.Tensor:
        return torch.from_numpy(df.to_numpy()).type(dtype)
    
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
            xyz = torch.stack([_df2torch(data_df[xyz_attr[i]], meta_dict['f_type']) for i in range(meta_dict['ndim'])], dim=1)
            active_attrs -= {xyz_attr[i] for i in range(meta_dict['ndim'])}
            data_dict['xyz'] = xyz
        if 'x_sig' in active_attrs:
            xyz_sig = torch.stack([_df2torch(data_df[xyz_sig_attr[i]], meta_dict['f_type']) for i in range(meta_dict['ndim'])], dim=1)
            active_attrs -= {xyz_attr[i] for i in range(meta_dict['ndim'])}
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
    
    loc_dict = {}
    loc_dict.update(meta_dict)
    loc_dict.update(data_dict)
    LocType = FrameLocSet if 'indf' in data_dict.keys() else EmitterLocSet
    return LocType(**loc_dict)