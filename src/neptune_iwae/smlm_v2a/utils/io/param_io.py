from typing import Union, Tuple

from pathlib import Path
import importlib.resources as pkg_resources

import json
import yaml
import torch

from .calibr_io import load_calibrations
from ..param_types import RecursiveNamespace, load_template, autofill_dict

    

class ParamHandling:
    file_extensions = ['.json', '.yml', '.yaml']
    
    def __init__(self):
        self.params_dot = None

    def _check_return_extension(self, filename:Path) -> str:
        """check if the extension of the input filename is valid"""
        extension = filename.suffix
        if extension not in self.file_extensions:
            raise ValueError(f"Filename must be in {self.file_extensions}")
        return extension
    
    def load_params(self, filename:Union[str, Path]) -> Tuple[RecursiveNamespace, RecursiveNamespace]:
        """
        load parameters from file
        Args:
            filename:   the filename of a parameter .json or .yaml file
        Returns:
            RecursiveNameSpace: parameters
            RecursiveNameSpace: calibration loaded from parameters.CalibrInOut
        """
        filename = Path(filename) if not isinstance(filename, Path) else filename
        extension = self._check_return_extension(filename)
        if extension == '.json':
            with open(filename) as json_file:
                params_dict = json.load(json_file)
        else:
            with open(filename) as yaml_file:
                params_dict = yaml.safe_load(yaml_file)

        params_ref = load_template('param_reference')
        params_dict = autofill_dict(params_dict, params_ref)
        params = RecursiveNamespace(**params_dict)

        # Set up the calibration
        calibration = load_calibrations(params.CalibrInOut.cam_path, params.CalibrInOut.csplinePSF_path, params.CalibrInOut.chnlT_path)
        
        # Set up some of the parameters
        params = autofill_from_calibration(params, calibration)
        params = autoset_scaling(params)
        params = set_dtypes(params)

        self.params_dot = params
        return params, calibration

    def write_params(self, filename:Union[str, Path], params:Union[dict, RecursiveNamespace]):
        """write parameters to filename"""
        filename = Path(filename) if not isinstance(filename, Path) else filename
        try:
            filename.parents[0].mkdir(parents=False, exist_ok=True)
        except FileNotFoundError:
            raise FileNotFoundError(f"cannot find {filename.parents[1]} to create folder {filename.parents[0]}")
        
        if isinstance(params, RecursiveNamespace):
            params = params.to_dict()
        params = print_dtypes(params)

        extension = self._check_return_extension(filename)
        if extension == '.json':
            with filename.open('w') as json_file:
                json.dump(params, json_file, indent=4)
        else:
            with filename.open('w') as yaml_file:
                yaml.dump(params, yaml_file)

    def convert_param_file(self, fname_scr:Union[str, Path], fname_dst:Union[str, Path]):
        """converting file from and to json / yaml"""
        params = self.load_params(fname_scr)[0]
        self.write_params(fname_dst, params)



def load_params(filename:Union[str, Path]) -> Tuple[RecursiveNamespace, RecursiveNamespace]: # alias
    return ParamHandling().load_params(filename)


def save_params(filename:Union[str, Path], params:Union[dict, RecursiveNamespace]): # alias
    ParamHandling().write_params(filename, params)


def autofill_from_calibration(params:RecursiveNamespace, calibr:RecursiveNamespace) -> RecursiveNamespace:
    """Automatically set some parameters from calibration"""
    
    if calibr.ndim != params.Simulation.ndim:
        raise ValueError(f"ndim mismatch: calibration.ndim={calibr.ndim}, Simulation.ndim={params.Simulation.ndim}")
    if calibr.nchannels != params.Simulation.nchannels:
        raise ValueError(f"nchannels mismatch: calibration.nchannels={calibr.nchannels}, Simulation.nchannels={params.Simulation.nchannels}")
    
    params.Camera.cam_pxszs = calibr.Camera.cam_pxszs
    
    if calibr.ndim == 3:
        if len(params.Simulation.emitter_extent) == 2:
            params.Simulation.emitter_extent.append([0, calibr.CSplinePSF.coeff_sizes[2]])
        elif params.Simulation.emitter_extent[2] is None:
            params.Simulation.emitter_extent[2] = [0, calibr.CSplinePSF.coeff_sizes[2]]
        else:
            params.Simulation.emitter_extent[2][0] = max(0, params.Simulation.emitter_extent[2][0])
            params.Simulation.emitter_extent[2][1] = min(calibr.CSplinePSF.coeff_sizes[2], params.Simulation.emitter_extent[2][1])

    return params    


def autoset_scaling(params:RecursiveNamespace) -> RecursiveNamespace:
    """Automatically set some scaling parameters if not provided"""
    set_if_none = lambda arg, value : value if arg is None else arg

    # autoset Scaling.input_scale and phot_max according to Simulation.photon_mu_sig
    _photon_mu_sig = params.Simulation.photon_mu_sig
    params.Scaling.input_scale = set_if_none(params.Scaling.input_scale, _photon_mu_sig[0] / 50.0)
    params.Scaling.phot_max = set_if_none(params.Scaling.phot_max, _photon_mu_sig[0] + 8.0 * _photon_mu_sig[1])

    # autoset Scaling.z_max according to Simulation.emitter_extent[2]
    if params.Simulation.ndim == 3:
        _zextent = params.Simulation.emitter_extent[2]
        params.Scaling.z_max = set_if_none(params.Scaling.z_max, 1.2 * _zextent[1])

    # autoset Scaling.input_offset and bkg_max according to Simulation_bkg_uniform
    _bkg_uniform = params.Simulation.bkg_uniform
    if isinstance(_bkg_uniform, (list, tuple)):
        _input_offset = 0.5 * (_bkg_uniform[0] + _bkg_uniform[1])
        _bkg_max = 1.2 * _bkg_uniform[1]
    else:
        _input_offset = _bkg_uniform
        _bkg_max = 1.2 * _bkg_uniform
    params.Scaling.input_offset = set_if_none(params.Scaling.input_offset, _input_offset)
    params.Scaling.bkg_max = set_if_none(params.Scaling.bkg_max, _bkg_max)

    return params


def set_dtypes(params:RecursiveNamespace) -> RecursiveNamespace:
    """Set the i_type and f_type for data types"""
    type_mapping = {'torch.int16':torch.int16, 'torch.int32':torch.int32, 'torch.int64':torch.int64,
                    'int16':torch.int16, 'int32':torch.int32, 'int64':torch.int64,
                    'short':torch.int16, 'int':torch.int32, 'long':torch.int64,
                    'torch.float16':torch.float16, 'torch.float32':torch.float32, 'torch.float64':torch.float64,
                    'float16':torch.float16, 'float32':torch.float32, 'float64':torch.float64,
                    'half':torch.float16, 'float':torch.float32, 'double':torch.float64}
    params.DataType.i_type = type_mapping[params.DataType.i_type] if params.DataType.i_type is not None else torch.int32
    params.DataType.f_type = type_mapping[params.DataType.f_type] if params.DataType.f_type is not None else torch.float32
    return params


def print_dtypes(params:dict) -> dict:
    """convert the dtype to strings for writing purpose"""
    params['DataType']['i_type'] = params['DataType']['i_type'].__str__() # lazy assumptions that the dtype is torch.dtype
    params['DataType']['f_type'] = params['DataType']['f_type'].__str__()
    return params


def copy_reference_params(path:Union[str, Path]):
    """Copy the reference params to the input path"""
    path  = Path(path) if not isinstance(path, Path) else path
    assert path.is_dir()

    from .. import reference_files
    pr = pkg_resources.read_text(reference_files, 'param_reference.yaml')
    (path / "param_reference.yaml").write_text(pr)
    

def convert_param_debug(param:RecursiveNamespace) -> RecursiveNamespace:
    """change some parameters for debug"""
    param.ExpInOut.model_out = 'network/debug.pt'
    param.Simulation.train_ds_size = 1024
    param.Simulation.test_ds_size = 128
    return param