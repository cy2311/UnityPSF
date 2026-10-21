from typing import Union, Tuple 
import datetime
import socket
from copy import deepcopy
import struct

import numpy as np
import torch



def current_info(comment:str=None) -> str:
    """returns a string with current time and host information, and with a comment string if available"""
    info = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + '_' + socket.gethostname()
    if comment is not None:
        info += '_' + comment
    return info




def read_roi(roi_fname):
    
    with open(roi_fname, 'rb') as fid:
        roi_info = fid.read()
    
    # the first 4 byte should be a string "Iout"
    strMagic = "".join([chr(b) for b in roi_info[:4]])
    if strMagic != 'Iout':
        raise TypeError("Invalid ImageJ roi")
    
    type_code = roi_info[6]                             # roi_type: the 6th byte
    if type_code == 1:
        roi_type = 'rect'
        top     = (roi_info[8]<<8) + roi_info[9]        # top: the 8-9th bytes
        left    = (roi_info[10]<<8) + roi_info[11]      # left: the 10-11th bytes
        bottom  = (roi_info[12]<<8) + roi_info[13]      # bottom: the 12-13th bytes
        right   = (roi_info[14]<<8) + roi_info[15]      # right: the 14-15th bytes
        roi = [top, left, bottom, right]
    elif type_code == 3:
        roi_type = 'line'
        x1 = struct.unpack('>f', bytes([roi_info[i] for i in range(18, 22)]))[0]
        y1 = struct.unpack('>f', bytes([roi_info[i] for i in range(22, 26)]))[0]
        x2 = struct.unpack('>f', bytes([roi_info[i] for i in range(26, 30)]))[0]
        y2 = struct.unpack('>f', bytes([roi_info[i] for i in range(30, 34)]))[0]
        roi = [[x1, y1], [x2, y2]] if x1 < x2 else [[x2, y2], [x1, y1]]
    else:
        raise TypeError("Only rectangle and line type of roi is supported")
    return roi_type, roi









def set_device(device:Union[torch.device, str, None]) -> torch.device:
    if device is None:
        return torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    else:
        return torch.device(device) if isinstance(device, str) else device









def to_tensor(arr:Union[list, tuple, np.ndarray, torch.Tensor], dtype:torch.dtype=None, device:Union[torch.device, str]=None)->torch.Tensor:
    """converts an input array to torch.Tensor with desired dtype and device"""
    if isinstance(arr, torch.Tensor):
        tensor = arr
    elif isinstance(arr, np.ndarray):
        tensor = torch.from_numpy(arr)
    else:
        tensor = torch.tensor(arr) 
    
    if (dtype is not None) and (device is not None):
        tensor = tensor.to(device, dtype=dtype)
    elif dtype is not None:
        tensor = tensor.to(dtype=dtype)
    elif device is not None:
        tensor = tensor.to(device)
    return tensor


def to_tensorslicer(ix: Union[int, list, slice, np.ndarray, torch.Tensor]) -> Union[list, slice, np.ndarray, torch.Tensor]:
    """
    Ensure indexing is compatible with torch tensor operations.
    NOTE:
    PyTorch automatically handles numpy integer and boolean arrays in indexing, so explicit conversion is unnecessary.
    """
    # Handle empty cases first for early optimization
    if isinstance(ix, (list, np.ndarray, torch.Tensor)) and len(ix) == 0:
        return [] 
    
    # Handle all-false boolean cases
    if isinstance(ix, list) and all([isinstance(x, bool) for x in ix]) and not any(ix):
        return []
    if isinstance(ix, (np.ndarray, torch.Tensor)) and ix.dtype in (np.bool_, torch.bool) and not ix.any():
        return [] 
    
    # Handle empty slices
    # NOTE: there are other cases that result in empty selection, but needs the tensor length to determine
    #       so leave it to handle externally
    if isinstance(ix, slice) and  ix.start == ix.stop:
        return []
        
    # Handle scalar types - convert to single-element lists for consistent behavior
    if isinstance(ix, int):
        return [ix]
    elif isinstance(ix, np.integer):
        return [int(ix)]
    elif isinstance(ix, torch.Tensor) and ix.dtype != torch.bool and ix.numel() == 1:
        return [int(ix)]
    
    # For lists, slice, numpy arrays, and torch tensors - return as-is since PyTorch handles them
    return ix