from typing import List, Tuple, Union, Callable

import torch


def specific_device_by_str(device:str) -> Tuple[str, int]:
    """ Converts torch compatible device string to device name and device index """
    
    if device != 'cpu' and device[:4] != 'cuda':
        raise ValueError("unsupported device string")

    if device == 'cpu':
        return 'cpu', None
    elif len(device) == 4:
        return 'cuda', None
    else:
        return 'cuda', int(device.split(':')[-1])


def get_device_capability() -> str:
    """Get the capability of the device"""
    
    capability = torch.cuda.get_device_capability()
    return f'{capability[0]}.{capability[1]}'


def get_device_type(device:Union[str, torch.device]) -> str:
    """ Returns the type of the device """
    
    if isinstance(device, torch.device):
        return device.type
    elif isinstance(device, str):
        if device == 'cpu':
            return 'cpu'
        elif device.startswith('cuda'):
            return 'cuda'
        else:
            raise ValueError("Unsupported device")


def get_max_batch_size(callable:Callable, x_size:Tuple, device:Union[str, torch.device], size_low:int, size_high:int) -> int:
    """
    Get the maximum batch size
    The maximum available batch size will be binary attempted (via a simple callable) between size_low and size_high
    Args:
        callable:   simple callable to test if a given batch size (together with *x_size) would be out of memory
        x_size:     the size (together with a given batch size) to test the memory
        device:     device to test on
        size_low:   lower limit of the attempted batch size
        size_high:  higher limit of the attempted batch size
    Returns:
        int:        maximum available batch size between size_low and size_high
    """
    
    if size_low > size_high:
        raise ValueError("Lower bound must be lower than upper bound.")

    bs = size_low
    bs_fail = size_high
    bs_pass = None

    while bs < size_high:
        try:
            x_try = torch.rand(bs, *x_size, device=device)
            callable(x_try)
            bs_pass = bs
        except RuntimeError as err:
            if 'CUDA out of memory.' not in str(err):
                raise err
            bs_fail = bs

        if bs_pass is None:
            raise RuntimeError("Lowest possible batch size is outside of provided bounds.")

        bs_new = int((bs_fail + bs_pass) / 2)
        if bs_new == bs:
            break
        bs = bs_new

    del x_try
    torch.cuda.empty_cache()
    return bs


def to_device(x:Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor], dict], device:str) -> Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor], dict]:
    """
    Move tensor(s) to device. 
    Callers should reassign the result to ensure proper reference counting to free GPU memories
    Example:
        >>> x = to_device(x)
        >>> a, b, c = to_device([a, b, c])
        >>> a_dict = to_device(a_dict)
    Args:
        x:      Can be a single tensor or a list/tuple/dictionary of tensors
        device: device to ship to
    Returns:
        Object of the same type with all tensors on desired device
    """
    if device.startswith('cuda'):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available on this system.")
        if device == 'cuda':
            device = f'cuda:{torch.cuda.current_device()}'
    target_device = torch.device(device)
    
    if x is None:
        return x
    elif isinstance(x, torch.Tensor):
        if x.device == target_device:
            return x
        return x.to(device)
    elif isinstance(x, (tuple, list)):
        return [to_device(elm, device) for elm in x]
    elif isinstance(x, dict):
        return {key: to_device(val, device) for key, val in x.items()}
    else:
        raise TypeError("Only support for single tensor or tuple/list/dict of tensors")