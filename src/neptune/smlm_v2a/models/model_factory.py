import warnings
from typing import Union, Tuple

import torch
import torch.nn as nn
from torch.jit import TracerWarning


def get_activation(activation:Union[str, nn.Module], kwargs:dict=None):
    """Create activation function instance"""
    
    activations_ = ['ReLU', 'LeakyReLU', 'ELU', 'PReLU', 'SELU', 'GELU', 'SiLU', 
                    'Mish', 'Tanh', 'Sigmoid', 'Identity']
    kwargs = {} if kwargs is None else kwargs

    if activation is None:
        return None
    elif isinstance(activation, str):
        if activation not in activations_:
            raise ValueError(f"Unknown activation: {activation}")
        if activation == 'LeakyReLU' and 'negative_slope' not in kwargs:
            kwargs['negative_slope'] = 0.01
        if activation in ['ReLU', 'LeakyReLU', 'ELU', 'SELU', 'SiLU', 'Mish']:
            return getattr(nn, activation)(inplace=True, **kwargs)
        else:
            return getattr(nn, activation)(**kwargs)    
    elif isinstance(activation, nn.Module):
        return activation
    else:
        raise TypeError("activation must be str or nn.Module")
    
    

def weight_by_gradient(layer:torch.nn.ModuleList, loss:torch.Tensor, optimizer:torch.optim.Optimizer) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Args:
        layer:      module layers
        loss:       (N, C, H, W) not-reduced loss values
        optimizer:  optimizer
    Returns:
        torch.Tensor:   (1, C=1, 1, 1) weight, weight per channel
        torch.Tensor:   (C,) loss_ch, channel-wise loss
        torch.Tensor:   float, weighted sum of loss_ch
    """

    channels_out = len(layer)
    loss_ch = loss.sum(-1).sum(-1).sum(0) / loss.numel() * channels_out
    
    head_grads = torch.zeros((channels_out,), dtype=loss.dtype, device=loss.device)
    weighting = torch.ones((channels_out,), dtype=loss.dtype, device=loss.device)

    for i in range(channels_out):
        head_grads[i] = torch.autograd.grad(loss_ch[i], layer[i].out_conv.weight, retain_graph=True)[0].abs().sum()
    idx_on = head_grads != 0.
    
    optimizer.zero_grad()
    norm = (1.0 / head_grads[idx_on]).sum()
    weighting[idx_on] = weighting[idx_on] / head_grads[idx_on]
    weighting[~idx_on] = 0.0
    weighting = weighting / norm

    loss_wch = (loss_ch * weighting).sum()
    weight = weighting.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

    return weight, loss_ch, loss_wch



class MultiHeads(nn.Module):
    """Multiple heads on UNet outputs"""

    def __init__(self, in_channels:int, out_channels:int, last_kernelsz:int, 
                 pad_convs:bool=True, norm_groups:int=0, activation:Union[str, nn.Module]='ReLU', ndim:int=2):
        """
        Args:
            in_channels:            number of input channels
            out_channels:           number of output channels
            last_kernelsz:          kernel size for nn.Conv2d
            pad_convs:              whether to use padded convolutions (padding = 1 if pad_convs)
            norm_groups:            the number of groups for GroupNorm, 0 for BatchNorm, <0 for no normalization
            activation:             str or nn.Module, the activation
        """
        
        super().__init__()
        if ndim not in {2, 3}:
            raise ValueError(f"{ndim} is unsupported number of dimensions, only 2 or 3 is available")
        _Conv = nn.Conv3d if ndim == 3 else nn.Conv2d
        
        self.padding = 1 if pad_convs else 0
        if norm_groups > 0:
            if in_channels % norm_groups != 0:
                raise ValueError("in_channels must be divisible by norm_groups.")
        self.norm_groups = norm_groups
        
        self.in_conv = _Conv(in_channels, in_channels, kernel_size=3, padding=self.padding)
        if self.norm_groups > 0:
            self.norm = nn.GroupNorm(self.norm_groups, in_channels) 
        elif self.norm_groups == 0:
            self.norm = nn.BatchNorm3d(in_channels) if ndim == 3 else nn.BatchNorm2d(in_channels)
        else:
            self.norm = nn.Identity()
        self.activation = get_activation(activation)
        self.out_conv = _Conv(in_channels, out_channels, kernel_size=last_kernelsz, padding=0)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        """ forward the x tensor of shape (N, C, (Depth, )H, W) through the output head """
        
        x = self.in_conv(x)
        x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        x = self.out_conv(x)
        return x
    
    def forward_(self, x):
        print(f"initial shape = {x.shape}")
        x = self.in_conv(x)
        print(f"after in_conv shape = {x.shape}")
        x = self.norm(x)
        print(f"after group norm shape = {x.shape}")
        if self.activation is not None:
            x = self.activation(x)
            print(f"after activation shape = {x.shape}")
        x = self.out_conv(x)
        print(f"after out_conv shape = {x.shape}")
        return x
    
