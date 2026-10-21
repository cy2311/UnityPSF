from abc import ABC, abstractmethod
import os
from typing import Union
import numbers

import torch
import torch.nn as nn

from .unet import UNet2d
from .model_factory import MultiHeads



class SMLMUNet(nn.Module, ABC):

    out_channels_heads = (1, 4, 4, 1) # p, pxyz_mu, pxyz_sig, bg
    ch_ix_sigmoid = [0, 1, 4, 5, 6, 7, 8, 9] # channel indicies with respect to activation function
    ch_ix_tanh = [2, 3]

    ch_ix_p = [0]
    ch_ix_pxyz_mu = slice(1, 5)
    ch_ix_pxyz_sig = slice(5, 9)
    ch_ix_bg = [9]
    
    sigma_eps_default = 0.001

    @staticmethod
    def _infer_z_mu_activation(param) -> str | None:
        arch_param = getattr(getattr(param, 'HyperParameter', None), 'arch_param', None)
        explicit = getattr(arch_param, 'z_mu_activation', None)
        if explicit is not None and str(explicit).strip() != '':
            return str(explicit).strip().lower()
        return 'tanh'

    def __init__(self, *, nch_in:int, nfeatures_init:int, nfeatures_inter:int=None,
                 norm_start_level:int=0, norm_groups:int=0, activation:Union[str, nn.Module]='ELU', 
                 dropout_start_level:int=None, p_dropout:float=-0.1, pool_mode:str='StrideConv', upsample_mode:str='bilinear', 
                 inter_activation:Union[str, nn.Module]='ELU', 
                 norm_head_groups:int=0, final_activation:Union[str, nn.Module]='ELU', 
                 disabled_attr:Union[int, tuple, list]=None, kaiming_normal:bool=True, z_mu_activation:str|None=None):
        """
        Abstract class for CNN-based shallow UNet for Single Molecule Localization 
        Args:
            nch_in:                 number of input channels
            nfeatures_init:         number of channels entering the UNet
            nfeatures_inter:        number of output channels for the convolution layer inbetween the UNet and MLTHeads, 
                                    set to None if there are no convolution layer between the UNet and the MLTHeads
            norm_start_level:       the level in the UNet to start normalization
            norm_groups:            options for normalization (<0 for no norm, =0 for batch norm, >0 for the gourp size of group norm)
            activation:             activation following the convolution in the CNN-based UNet
            dropout_start_level:    the level in the UNet to start dropout
            p_dropout:              options for dropout (<=0 for no dropout, >0 for dropout probability)
            pool_mode:              pooling mode
                                    MaxPool: nn.MaxPool(2)
                                    StrideConv: nn.Conv2d(in_channels, in_channels, kernel_size=2, stride=2)
            upsample_mode:          upsampling mode for F.interpolate
            inter_activation:       the activation following the intermediate convolution between the UNet and the MLTHeads
            norm_head_groups:       options for normalization in the MLTHeads (<0 for no norm, =0 for batch norm, >0 for the gourp size of group norm)
            final_activation:       activation before output
            disabled_attr:          disabled attributes
            kaiming_normal:         option to apply kaiming normalization
        """
        super().__init__()

        # Register sigma as parameter such that it is stored in the models state dict and loaded correctly
        self.register_parameter('sigma_eps', nn.Parameter(torch.tensor([self.sigma_eps_default]), requires_grad=False))
        self.nch_in = nch_in
        self.nfeatures_init = nfeatures_init
        self.nfeatures_inter = nfeatures_inter
        self.activation = activation
        self.norm_start_level = norm_start_level
        self.norm_groups = norm_groups
        self.dropout_start_level = dropout_start_level
        self.p_dropout = p_dropout
        self.pool_mode = pool_mode
        self.upsample_mode = upsample_mode
        self.inter_activation = inter_activation
        self.norm_head_groups = norm_head_groups
        self.final_activation = final_activation
        self.disabled_attr_ix = [disabled_attr] if isinstance(disabled_attr, numbers.Integral) else disabled_attr
        self.kaiming_normal = kaiming_normal

        # Allow overriding z_mu activation per run (e.g. for symmetric z ranges).
        z_mu_act = os.getenv("SMLM_Z_MU_ACT", "").strip().lower()
        if not z_mu_act and z_mu_activation is not None:
            z_mu_act = str(z_mu_activation).strip().lower()
        if not z_mu_act:
            z_mu_act = "tanh"
        if z_mu_act:
            # Copy class defaults to avoid mutating shared lists.
            self.ch_ix_sigmoid = list(self.ch_ix_sigmoid)
            self.ch_ix_tanh = list(self.ch_ix_tanh)
            if z_mu_act == "tanh":
                if 4 in self.ch_ix_sigmoid:
                    self.ch_ix_sigmoid.remove(4)
                if 4 not in self.ch_ix_tanh:
                    self.ch_ix_tanh.append(4)
            elif z_mu_act == "sigmoid":
                if 4 in self.ch_ix_tanh:
                    self.ch_ix_tanh.remove(4)
                if 4 not in self.ch_ix_sigmoid:
                    self.ch_ix_sigmoid.append(4)
        
        self.mt_heads = nn.ModuleList()
        _in_channels = self.nfeatures_init if self.nfeatures_inter is None else self.nfeatures_inter
        for _out_channels in self.out_channels_heads:
            self.mt_heads.append(MultiHeads(in_channels=_in_channels, out_channels=_out_channels, last_kernelsz=1,
                                            pad_convs=True, norm_groups=self.norm_head_groups, activation=self.final_activation))

    @staticmethod
    def weight_init(m:nn.Module):
        """ Applies Kaiming normal init. Call this recursively by model.apply(model.weight_init) """
        
        if isinstance(m, (nn.Conv2d, nn.Conv3d)):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm3d, nn.GroupNorm)):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)
    
    def _apply_nonlinearity(self, o:torch.Tensor) -> torch.Tensor:
        """ Applies non-linearity accordingly to the unet output of shape (N, nch_out, (Depth, )H, W) """
        
        # clamp the detection before nonlinearities
        o[:, [0]] = torch.clamp(o[:, [0]], min=-8.0, max=8.0)  # Prevent overflow
        
        # apply nonlinearities
        o[:, self.ch_ix_sigmoid] = torch.sigmoid(o[:, self.ch_ix_sigmoid])
        o[:, self.ch_ix_tanh] = torch.tanh(o[:, self.ch_ix_tanh])

        # add epsilon to sigmas and rescale
        o[:, self.ch_ix_pxyz_sig] = 3.0 * o[:, self.ch_ix_pxyz_sig] + self.sigma_eps

        # disabled attributes get set to constants
        if self.disabled_attr_ix is not None:
            for ix in self.disabled_attr_ix:
                o[:, 1 + ix] = 0.0
                o[:, 5 + ix] = 0.1
        return o
    
    @abstractmethod
    def forward(self, x:torch.Tensor):
        raise NotImplementedError



class SingleUNet(SMLMUNet):

    def __init__(self, *, nch_in:int, depth:int, nfeatures_init:int, nfeatures_inter:int=None,
                 norm_start_level:int=0, norm_groups:int=0, activation:Union[str, nn.Module]='ELU', 
                 dropout_start_level:int=None, p_dropout:float=0.1, pool_mode:str='StrideConv', upsample_mode:str='bilinear',
                 inter_activation:Union[str, nn.Module]='ELU',
                 norm_head_groups:int=0, final_activation:Union[str, nn.Module]='ELU',
                 disabled_attr:Union[int, tuple, list]=None, kaiming_normal:bool=True, depthwise:bool=True,
                 z_mu_activation:str|None=None):
        
        super().__init__(nch_in=nch_in, nfeatures_init=nfeatures_init, nfeatures_inter=nfeatures_inter,
                         norm_start_level=norm_start_level, norm_groups=norm_groups, activation=activation, 
                         dropout_start_level=dropout_start_level, p_dropout=p_dropout, pool_mode=pool_mode, upsample_mode=upsample_mode,
                         inter_activation=inter_activation, norm_head_groups=norm_head_groups, final_activation=final_activation,
                         disabled_attr=disabled_attr, kaiming_normal=kaiming_normal, z_mu_activation=z_mu_activation)

        self.depth = depth
        self.unet = UNet2d(nch_in=self.nch_in, nch_out=self.nfeatures_inter, depth=self.depth, nfeatures_init=self.nfeatures_init, pad_convs=True, 
                           norm_start_level=self.norm_start_level, norm_groups=self.norm_groups, activation=self.activation,  
                           dropout_start_level=self.dropout_start_level, p_dropout=self.p_dropout, 
                           pool_mode=self.pool_mode, upsample_mode=self.upsample_mode, out_activation=self.inter_activation,
                           depthwise=depthwise)

        if self.kaiming_normal:
            self.apply(self.weight_init)
            nn.init.kaiming_normal_(self.mt_heads[0].in_conv.weight, mode='fan_in', nonlinearity='relu')
            nn.init.kaiming_normal_(self.mt_heads[0].out_conv.weight, mode='fan_in', nonlinearity='linear')
            nn.init.constant_(self.mt_heads[0].out_conv.bias, -6.0) 
        
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = self.unet(x)
        x_heads = [mt_head(x) for mt_head in self.mt_heads]
        o = torch.cat(x_heads, dim=1)
        return self._apply_nonlinearity(o)
    
    @classmethod
    def parse(cls, param):
        return cls(nch_in = param.HyperParameter.channels_in,
                   depth = param.HyperParameter.arch_param.depth,
                   nfeatures_init = param.HyperParameter.arch_param.nfeatures_init,
                   nfeatures_inter = param.HyperParameter.arch_param.nfeatures_inter,
                   norm_start_level = param.HyperParameter.arch_param.norm_start_level,
                   norm_groups = param.HyperParameter.arch_param.norm_groups,
                   activation = param.HyperParameter.arch_param.activation,
                   dropout_start_level = param.HyperParameter.arch_param.dropout_start_level,
                   p_dropout = param.HyperParameter.arch_param.p_dropout,
                   pool_mode = param.HyperParameter.arch_param.pool_mode,
                   upsample_mode = param.HyperParameter.arch_param.upsample_mode,
                   inter_activation = param.HyperParameter.arch_param.inter_activation,
                   norm_head_groups = param.HyperParameter.arch_param.norm_head_groups,
                   final_activation = param.HyperParameter.arch_param.final_activation,
                   disabled_attr = param.DataFilter.disabled_attributes,
                   kaiming_normal = param.HyperParameter.arch_param.kaiming_normal,
                   depthwise = param.HyperParameter.arch_param.depthwise,
                   z_mu_activation = cls._infer_z_mu_activation(param))
    


class DoubleUNet(SMLMUNet):

    def __init__(self, *, nch_in:int, depth_shared:int, depth_union:int, nfeatures_init:int, nfeatures_inter:int=None,
                 norm_start_level:int=0, norm_groups:int=0, activation:Union[str, nn.Module]='ELU', 
                 dropout_start_level:int=None, p_dropout:float=0.1, pool_mode:str='StrideConv', upsample_mode:str='bilinear',
                 inter_activation:Union[str, nn.Module]='ELU',
                 norm_head_groups:int=0, final_activation:Union[str, nn.Module]='ELU',
                 disabled_attr:Union[int, tuple, list]=None, kaiming_normal:bool=True, depthwise:bool=True,
                 z_mu_activation:str|None=None):
        
        super().__init__(nch_in=nch_in, nfeatures_init=nfeatures_init, nfeatures_inter=nfeatures_inter,
                         norm_start_level=norm_start_level, norm_groups=norm_groups, activation=activation, 
                         dropout_start_level=dropout_start_level, p_dropout=p_dropout, pool_mode=pool_mode, upsample_mode=upsample_mode,
                         inter_activation=inter_activation, norm_head_groups=norm_head_groups, final_activation=final_activation,
                         disabled_attr=disabled_attr, kaiming_normal=kaiming_normal, z_mu_activation=z_mu_activation)
        
        self.depth_shared = depth_shared
        self.unet_shared = UNet2d(nch_in=1, nch_out=nfeatures_inter, depth=self.depth_shared, nfeatures_init=self.nfeatures_init, pad_convs=True, 
                                  norm_start_level=self.norm_start_level, norm_groups=self.norm_groups, activation=self.activation,  
                                  dropout_start_level=self.dropout_start_level, p_dropout=self.p_dropout, 
                                  pool_mode=self.pool_mode, upsample_mode=self.upsample_mode, out_activation=inter_activation,
                                  depthwise=depthwise) 

        self.depth_union = depth_union
        self.unet_union = UNet2d(nch_in=self.nch_in*self.nfeatures_init, nch_out=self.nfeatures_inter, depth=self.depth_union, nfeatures_init=self.nfeatures_init, pad_convs=True, 
                                 norm_start_level=self.norm_start_level, norm_groups=self.norm_groups, activation=self.activation,
                                 dropout_start_level=self.dropout_start_level, p_dropout=self.p_dropout,
                                 pool_mode=self.pool_mode, upsample_mode=self.upsample_mode, out_activation=inter_activation,
                                 depthwise=depthwise) 

        if kaiming_normal:
            self.apply(self.weight_init)
            nn.init.kaiming_normal_(self.mt_heads[0].in_conv.weight, mode='fan_in', nonlinearity='relu')
            nn.init.kaiming_normal_(self.mt_heads[0].out_conv.weight, mode='fan_in', nonlinearity='linear')
            nn.init.constant_(self.mt_heads[0].out_conv.bias, -6.0)

    def _forward_core(self, x:torch.Tensor) -> torch.Tensor:
        if self.nch_in > 1:
            outputs = [self.unet_shared(x[:, [ix]]) for ix in range(self.nch_in)]
            o = torch.cat(outputs, dim=1)
        elif self.nch_in == 1:
            o = self.unet_shared(x)
        else:
            raise ValueError(f"nch_in={self.nch_in} is invalid, should be positive integers")
        return self.unet_union(o)
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = self._forward_core(x)
        x_heads = [mt_head(x) for mt_head in self.mt_heads]
        o = torch.cat(x_heads, dim=1)
        return self._apply_nonlinearity(o)
    
    @classmethod
    def parse(cls, param):
        return cls(nch_in = param.HyperParameter.channels_in,
                   depth_shared = param.HyperParameter.arch_param.depth_shared,
                   depth_union = param.HyperParameter.arch_param.depth_union,
                   nfeatures_init = param.HyperParameter.arch_param.nfeatures_init,
                   nfeatures_inter = param.HyperParameter.arch_param.nfeatures_inter,
                   norm_start_level = param.HyperParameter.arch_param.norm_start_level,
                   norm_groups = param.HyperParameter.arch_param.norm_groups,
                   activation = param.HyperParameter.arch_param.activation,
                   dropout_start_level = param.HyperParameter.arch_param.dropout_start_level,
                   p_dropout = param.HyperParameter.arch_param.p_dropout,
                   pool_mode = param.HyperParameter.arch_param.pool_mode,
                   upsample_mode = param.HyperParameter.arch_param.upsample_mode,
                   inter_activation = param.HyperParameter.arch_param.inter_activation,
                   norm_head_groups = param.HyperParameter.arch_param.norm_head_groups,
                   final_activation = param.HyperParameter.arch_param.final_activation,
                   disabled_attr = param.DataFilter.disabled_attributes,
                   kaiming_normal = param.HyperParameter.arch_param.kaiming_normal,
                   depthwise = param.HyperParameter.arch_param.depthwise,
                   z_mu_activation = cls._infer_z_mu_activation(param))
