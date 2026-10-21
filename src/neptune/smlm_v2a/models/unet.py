from abc import ABC, abstractmethod
from typing import Union, Optional
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.jit import TracerWarning

from .model_factory import get_activation

"""
by constantin pape
from https://github.com/constantinpape/mu-net
"""


class ConvBlock(nn.Module):
    """
    Convolutional block with optional normalization
    N, C_in, H, W --> N, C_out, H, W via convolution 
    """
    
    def __init__(self, level:int, in_channels:int, out_channels:int, padding:int=1, 
                 norm_start_level:int=0, norm_groups:int=0, 
                 activation:Union[str, nn.Module]='ReLU', activation_kwargs:dict=None, 
                 dropout_start_level:int=3, p_dropout:float=0.1, 
                 ndim:int=2, depthwise:bool=True):
        """
        Args:
            level:                  the level where the covolution block is in a UNet
            in_channels:            number of channels of the input
            out_channels:           number of channels of the output
            padding:                padding for Convolution
            norm_start_level:       the level that a UNet starts normalization
            norm_groups:            number of groups for Group_Normalization, set to 0 for Batch_Normalization, set to negative for No_Normalization
            activation:             activation function
            activation_kwargs:      dictionary containing the keyword arguments corresponding to the activation function
            dropout_start_level:    the level that a UNet starts dropout
            p_dropout:              portion for dropout, set to 0 or negative for No_Dropout
            ndim:                   dimensionality for convolution
            depthwise:              set True for depthwise convolution to improve the computation efficiency
        """
        super().__init__()

        # Convolution
        self.conv1 = self.create_conv(in_channels, out_channels, padding, ndim, depthwise)
        self.conv2 = self.create_conv(out_channels, out_channels, padding, ndim, depthwise)

        # Normalization
        if norm_start_level is not None and level >= norm_start_level and norm_groups >= 0:
            if norm_groups > 0:
                if out_channels % norm_groups != 0:
                    raise ValueError("number of groupping channels (norm_groups) should be divisible by out_channels (out_channels)")
                self.norm1 = nn.GroupNorm(norm_groups, out_channels)
                self.norm2 = nn.GroupNorm(norm_groups, out_channels)
            else:
                self.norm1 = nn.BatchNorm3d(out_channels) if ndim == 3 else nn.BatchNorm2d(out_channels)
                self.norm2 = nn.BatchNorm3d(out_channels) if ndim == 3 else nn.BatchNorm2d(out_channels)
        else:
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()
        
        # Activation 
        self.activation1 = get_activation(activation, activation_kwargs)
        self.activation2 = get_activation(activation, activation_kwargs)

        # Dropout
        if dropout_start_level is not None and level >= dropout_start_level and p_dropout > 0:
            self.dropout = nn.Dropout3d(p_dropout) if ndim == 3 else nn.Dropout2d(p_dropout)
        else:
            self.dropout = nn.Identity()

    @staticmethod
    def create_conv(in_channels:int, out_channels:int, padding:int, ndim:int, depthwise:bool=True) -> Union[nn.Sequential, nn.Module]:
        """ 
        Creates a convolution instance 
        Args:
            in_channels:    number of channels of the input
            out_channels:   number of channels of the output
            padding:        padding for Convolution
            ndim:           dimensionality for convolution
            depthwise:      set True for depthwise convolution to improve the computation efficiency
        Returns:
            nn.Module:      the convolution instance    
        """ 
        _Conv = nn.Conv3d if ndim == 3 else nn.Conv2d
        if depthwise:
            return nn.Sequential(_Conv(in_channels, in_channels, kernel_size=3, padding=padding, groups=in_channels),
                                 _Conv(in_channels, out_channels, kernel_size=1))
        else:
            return _Conv(in_channels, out_channels, kernel_size=3, padding=padding)
        
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        """forward the input tensor x of shape (N, C, (Depth, )H, W) through the convolution block"""
        # First convolution sequence
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.activation1(x)
        
        # Second convolution sequence
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.activation2(x)
        
        # Dropout
        x = self.dropout(x)
        return x
    


class Pooler(nn.Module):
    """
    Pooling the input
    N, C, H, W --> N, C, H/2, W/2 via convolution(stride=2) or maxpool(2) 
    """
    
    pool_types = ['MaxPool', 'StrideConv']
    
    def __init__(self, n_channels:int, pool_type:str='StrideConv', ndim:int=2):
        """
        Args:
            n_channels: number of channels for the working input
            pool_types: pooling method {'MaxPool' or 'StrideConv'}
        """
        super().__init__()

        if pool_type not in self.pool_types:
            raise ValueError(f"{pool_type} is not supported pooling method, only {self.pool_types} available")
        
        if pool_type == 'StrideConv':
            self.pooler = nn.Conv3d(n_channels, n_channels, 2, stride=2) if ndim == 3 else nn.Conv2d(n_channels, n_channels, 2, stride=2)
        else:
            self.pooler = nn.MaxPool3d(2) if ndim == 3 else nn.MaxPool2d(2)
        
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        """forward the input tensor x of shape (N, C, (Depth, )H, W) through the Pooler"""
        return self.pooler(x)
    


class Upsample(nn.Module):
    """
    Upsample the input and change the number of channels
    N, C_in, H, W --> N, C_out, 2*H, 2*W via convolution 
    """

    def __init__(self, scale_factor:int, nch_in:int=None, nch_out:int=None, 
                 mode:str='bilinear', align_corners:bool=False, ndim:int=2):
        """
        Args:
            scale_factor:   the scale_factor argument as in torch.nn.functional.interpolate
            nch_in:         number of the input channels
            nch_out:        number of the output channels
            mode:           the mode argument as in torch.nn.functional.interpolate
            align_corners:  the align_corners argument as in torch.nn.functional.interpolate
            ndim:           dimensionality of the input
        """
        super().__init__()

        align_corners = align_corners if mode in ['linear', 'bilinear', 'bicubic', 'trilinear'] else None
        self.interp = lambda x : F.interpolate(x, scale_factor=scale_factor, mode=mode, align_corners=align_corners)
        
        if nch_in != nch_out:
            self.conv = nn.Conv3d(nch_in, nch_out, 1) if ndim == 3 else nn.Conv2d(nch_in, nch_out, 1)
        else:
            self.conv = None
        
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        """forward the x tensor of shape (N, C, (Depth, )H, W) through the Upsampler"""
        x = self.interp(x)
        return x if self.conv is None else self.conv(x)
    


class UNetBase(nn.Module, ABC):
    """ Unet Base Class """
    
    def __init__(self, nch_in:int, nch_out:Optional[int], depth:int, nfeatures_init:int=64, pad_convs:bool=True, 
                 norm_start_level:int=0, norm_groups:int=0, activation:Union[str, nn.Module]='ReLU', 
                 dropout_start_level:int=3, p_dropout:float=0.1, pool_mode:str='StrideConv', upsample_mode:str='bilinear', 
                 out_activation:Union[str, nn.Module]='ReLU', depthwise:bool=True):
        """
        Args:
            nch_in:                 number of input channels
            nch_out:                set None to add output convlution layer externally
            depth:                  depth of the U-net network
            nfeatures_init:         number of features after first convolution
            pad_convs:              whether to use padded convolutions (padding = 1 if pad_convs)
            norm_start_level:       the level where normalizations starts
            norm_groups:            the number of groups for GroupNorm, 0 for BatchNorm, <0 for no normalization
            activation:             str or nn.Module, the activation
            dropout_start_level:    the level where the dropout starts
            p_dropout:              the probability for dropout
            pool_mode:              str, {'MaxPool', 'StrideConv'} the pooling mode
            upsample_mode:          str, the upsample mode for F.interpolate
            out_activation:         str or nn.Module, activation following the output convolution if nch_out is not None
        NOTE:
            if nch_out is set None, the output will be of shape (N, nfeatures_init, (Depth, )H, W), and an external layer is expeted
            to transform the output to desired layers of features    
        """
        super().__init__()
        
        self.nch_in = nch_in
        self.nch_out = nch_out
        self.depth = depth
        
        # HyperParameters
        self.padding = 1 if pad_convs else 0
        self.norm_start_level = norm_start_level
        self.norm_groups = norm_groups
        self.activation = get_activation(activation)
        self.dropout_start_level = dropout_start_level
        self.p_dropout = p_dropout
        self.pool_mode = pool_mode
        self.upsample_mode = upsample_mode
        self.depthwise = bool(depthwise)
        
        # Modules for encoder and pooling
        self.encoder = nn.ModuleList()
        self.pooler = nn.ModuleList()
        for level in range(self.depth):
            _in_channels = self.nch_in if level == 0 else nfeatures_init * 2 ** (level - 1)
            _out_channels = nfeatures_init * 2 ** level
            self.encoder.append(self._conv_block(level, _in_channels, _out_channels))
            self.pooler.append(self._pooler(_out_channels))

        # Module for base
        _in_channels = nfeatures_init * 2 ** (self.depth - 1)
        _out_channels = nfeatures_init * 2 ** self.depth
        self.base = self._conv_block(self.depth, _in_channels, _out_channels)

        # Module for decoder and upsampling
        self.upsampler = nn.ModuleList()
        self.decoder = nn.ModuleList()
        for level in range(self.depth, 0, -1):
            _in_channels = nfeatures_init * 2 ** level
            _out_channels = nfeatures_init * 2 ** (level - 1)
            self.upsampler.append(self._upsampler(_in_channels, _out_channels))
            self.decoder.append(self._conv_block(level, _in_channels, _out_channels))
            
        # Output
        self.out_conv = None if self.nch_out is None else self._out_conv(nfeatures_init, self.nch_out)
        self.out_activation = get_activation(out_activation)

    @staticmethod
    def _crop_tensor(input_:torch.Tensor, shape_to_crop:torch.Size) -> torch.Tensor:
        """
        crops the input tensor according to the given shape
        Args:
            input_:         (N, C, (Depth, )H, W) the input tensor
            shape_to_crop:  (4,) or (5,): len(shap_to_crop) should equals to len(input_.shape)
        Returns:
            torch.Tensor:   the cropped tensor
        """
        input_shape = input_.shape
        shape_diff = tuple((ish - csh) // 2 for ish, csh in zip(input_shape, shape_to_crop))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=TracerWarning)
            if all([sd == 0 for sd in shape_diff]):
                return input_
        crop = tuple(slice(sd, sh - sd) for sd, sh in zip(shape_diff, input_shape))
        return input_[crop]
    
    def _crop_and_concat(self, from_decoder:torch.Tensor, from_encoder:torch.Tensor) -> torch.Tensor:
        """crop the 'from_encoder' tensor and concatenate along the channel-dim both
        Args:
            from_decoder:   (N, C1, (Depth1, )H1, W1) tensor from decoder
            from_encoder:   (N, C2, (Depth2, )H2, W2) tensor from encoder, usually H2=H1 and W2=W1
        Returns:
            torch.Tensor:   (N, 2*C2, (Depth1, )H1, W1) the concatenated tensor
        """
        cropped = self._crop_tensor(from_encoder, from_decoder.shape)
        return torch.cat((cropped, from_decoder), dim=1)
    
    def forward(self, input:torch.Tensor) -> torch.Tensor:
        """ forward the input tensor through all the neural nodes """
        x = input
        encoder_out = []
        for level in range(self.depth):
            x = self.encoder[level](x)
            encoder_out.append(x)
            x = self.pooler[level](x)
        
        x = self.base(x)

        encoder_out = encoder_out[::-1]
        for level in range(self.depth):
            x = self.upsampler[level](x)
            x = self.decoder[level](self._crop_and_concat(x, encoder_out[level]))

        if self.out_conv is not None:
            x = self.out_conv(x)
            if self.out_activation is not None:
                x = self.out_activation(x)
        return x
    
    def forward_(self, input:torch.Tensor) -> torch.Tensor:
        """ alias of the forward method but to test the shape flow """
        x = input
        print(f"input.shape = {list(input.shape)}")
        encoder_out = []
        for level in range(self.depth):
            _shape_0 = x.shape
            x = self.encoder[level](x)
            _shape_1 = x.shape
            encoder_out.append(x)
            x = self.pooler[level](x)
            _shape_2 = x.shape
            print(f"level_{level}: {list(_shape_0)} -encoder- {list(_shape_1)} -pooler- {list(_shape_2)}")
        
        _shape_0 = x.shape
        x = self.base(x)
        _shape_1 = x.shape
        print(f"level_{self.depth}: {list(_shape_0)} -base- {list(_shape_1)}")

        encoder_out = encoder_out[::-1]
        for level in range(self.depth):
            _shape_0 = x.shape
            x = self.upsampler[level](x)
            _shape_1 = x.shape
            _shape_2 = self._crop_and_concat(x, encoder_out[level]).shape
            x = self.decoder[level](self._crop_and_concat(x, encoder_out[level]))
            _shape_3 = x.shape
            print(f"level_{level}: {list(_shape_0)} -upsampler- {list(_shape_1)} -concat- {list(_shape_2)} -decoder- {list(_shape_3)}")

        if self.out_conv is not None:
            _shape_0 = x.shape
            x = self.out_conv(x)
            _shape_1 = x.shape
            print(f"out: {list(_shape_0)} -out- {list(_shape_1)}")
            if self.out_activation is not None:
                x = self.out_activation(x)
        return x
    
    @abstractmethod
    def _conv_block(self, level:int, in_channels:int, out_channels:int) -> nn.Module:
        raise NotImplementedError
    
    @abstractmethod
    def _upsampler(self, in_channels:int, out_channels:int) -> Upsample:
        raise NotImplementedError
    
    @abstractmethod
    def _pooler(self, n_channels:int) -> nn.Module:
        raise NotImplementedError
        
    @abstractmethod
    def _out_conv(self, in_channels:int, out_channels:int) -> nn.Module:
        raise NotImplementedError
    


class UNet2d(UNetBase):
    """2d UNet for N x C x H x W"""

    def __init__(self, nch_in:int, nch_out:Optional[int], depth:int, nfeatures_init:int=64, pad_convs:bool=True, 
                 norm_start_level:int=0, norm_groups:int=0, activation:Union[str, nn.Module]='ReLU', 
                 dropout_start_level:int=None, p_dropout:float=0.1, pool_mode:str='StrideConv', upsample_mode:str='bilinear',
                 out_activation:Union[str, nn.Module]='ReLU', depthwise:bool=True):
        
        super().__init__(nch_in=nch_in, nch_out=nch_out, depth=depth, nfeatures_init=nfeatures_init, pad_convs=pad_convs, 
                         norm_start_level=norm_start_level, norm_groups=norm_groups, activation=activation, 
                         dropout_start_level=dropout_start_level, p_dropout=p_dropout, pool_mode=pool_mode, upsample_mode=upsample_mode,
                         out_activation=out_activation, depthwise=depthwise)

    def _conv_block(self, level:int, in_channels:int, out_channels:int) -> nn.Module:
       return ConvBlock(level=level, in_channels=in_channels, out_channels=out_channels, padding=self.padding, 
                        norm_start_level=self.norm_start_level, norm_groups=self.norm_groups, activation=self.activation, 
                        dropout_start_level=self.dropout_start_level, p_dropout=self.p_dropout,
                        ndim=2, depthwise=self.depthwise)
    
    def _upsampler(self, in_channels:int, out_channels:int) -> Upsample:
        return Upsample(scale_factor=2, nch_in=in_channels, nch_out=out_channels, 
                        mode=self.upsample_mode, align_corners=False, ndim=2)
    
    def _pooler(self, n_channels:int) -> nn.Module:
        return Pooler(n_channels=n_channels, pool_type=self.pool_mode, ndim=2)
        
    def _out_conv(self, in_channels:int, out_channels:int) -> nn.Module:
        return nn.Conv2d(in_channels, out_channels, 1)
