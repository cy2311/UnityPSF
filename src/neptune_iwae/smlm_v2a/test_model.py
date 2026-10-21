import sys
import pathlib
import time
import torch

from .models.unet import UNet2d
from .models.model_factory import MultiHeads
from .models.smlm_model import SingleUNet, DoubleUNet

from .utils.io import param_io


def test_unet():
    input = torch.rand(size=(64, 1, 40, 40), dtype=torch.float32)

    unet2d = UNet2d(nch_in=1, nch_out=10, depth=2, nfeatures_init=48, pad_convs=True,
                    norm_start_level=0, norm_groups=0, activation='ReLU', dropout_start_level=3, p_dropout=-0.1,
                    out_conv=False, out_activation=None, pool_mode='StrideConv', upsample_mode='bilinear')
    
    print(unet2d.encoder)
    print(unet2d.base)
    print(unet2d.decoder)
    o = unet2d.forward_(input)
    print(o.shape)


def test_heads():
    x = torch.rand(size=(64, 64, 64, 64))
    mt_heads = MultiHeads(in_channels=64, out_channels=10, last_kernelsz=1, pad_convs=True, norm_groups=-1)
    print(mt_heads)
    o = mt_heads.forward_(x)
    print(o.shape)


def test_model():
    fpath = pathlib.Path('D:/Data/simulations/20250911_test/fullview_settings')
    param, calibration = param_io.load_params(fpath / pathlib.Path('param_test.yaml'))
    device = 'cuda:0'
    
    single_unet = SingleUNet.parse(param).to(device)
    double_unet = DoubleUNet.parse(param).to(device)
    
    x = torch.rand(size=(64, 3, 64, 64), dtype=torch.float32).to(device)
    
    tas = time.time()
    o0 = single_unet(x)
    print(f"single_unet: {time.time()-tas:.2f} on device {device}")
    o1 = double_unet(x)
    print(f"double_unet: {time.time()-tas:.2f} on device {device}")



if __name__ == '__main__':
    
    if sys.argv[1] == 'unet':
        test_unet()
    elif sys.argv[1] == 'mt_heads':
        test_heads()
    elif sys.argv[1] == 'smlm_model':
        test_model()
    