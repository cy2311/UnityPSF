from typing import Union, Tuple
from copy import deepcopy

import math
from scipy.interpolate import LinearNDInterpolator
import torch

from ..utils.helpers import to_tensor

class ChannelTrans:
    
    """ NOTE: The reference channel is always the 0-th channel """
    
    def __init__(self, nchannels:int, R2Ts:torch.Tensor=None, Ts2R:torch.Tensor=None):
        """
        Args:
            nchannels:  number of channels
            R2Ts:       (nchannels-1, warpdeg**2, 2), polynomial transformation coefficients from the reference (0-th channel) to the target channels
            Ts2R:       (nchannels-1, warpdeg**2, 2), polynomial transformation coefficients from the target channels to the reference channel (0-th channel)
        """
        assert nchannels > 1, "only valid for multichannels (nchannels > 1)"
        assert (R2Ts is None and Ts2R is None) or (R2Ts is not None and Ts2R is not None), "R2Ts and Ts2R must all be None or not None"
        self.nchannels = nchannels
        if R2Ts is not None:
            assert R2Ts.shape == Ts2R.shape, f"R2Ts and Ts2R should be of the same size, but R2Ts.shape={R2Ts.shape}, Ts2R.shape={Ts2R.shape}"
            assert R2Ts.shape[0] == self.nchannels - 1, f"shape mismatch: R2Ts must be of shape ((nchannels-1, warpdeg**2, 2)), but R2Ts.shape[0]={R2Ts.shape[0]} while nchannels={self.nchannels}"
            assert self._is_square(R2Ts.shape[1]), f"shape mismatch: R2Ts must be of shape ((nchannels-1, warpdeg**2, 2)), but R2Ts.size(1)={R2Ts.shape[1]} is not perfect square"
            assert R2Ts.shape[2] == 2, f"shape mismatch: R2Ts must be of shape ((nchannels-1, warpdeg**2, 2)), but R2Ts.size(2)={R2Ts.shape[2]}"
            self.coeff_R2Ts = to_tensor(R2Ts, dtype=torch.float32, device='cpu')
            self.coeff_Ts2R = to_tensor(Ts2R, dtype=torch.float32, device='cpu')
            self.warpdeg = math.isqrt(self.coeff_R2Ts.shape[1])
            self._identical = False
        else:
            self.coeff_R2Ts = torch.zeros((self.nchannels-1, 4, 2), dtype=torch.float32, device='cpu')
            for i in range(self.nchannels - 1):
                self.coeff_R2Ts[i, 1, 0] = 1.0
                self.coeff_R2Ts[i, 2, 1] = 1.0
            self.coeff_Ts2R = self.coeff_R2Ts.detach().clone()
            self.warpdeg = 2
            self._identical = True


    def coor_polytrans(self, xyz_src:torch.Tensor, chID:int=1, mode='R2T', offset_src:torch.Tensor=None, offset_dst:torch.Tensor=None) -> torch.Tensor:
        """
        Polynomial transformation of the coordinates from the source to the destination channel
        Args:
            xyz_scr:        (ncoors, {2, 3}) FloatTensor, [[x_scr, y_scr(, z_scr)],...] coordinates in source channel
            chID:           int, the target channel index (mode=='R2T') or the source channel index ('mode=='T2R')
            mode:           str {'R2T', 'T2R'}, 
                            'R2T' for transform from the reference (0-th) to the chID-th target channel
                            'T2R' for transform from the chID-th target to the reference (0-th) channel  
            offset_scr:     (ncoors, 2) or (2,) FloatTensor, [[offsetx, offsety],...] the offsets of the coordinates in the source channel
                            so that xyz_ = xyz_src + offset_src adopts the coordinates originated in the left upper corner of the 
                            camera-chip on which the transform coefficients were calibrated 
            offset_dst:     (ncoors, 2) or (2,) FloatTensor, the offsets of the coordinates in the destination channel   
        Returns:
            xyz_dst:        (ncoors, {2, 3}) FloatTensor, [[x_dst, y_dst(, z_dst)],...] coordinates in the destination channel transformed from the source channel
        """
        assert chID < self.nchannels, f"the {chID} exceeds the number of channels"
        assert mode in {'R2T', 'T2R'}, f"{mode} is unsupported transform mode, must be 'R2T' or 'T2R'"
        _device = xyz_src.device
        _f_type = xyz_src.dtype

        offset_src = torch.zeros((2,), dtype=_f_type, device=_device) if offset_src is None else offset_src.to(_device, dtype=_f_type)    
        offset_dst = torch.zeros((2,), dtype=_f_type, device=_device) if offset_dst is None else offset_dst.to(_device, dtype=_f_type)
        
        xyz_dst = xyz_src.detach().clone()
        xyz_dst[:, :2] += offset_src
        
        if not self._identical:    
            transB2A = self._transB2A_build(xyz_dst[:, :2])
            coeff_ = self.coeff_R2Ts[chID-1] if mode == 'R2T' else self.coeff_Ts2R[chID-1]
            xyz_dst[:, :2] = torch.matmul(transB2A, coeff_.to(device=_device, dtype=_f_type))
            
        xyz_dst[:, :2] -= offset_dst        
        return xyz_dst
    

    def coor_sig_polytrans(self, xyz_src:torch.Tensor, xyz_sig_src:torch.Tensor, chID:int=1, mode='R2T', offset_src:torch.Tensor=None) -> torch.Tensor:
        """
        Polynomial transformation of the coordinates from the source to the destination channel
        Args:
            xyz_scr:        (ncoors, {2, 3}) [[x_scr, y_scr(, z_scr)],...] coordinates in source channel
            xyz_sig_scr:    (ncoors, {2, 3}) [[x_sig_scr, y_sig_scr(, z_sig_scr)],...] coordinates sigma in source channel 
            chID:           the target channel index (mode=='R2T') or the source channel index ('mode=='T2R')
            mode:           {'R2T', 'T2R'}
                            'R2T' for transform from the reference (0-th) to the chID-th target channel
                            'T2R' for transform from the chID-th target to the reference (0-th) channel  
            offset_scr:     (ncoors, 2) or (2,) [[offsetx, offsety],...] the offsets of the coordinates in the source channel
                            so that xyz_ = xyz_src + offset_src adopts the coordinates originated in the left upper corner of the 
                            camera-chip on which the transform coefficients were calibrated 
            offset_dst:     (ncoors, 2) or (2,) the offsets of the coordinates in the destination channel   
        Returns:
            xyz_sig_dst:    (ncoors, {2, 3}) [[x_sig_dst, y_sig_dst(, z_sig_dst)],...] coordinates sigma in the destination channel transformed from the source channel
        """
        assert chID < self.nchannels, f"the {chID} exceeds the number of channels"
        assert mode in {'R2T', 'T2R'}, f"{mode} is unsupported transform mode, must be 'R2T' or 'T2R'"
        assert xyz_src.device == xyz_sig_src.device, f"xyz_src.device={xyz_src.device}, but xyz_sig_src.device={xyz_sig_src.device}"
        
        if chID == 0 or self._identical:
            return xyz_sig_src.clone()
        
        _f_type = xyz_src.dtype
        _device = xyz_src.device

        offset_src = torch.zeros((2,), dtype=_f_type, device=_device) if offset_src is None else offset_src.to(_device, dtype=_f_type)
        xy_ = xyz_src[:, :2] + offset_src
        dtransB2A_dx, dtransB2A_dy = self._transB2A_build_deriv(xy_)
        
        coeff_ = self.coeff_R2Ts[chID-1] if mode == 'R2T' else self.coeff_Ts2R[chID-1]
        dxy_dx = torch.matmul(dtransB2A_dx, coeff_.to(device=_device, dtype=_f_type))
        dxy_dy = torch.matmul(dtransB2A_dy, coeff_.to(device=_device, dtype=_f_type))

        xyz_sig_dst = xyz_sig_src.detach().clone()
        xyz_sig_dst[:, 0] = torch.sqrt(dxy_dx[:, 0]**2 * xyz_sig_src[:, 0]**2 + dxy_dy[:, 0]**2 * xyz_sig_src[:, 1]**2)
        xyz_sig_dst[:, 1] = torch.sqrt(dxy_dx[:, 1]**2 * xyz_sig_src[:, 0]**2 + dxy_dy[:, 1]**2 * xyz_sig_src[:, 1]**2)
        return xyz_sig_dst
    

    def extent_polytrans(self, xextent_src:Tuple[int, int], yextent_src:Tuple[int, int], chID:int=1, mode='R2T') -> Tuple[int, int]:
        """
        Polynomial transformation of the rectangle extent from the source to the destination channel, so that
        the rectangle extent in the destination channel is sufficient to handle all the transformations within
        the source channel
        Args:
            xextent_src:    [west, east], extents in x-axis in the source channel
            yextent_src:    [north, south], extents in y-axis in the source channel
            chID:           int, the target channel index (mode=='R2T') or the source channel index ('mode=='T2R')
            mode:           str {'R2T', 'T2R'}, 
                            'R2T' for transform from the reference (0-th) to the chID-th target channel
                            'T2R' for transform from the chID-th target to the reference (0-th) channel     
        Returns:
            xextent_dst:    [west, east], extents in x-axis in the destination channel
            yextent_dst:    [north, south], extents in y-axis in the destination channel
        """
        if self._identical:
            return deepcopy(xextent_src), deepcopy(yextent_src)
        
        x_src = to_tensor(xextent_src, dtype=torch.float32, device='cpu').repeat(2).unsqueeze(1)
        y_src = to_tensor(yextent_src, dtype=torch.float32, device='cpu').repeat_interleave(2).unsqueeze(1)
        xy_src = torch.cat((x_src, y_src), dim=1)
        
        transB2A = self._transB2A_build(xy_src)
        coeff_ = self.coeff_R2Ts[chID-1] if mode == 'R2T' else self.coeff_Ts2R[chID-1]
        xy_dst = torch.matmul(transB2A, coeff_.cpu())

        xextent_dst = xy_dst[:, 0].min().floor().int().item(), xy_dst[:, 0].max().ceil().int().item()
        yextent_dst = xy_dst[:, 1].min().floor().int().item(), xy_dst[:, 1].max().ceil().int().item()
        
        return xextent_dst, yextent_dst
    

    def im_polytrans(self, xextent_dst:Tuple[int, int], yextent_dst:Tuple[int, int], im_src:torch.Tensor, 
                     offset_src:torch.IntTensor=None, chID:int=1, mode='R2T') -> torch.Tensor:
        """
        Polynomial transformation of the image from the source to the destination channel
        Args:
            xextent_dst:    [west, east], extents in x-axis in the destination channel
            yextent_dst:    [north, south], extents in y-axis in the destination channel   
            im_scr:         (im_ysize, im_xsize) FloatTensor, image in source channel to be transformed
            offset_src:     [west, north], offset of the im_src in the source channel
            chID:           int, the target channel index (mode=='R2T') or the source channel index ('mode=='T2R')
            mode:           str {'R2T', 'T2R'}, 
                            'R2T' for transform from the reference (0-th) to the chID-th target channel
                            'T2R' for transform from the chID-th target to the reference (0-th) channel         
        Returns:
            im_dst:         (bottom-top, right-left) FloatTensor, image in destination channel transformed from the source channel
        """
        assert chID > 0 and chID < self.nchannels, f"the {chID} exceeds the number of channels or it refers back to the reference (0-th) channel"
        assert mode in {'R2T', 'T2R'}, f"{mode} is unsupported transform mode, must be 'R2T' or 'T2R'"
        if offset_src is None:
            offset_src = 0, 0

        imHeight, imWidth = im_src.shape
        if self._identical:
            if (offset_src[0] <= xextent_dst[0]) and (offset_src[0] + imWidth >= xextent_dst[1]) and \
               (offset_src[1] <= yextent_dst[0]) and (offset_src[1] + imHeight >= yextent_dst[1]):
                return im_src[yextent_dst[0]-offset_src[1] : yextent_dst[1]-offset_src[1], \
                              xextent_dst[0]-offset_src[0] : xextent_dst[1]-offset_src[0]].detach().clone()
            else:
                raise ValueError("roi mismatch: im_src in the source channel with offset_src does not match the extent in destination channel")
        
        _device = im_src.device
        _f_type = im_src.dtype
        xvec = torch.arange(*xextent_dst, dtype=_f_type, device=_device) + 0.5
        yvec = torch.arange(*yextent_dst, dtype=_f_type, device=_device) + 0.5
        xy_0 = torch.cat((xvec.repeat(yextent_dst[1]-yextent_dst[0]).view(-1, 1), yvec.repeat_interleave(xextent_dst[1]-xextent_dst[0]).view(-1, 1)), dim=1)

        # translate the xy_0 from the target channel to the source channel to determine which pixels in the source channel are going to be used
        transB2A = self._transB2A_build(xy_0)
        coeff_ = self.coeff_Ts2R[chID-1] if mode == 'R2T' else self.coeff_R2Ts[chID-1]
        xy_src = torch.matmul(transB2A, coeff_.to(device=_device, dtype=_f_type))

        # retrieve values from the source image at the pixels those match the roi in the target channel
        idx = xy_src[:, 0].floor().int() - offset_src[0]
        idy = xy_src[:, 1].floor().int() - offset_src[1]
        mask = (idx >= 0) & (idx < imWidth) & (idy >= 0) & (idy < imHeight)
        z = im_src[idy[mask], idx[mask]].detach().cpu().numpy()
        
        # translate the position of the matching pixels from the source to the destination channel
        transB2A = self._transB2A_build(torch.floor(xy_src[mask]) + 0.5)
        coeff_ = self.coeff_R2Ts[chID-1] if mode == 'R2T' else self.coeff_Ts2R[chID-1]
        xy_dst = torch.matmul(transB2A, coeff_.to(device=_device, dtype=_f_type)).detach().cpu().numpy()

        # interpolation via scipy
        interp = LinearNDInterpolator(xy_dst, z)
        Y, X = torch.meshgrid(yvec, xvec, indexing='ij')
        z_interp = interp(X.detach().cpu().numpy(), Y.detach().cpu().numpy())
        z_interp = torch.from_numpy(z_interp)
        return z_interp.to(device=_device, dtype=_f_type)

    
    def _transB2A_build(self, xy_chB:torch.Tensor) -> torch.Tensor:
        """
        build a transformation matrix transB2A for polynomial transformation from channel B to channel A, so that torch.matmul(transB2A, coeff_B2A)=[[x_chA, y_chA],...]
        where coeff_B2A is of the size(warpdeg**2, 2)
        transB2A (take deg=3 as an example): 
        each row: [ yB^0*xB^0, yB^0*xB^1, yB^0*xB^2, yB^1*xB^0, yB^1*xB^1, yB^1*xB^2, yB^2*xB^0, yB^2*xB^1, yB^2*xB^2]
        INPUT:
            xy_chB:         (ncoors, {2, 3}) FloatTensor, [[x_chB, y_chB(, z_chB)],...] coordinates in channel B that is to be transformed to channel A
        RETURN:
            transB2A:       (ncoors, self.warpdeg**2) FloatTensor, the transformation matrix as described above
        """
        exponents = torch.arange(self.warpdeg, dtype=xy_chB.dtype, device=xy_chB.device)        
        x_pow = xy_chB[:, [0]] ** exponents[None, :]
        y_pow = xy_chB[:, [1]] ** exponents[None, :]
        transB2A = y_pow[:, :, None] * x_pow[:, None, :]
        return transB2A.view(-1, self.warpdeg**2)
    

    def _transB2A_build_deriv(self, xy_chB:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        build two transformation matrix dtransB2A_dx and dtransB2A_dy for polynomial transformation from channel B to channel A, so that 
        torch.matmul(dtransB2A_dx, coeff_B2A)=[[dx_chA/dx, dy_chA/dx],...]
        torch.matmul(dtransB2A_dy, coeff_B2A)=[[dx_chA/dy, dy_chA/dy],...]
        where coeff_B2A is of the size(warpdeg**2, 2)
        INPUT:
            xy_chB:         (ncoors, {2, 3}) FloatTensor, [[x_chB, y_chB(, z_chB)],...] coordinates in channel B that is to be transformed to channel A
        RETURN:
            dtransB2A_dx:   (ncoors, self.warpdeg**2) FloatTensor, the transformation matrix as described above
            dtransB2A_dy:   (ncoors, self.warpdeg**2) FloatTensor, the transformation matrix as described above
        """
        exponents = torch.arange(self.warpdeg, dtype=xy_chB.dtype, device=xy_chB.device)        
        x_pow = xy_chB[:, [0]] ** exponents[None, :]
        x_pow_ = torch.roll(x_pow, shifts=1, dims=1)
        x_pow_ *= exponents

        y_pow = xy_chB[:, [1]] ** exponents[None, :]
        y_pow_ = torch.roll(y_pow, shifts=1, dims=1)
        y_pow_ *= exponents
         
        dtransB2A_dx = y_pow[:, :, None] * x_pow_[:, None, :]
        dtransB2A_dy = y_pow_[:, :, None] * x_pow[:, None, :]
        return dtransB2A_dx.view(-1, self.warpdeg**2), dtransB2A_dy.view(-1, self.warpdeg**2)
    

    @staticmethod
    def _is_square(i:int) -> bool:
        return i == math.isqrt(i) ** 2
    



