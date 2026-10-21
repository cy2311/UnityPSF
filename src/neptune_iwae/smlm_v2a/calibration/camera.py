from abc import ABC, abstractmethod
from typing import Union, Tuple

import math
import torch


class Camera(ABC):
    """Abstract camera module"""

    def __init__(self, chipszx:int, chipszy:int, QE:float, spur_noise:float, 
                 offset:Union[float, torch.Tensor], A2D:Union[float, torch.Tensor], var:Union[float, torch.Tensor], 
                 EMGain:float=1.0, scmos_var:bool=False):
        """
        Args:
            chipszx:        chip width of the camera
            chipszy:        chip height of the camera
            QE:             quantum efficiency (0 ~ 1)
            spur_noise:     spurious noise
            offset:         (chipszy, chipszx) or float, manufacturer offset
            var:            (chipszy, chipszx) or float, manufacturer read variance
            A2D:            (chipszy, chipszx) or float, manufacturer analog-to-digital convention
            EMGain:         the EMGain applied to an EMCCD
            scmos_var:      whether to treat the camera as a low QE sCMOS camera
        """
        self.chipszx = chipszx
        self.chipszy = chipszy
        self.QE = QE
        self.spur_noise = spur_noise
        
        self.scmos_var = scmos_var
        if self.scmos_var:
            assert offset.shape == (self.chipszy, self.chipszx), f"offset should be of size ({self.chipszy}, {self.chipszx}) when scmos_var is True"
            assert var.shape == (self.chipszy, self.chipszx), f"var should be of size ({self.chipszy}, {self.chipszx}) when scmos_var is True"
            assert A2D.shape == (self.chipszy, self.chipszx), f"A2D should be of size ({self.chipszy}, {self.chipszx}) when scmos_var is True"
            self.offset = offset
            self.var = var
            self.A2D = A2D
            self.EMGain = 1.0
        else:
            self.offset = offset.mean().item() if isinstance(offset, torch.Tensor) else offset
            self.var = var.mean().item() if isinstance(var, torch.Tensor) else var
            self.A2D = A2D.mean().item() if isinstance(A2D, torch.Tensor) else A2D
            self.EMGain = EMGain    

    def shot_noise_impl(self, x:torch.Tensor) -> torch.Tensor:
        """implement the shot noise to the input frame"""
        return torch.distributions.poisson.Poisson(self.QE * x + self.spur_noise, validate_args=False).sample()

    def gamma_noise_impl(self, x:torch.Tensor) -> torch.Tensor:
        """implement the gamma noise when EMGain is applied in an EMCCD"""
        if self.EMGain <= 1.0:
            return x.clone()
        return torch.distributions.gamma.Gamma(x, 1.0 / self.EMGain, validate_args=False).sample()
        
    def read_noise_impl(self, x:torch.Tensor, xextent:Tuple[int, int]=None, yextent:Tuple[int, int]=None) -> torch.Tensor:
        """implement the read_out noise of the camera"""
        if isinstance(self.var, torch.Tensor):
            assert (xextent is not None) and (yextent is not None), "extents(roi) is needed for SCMOS Camera"
            _var = self.var[yextent[0] : yextent[1], xextent[0] : xextent[1]].to(x.device)
            return x + torch.sqrt(_var) * torch.randn_like(x)
        else:
            return x + math.sqrt(float(self.var)) * torch.randn_like(x)
        
    def convention(self, x:torch.Tensor, xextent:Tuple[int, int]=None, yextent:Tuple[int, int]=None) -> torch.Tensor:
        """convert from the analog signal (electrons) to pixel digitals"""
        if isinstance(self.A2D, torch.Tensor):
            assert (xextent is not None) and (yextent is not None), "extents(roi) is needed for SCMOS Camera"
            x = x * self.A2D[yextent[0] : yextent[1], xextent[0] : xextent[1]].to(x.device)
        else:
            x = x * self.A2D
        return x.floor()

    @staticmethod
    def _santiy_check(xextent:Tuple[int, int], yextent:Tuple[int, int], frame:torch.Tensor):
        """
        checks if the input phot_frame corresponds to the given roi / extents
        Args:
            xextent:    x-axis extent of the photon frame w.r.t the camera chip
            yextent:    y-axis extent of the photon frame w.r.t the camera chip 
            frame:      (N, H, W) the photon frame
        """
        if xextent[1]-xextent[0] != frame.shape[-1]:
            raise ValueError(f"shape mismatch: xextent[1]-xextent[0]={xextent[1]-xextent[0]}, phot_frame.shape[-1]={frame.shape[-1]}")
        if yextent[1]-yextent[0] != frame.shape[-2]:
            raise ValueError(f"shape mismatch: yextent[1]-yextent[0]={yextent[1]-yextent[0]}, phot_frame.shape[-2]={frame.shape[-2]}")

    @abstractmethod
    def forward(self, xextent:Tuple[int, int], yextent:Tuple[int, int], phot_frame:torch.Tensor) -> torch.Tensor:
        """Convert photon frame to camera frame"""
        raise NotImplementedError
    
    @abstractmethod
    def backward(self, xextent:Tuple[int, int], yextent:Tuple[int, int], cam_frame:torch.Tensor) -> torch.Tensor:
        """Approximate the photon frame from the camera frame"""
        raise NotImplementedError
    


class PerfectCamera(Camera):
    """A perfect camera that only exhibits shot noise"""

    def __init__(self, chipszx:int, chipszy:int):
        """
        Args:
            chipszx:        chip width of the camera
            chipszy:        chip height of the camera
        """
        super().__init__(chipszx=chipszx, chipszy=chipszy, QE=1.0, spur_noise=0.0, 
                         offset=0.0, A2D=1.0, var=0.0, EMGain=1.0, scmos_var=False)
        
    def forward(self, xextent:Tuple[int, int], yextent:Tuple[int, int], phot_frame:torch.Tensor) -> torch.Tensor:
        """
        checks if the input phot_frame corresponds to the given roi / extents
        Args:
            xextent:    x-axis extent of the photon frame w.r.t the camera chip
            yextent:    y-axis extent of the photon frame w.r.t the camera chip 
            phot_frame: (N, H, W) the photon frame
        Returns:
            torch.Tensor:   (N, H, W) the camera frame sampled from the phot_frame
        """
        
        self._santiy_check(xextent=xextent, yextent=yextent, frame=phot_frame)
        cam_frame = self.shot_noise_impl(phot_frame)
        cam_frame = self.convention(cam_frame)
        return cam_frame.clamp(min=0, max=65535)
    
    def backward(self, xextent:Tuple[int, int], yextent:Tuple[int, int], cam_frame:torch.Tensor) -> torch.Tensor:
        """
        checks if the input phot_frame corresponds to the given roi / extents
        Args:
            xextent:    x-axis extent of the photon frame w.r.t the camera chip
            yextent:    y-axis extent of the photon frame w.r.t the camera chip 
            cam_frame:  (N, H, W) the camera frame
        Returns:
            torch.Tensor:   (N, H, W) the photon frame predicted from the camera frame
        """
        self._santiy_check(xextent=xextent, yextent=yextent, frame=cam_frame)        
        return cam_frame.clone()
    
    @classmethod
    def parse(cls, calibration):
        return cls(chipszx=calibration.Camera.chipszx, chipszy=calibration.Camera.chipszy)
        


class EMCCD(Camera):
    """An EMCCD camera"""

    def __init__(self, chipszx:int, chipszy:int, QE:float, spur_noise:float, 
                 offset:float, A2D:float, var:float, EMGain:float):
        """
        Args:
            chipszx:        chip width of the camera
            chipszy:        chip height of the camera
            QE:             quantum efficiency (0 ~ 1)
            spur_noise:     spurious noise
            offset:         manufacturer offset
            var:            manufacturer read variance
            A2D:            manufacturer analog-to-digital convention
            EMGain:         the EMGain applied to an EMCCD
        """
        super().__init__(chipszx=chipszx, chipszy=chipszy, QE=QE, spur_noise=spur_noise, 
                         offset=offset, A2D=A2D, var=var, EMGain=EMGain, scmos_var=False)
        
    def forward(self, xextent:Tuple[int, int], yextent:Tuple[int, int], phot_frame:torch.Tensor) -> torch.Tensor:
        """
        checks if the input phot_frame corresponds to the given roi / extents
        Args:
            xextent:    x-axis extent of the photon frame w.r.t the camera chip
            yextent:    y-axis extent of the photon frame w.r.t the camera chip 
            phot_frame: (N, H, W) the photon frame
        Returns:
            torch.Tensor:   (N, H, W) the camera frame sampled from the phot_frame
        """
        
        self._santiy_check(xextent=xextent, yextent=yextent, frame=phot_frame)
        cam_frame = self.shot_noise_impl(phot_frame)
        cam_frame = self.gamma_noise_impl(cam_frame)
        cam_frame = self.read_noise_impl(cam_frame)
        cam_frame = self.convention(cam_frame)
        cam_frame += self.offset
        return cam_frame.clamp(min=0, max=65535)
    
    def backward(self, xextent:Tuple[int, int], yextent:Tuple[int, int], cam_frame:torch.Tensor) -> torch.Tensor:
        """
        checks if the input phot_frame corresponds to the given roi / extents
        Args:
            xextent:    x-axis extent of the photon frame w.r.t the camera chip
            yextent:    y-axis extent of the photon frame w.r.t the camera chip 
            cam_frame:  (N, H, W) the camera frame
        Returns:
            torch.Tensor:   (N, H, W) the photon frame predicted from the camera frame
        """
        self._santiy_check(xextent=xextent, yextent=yextent, frame=cam_frame) 
        phot_frame = (cam_frame - self.offset) / self.A2D
        phot_frame /= self.EMGain
        phot_frame -= self.spur_noise
        phot_frame /= self.QE       
        return phot_frame
    
    @classmethod
    def parse(cls, calibration):
        return cls(chipszx = calibration.Camera.chipszx,
                   chipszy = calibration.Camera.chipszy,
                   QE = calibration.Camera.QE,
                   spur_noise = calibration.Camera.spur_noise,
                   offset = calibration.Camera.offset,
                   A2D = calibration.Camera.A2D,
                   var = calibration.Camera.var,
                   EMGain = calibration.Camera.EMGain)
    


class sCMOS95(Camera):
    """An high QE sCMOS (e.g. Prime95B) and neglectable pixel read-out noise"""

    def __init__(self, chipszx:int, chipszy:int, QE:float, spur_noise:float, 
                 offset:float, A2D:float, var:float):
        """
        Args:
            chipszx:        chip width of the camera
            chipszy:        chip height of the camera
            QE:             quantum efficiency (0 ~ 1)
            spur_noise:     spurious noise
            offset:         manufacturer offset
            var:            manufacturer read variance
            A2D:            manufacturer analog-to-digital convention
        """
        super().__init__(chipszx=chipszx, chipszy=chipszy, QE=QE, spur_noise=spur_noise, 
                         offset=offset, A2D=A2D, var=var, EMGain=1.0, scmos_var=False)
        
    def forward(self, xextent:Tuple[int, int], yextent:Tuple[int, int], phot_frame:torch.Tensor) -> torch.Tensor:
        """
        checks if the input phot_frame corresponds to the given roi / extents
        Args:
            xextent:    x-axis extent of the photon frame w.r.t the camera chip
            yextent:    y-axis extent of the photon frame w.r.t the camera chip 
            phot_frame: (N, H, W) the photon frame
        Returns:
            torch.Tensor:   (N, H, W) the camera frame sampled from the phot_frame
        """
        
        self._santiy_check(xextent=xextent, yextent=yextent, frame=phot_frame)
        cam_frame = self.shot_noise_impl(phot_frame)
        cam_frame = self.read_noise_impl(cam_frame)
        cam_frame = self.convention(cam_frame)
        cam_frame += self.offset
        return cam_frame.clamp(min=0, max=65535)
    
    def backward(self, xextent:Tuple[int, int], yextent:Tuple[int, int], cam_frame:torch.Tensor) -> torch.Tensor:
        """
        checks if the input phot_frame corresponds to the given roi / extents
        Args:
            xextent:    x-axis extent of the photon frame w.r.t the camera chip
            yextent:    y-axis extent of the photon frame w.r.t the camera chip 
            cam_frame:  (N, H, W) the camera frame
        Returns:
            torch.Tensor:   (N, H, W) the photon frame predicted from the camera frame
        """
        self._santiy_check(xextent=xextent, yextent=yextent, frame=cam_frame) 
        phot_frame = (cam_frame - self.offset) / self.A2D
        phot_frame -= self.spur_noise
        phot_frame /= self.QE       
        return phot_frame
    
    @classmethod
    def parse(cls, calibration):
        return cls(chipszx = calibration.Camera.chipszx,
                   chipszy = calibration.Camera.chipszy,
                   QE = calibration.Camera.QE,
                   spur_noise = calibration.Camera.spur_noise,
                   offset = calibration.Camera.offset,
                   A2D = calibration.Camera.A2D,
                   var = calibration.Camera.var)
    


class sCMOS82(Camera):
    """Low QE sCMOS where offset, var, and A2D are pixel-dependent"""

    def __init__(self, chipszx:int, chipszy:int, QE:float, spur_noise:float, 
                 offset:torch.Tensor, A2D:torch.Tensor, var:torch.Tensor):
        """
        Args:
            chipszx:        chip width of the camera
            chipszy:        chip height of the camera
            QE:             quantum efficiency (0 ~ 1)
            spur_noise:     spurious noise
            offset:         (chipszy, chipszx) manufacturer offset
            var:            (chipszy, chipszx) manufacturer read variance
            A2D:            (chipszy, chipszx) manufacturer analog-to-digital convention
        """
        super().__init__(chipszx=chipszx, chipszy=chipszy, QE=QE, spur_noise=spur_noise, 
                         offset=offset, A2D=A2D, var=var, EMGain=1.0, scmos_var=True)

    def forward(self, xextent:Tuple[int, int], yextent:Tuple[int, int], phot_frame:torch.Tensor) -> torch.Tensor:
        """
        checks if the input phot_frame corresponds to the given roi / extents
        Args:
            xextent:    x-axis extent of the photon frame w.r.t the camera chip
            yextent:    y-axis extent of the photon frame w.r.t the camera chip 
            phot_frame: (N, H, W) the photon frame
        Returns:
            torch.Tensor:   (N, H, W) the camera frame sampled from the phot_frame
        """
        
        self._santiy_check(xextent=xextent, yextent=yextent, frame=phot_frame)
        cam_frame = self.shot_noise_impl(phot_frame)
        cam_frame = self.read_noise_impl(cam_frame, xextent=xextent, yextent=yextent)
        cam_frame = self.convention(cam_frame, xextent=xextent, yextent=yextent)
        cam_frame += self.offset[yextent[0]:yextent[1], xextent[0]:xextent[1]].to(phot_frame.device)
        return cam_frame.clamp(min=0, max=65535)
    
    def backward(self, xextent:Tuple[int, int], yextent:Tuple[int, int], cam_frame:torch.Tensor) -> torch.Tensor:
        """
        checks if the input phot_frame corresponds to the given roi / extents
        Args:
            xextent:    x-axis extent of the photon frame w.r.t the camera chip
            yextent:    y-axis extent of the photon frame w.r.t the camera chip 
            cam_frame:  (N, H, W) the camera frame
        Returns:
            torch.Tensor:   (N, H, W) the photon frame predicted from the camera frame
        """
        self._santiy_check(xextent=xextent, yextent=yextent, frame=cam_frame) 
        _offset = self.offset[yextent[0]:yextent[1], xextent[0]:xextent[1]].to(cam_frame.device)
        _A2D = self.A2D[yextent[0]:yextent[1], xextent[0]:xextent[1]].to(cam_frame.device)
        phot_frame = (cam_frame - _offset) / _A2D
        phot_frame -= self.spur_noise
        phot_frame /= self.QE       
        return phot_frame
    
    @classmethod
    def parse(cls, calibration):
        return cls(chipszx = calibration.Camera.chipszx,
                   chipszy = calibration.Camera.chipszy,
                   QE = calibration.Camera.QE,
                   spur_noise = calibration.Camera.spur_noise,
                   offset = calibration.Camera.offset,
                   A2D = calibration.Camera.A2D,
                   var = calibration.Camera.var)
