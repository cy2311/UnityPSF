import torch


class FrameRescalar:
    """Rescales the frames to (0~1)"""

    def __init__(self, scale:float=1.0, offset:float=0.0):
        self.scale = scale if scale is not None else 1.0
        self.offset = offset if offset is not None else 0.0

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        if x is None:
            return None
        return (x - self.offset) / self.scale
    


class BkgFrameRescalar(FrameRescalar):
    """Rescales the background frames by normalizing tha maximum to 1"""

    def __init__(self, scale:float=1.0):
        super().__init__(scale=scale, offset=0.0)
 


class TargetProcess:
    def __init__(self, disable_attr:list=None, phot_max:float=None, z_max:float=None):
        """
        Target corresponding to the Gausian-Mixture Model Loss. 
        Args:
            disable_attr:       index of the attribute / channels one wants to disable (phot, x, y, z).
            phot_max:           maximum photon to normalize the phot channel
            z_max:              maximum z to normalize the z channel
        """
        self.attr_ix = disable_attr if disable_attr is None or isinstance(disable_attr, (list, tuple)) else [disable_attr]
        self.phot_max = phot_max
        self.z_max = z_max              

    def forward(self, pxyz_tar:torch.Tensor) -> torch.Tensor:
        """
        generate and normalize the targets from an localization set
            pxyz_tar:           (nfrms, n_max, 4) or (n_max, 4) the pxyz target to be processed
        Returns:
            torch.Tensor:       (nfrms, n_max, 4) or (n_max, 4) the pxyz target
        """
        # disable desired attributes
        if self.attr_ix is not None:
            pxyz_tar[..., self.attr_ix] = 0.0

        # rescale the attributes to (0 ~ 1) for GMMLoss calculation
        if self.phot_max is not None:
            pxyz_tar[..., 0] /= self.phot_max
        if self.z_max is not None:
            pxyz_tar[..., 3] /= self.z_max
        
        return pxyz_tar