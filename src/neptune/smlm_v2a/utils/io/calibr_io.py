from typing import Union

from pathlib import Path
import pickle

from ..param_types import RecursiveNamespace, load_template, autofill_dict

    

class CalibrHandling:

    def __init__(self):
        self.cam_path = None
        self.csplinePSF_path = None
        self.chnlT_path = None


    def load_calibrations(self, cam_path:Union[Path, str], csplinePSF_path:Union[Path, str], chnlT_path:Union[Path, str]=None) -> RecursiveNamespace:
        """
        load calibrations from filenames
        Args:
            cam_path:           the .pkl filename of the camera calibration
            csplinePSF_path:    the .pkl filename of the CSplinePSF calibration
            chnlT_path:         the .pkl filename of the channel_transform calibration
        """
        self.cam_path = cam_path if isinstance(cam_path, Path) else Path(cam_path)
        self.csplinePSF_path = csplinePSF_path if isinstance(csplinePSF_path, Path) else Path(csplinePSF_path)
        self.chnlT_path = chnlT_path if isinstance(chnlT_path, Path) else Path(chnlT_path) if chnlT_path is not None else None

        with open(self.cam_path, 'rb') as fid:
            camera = pickle.load(fid)

        with open(self.csplinePSF_path, 'rb') as fid:
            csplinePSF = pickle.load(fid)
        
        if self.chnlT_path is not None:
            with open(self.chnlT_path, 'rb') as fid:
                chnlT = pickle.load(fid)
        else:
            chnlT = {'R2Ts':None, 'Ts2R':None}

        nchannels = self._assert_nchannels(camera, csplinePSF, chnlT)
        ndim = 2 if csplinePSF['modality'] == '2Dfrm' else 3
        if ndim == 3:
            camera['cam_pxszs'].append(csplinePSF['zstepsz_nm'])

        calibrations_dict = {'ndim':ndim, 'nchannels':nchannels, 'Camera':camera, 'CSplinePSF':csplinePSF, 'ChannelTrans':chnlT}
        calibrations_template = load_template('calibr_template')
        calibrations_dict = autofill_dict(calibrations_dict, calibrations_template)
        calibrations = RecursiveNamespace(**calibrations_dict)
        return calibrations
    

    @staticmethod
    def write_calibrations(filename:Union[str, Path], calibrations:Union[dict, RecursiveNamespace]):
        """write calibrations to filename"""
        filename = Path(filename) if not isinstance(filename, Path) else filename
        try:
            filename.parents[0].mkdir(parents=False, exist_ok=True)
        except FileNotFoundError:
            raise FileNotFoundError(f"cannot find {filename.parents[1]} to create folder {filename.parents[0]}")
        
        if isinstance(calibrations, RecursiveNamespace):
            calibrations = calibrations.to_dict()
        with open(filename, 'wb') as fid:
            pickle.dump(calibrations, fid, pickle.HIGHEST_PROTOCOL)


    @staticmethod
    def _assert_nchannels(cam:dict, csplinePSF:dict, chnlT:dict) -> int:
        """
        sanity check if the calibration files are cross-consistant in terms of the nchannels
        Args:
            cam:        the camera calibration dictionary
            csplinePSF: the csplinePSF calibration dictionary
            chnlT:      the channel transfer calibration dictionary
        Returns:
            int:        number of channels after assertation
        """
        if cam['view_type'] not in {'fullview', 'dualview', 'quadview'}:
            raise ValueError(f"{cam['view_type']} is unsupported, supported are 'fullview', 'dualview', and 'quadview'.")
        
        nchannels = 1 if cam['view_type'] == 'fullview' else 2 if cam['view_type'] == 'dualview' else 4

        if csplinePSF['nchannels'] != nchannels:
            raise ValueError(f"nchannel mismatch: cam['view_type]={cam['view_type']}, csplinePSF['nchannels']={csplinePSF['nchannels']}")
        
        if nchannels > 1:
            if chnlT['R2Ts'] is None or chnlT['Ts2R'] is None:
                raise ValueError(f"nchannel mismatch: cam['view_type]={cam['view_type']}, at least one of chnlT's items is None")
            
        return nchannels


    
def load_calibrations(cam_path:Union[Path, str], csplinePSF_path:Union[Path, str], chnlT_path:Union[Path, str]=None) -> RecursiveNamespace: # alias
    return CalibrHandling().load_calibrations(cam_path, csplinePSF_path, chnlT_path)


def save_calibrations(filename:Union[str, Path], calibrations:Union[dict, RecursiveNamespace]): # alias
    CalibrHandling().write_calibrations(filename, calibrations)