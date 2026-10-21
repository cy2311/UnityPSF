from __future__ import annotations
import warnings
from abc import ABC, abstractmethod
from typing import Tuple, List, Union
import copy

import math
import torch
import numpy as np

from ..utils.helpers import to_tensor, to_tensorslicer



class LocSet(ABC):
    """
    Abstract base class representing localization data with metadata.
    Localizations should be in camera pixel units and originate at [0.0, 0.0] of the camera chip where channel warping and PSF calibration were performed.
    Args (Meta):
        ndim:           int, number of dimension
        xextent:        Tuple[int, int], [west, east] extents in x-axis
        yextent:        Tuple[int, int], [north, south] extents in y-axis
        zextent:        Tuple[int, int], [up, bottom] extents in z-axis
        nchannels:      int, number of channels
        R2Ts:           torch.FloatTensor of shape (nchannels-1, 2, warpdeg**2), polynomial warpping coefficients from the reference (0-th) channel to the others
        Ts2R:           torch.FloatTensor of shape (nchannels-1, 2, warpdeg**2), polynomial warpping coefficients from the others to the reference (0-th) channel
        i_type:         torch.dtype, integer data type, default = torch.int32
        f_type:         torch.dtype, float data type, default = torch.float32
    Raises:
        ValueError:     For invalid dimensionality, extents, or channel configurations
        TypeError:      For unsupported data types
    NOTE:
        Subclasses must override data_attr with specific data attributes.
        All subclasses must include 'xyz' and 'phot' in data_attr.
        Multi-channel data must include 'fracN' in data_attr.
    """
    
    _rtol = 1E-6    # relative tolerance to determine if a float is close to another 
    _atol = 1E-8    # absolute tolerance to determine if a float is close to another

    meta_attr = ['ndim', 'xextent', 'yextent', 'zextent', 'nchannels', 'R2Ts', 'Ts2R', 'i_type', 'f_type']
    data_attr = [] 
    
    def __init__(self, ndim:int, xextent:Tuple[int, int], yextent:Tuple[int, int], zextent:Tuple[int, int]=None,
                 nchannels:int=1, R2Ts:torch.Tensor=None, Ts2R:torch.Tensor=None, 
                 i_type:torch.dtype=torch.int32, f_type:torch.dtype=torch.float32):
        
        if i_type not in (torch.int16, torch.int32, torch.int64):
            raise TypeError(f"{i_type} is not supported for torch integers")
        if f_type not in (torch.float16, torch.float32, torch.float64):
            raise TypeError(f"{f_type} is not supported for torch floats")
        self.i_type = i_type
        self.f_type = f_type

        if ndim not in {2, 3}:
            raise ValueError(f"Only 2- or 3-dimensionality is supported, but the given ndim is {ndim}")
        self.ndim = ndim
        self.xextent = xextent
        self.yextent = yextent
        self.zextent = zextent
        
        if nchannels < 1:
            raise ValueError(f"{self.nchannels} is invalid number for single-or multi-channels")
        self.nchannels = nchannels
        self.R2Ts = None if R2Ts is None else to_tensor(R2Ts, dtype=self.f_type, device='cpu')
        self.Ts2R = None if Ts2R is None else to_tensor(Ts2R, dtype=self.f_type, device='cpu')

    @property
    def meta(self) -> dict:
        """Return metadata of LocSet"""
        return {k: getattr(self, k) for k in self.meta_attr}

    @property
    def data(self) -> dict:
        """Return intrinsic data (without metadata)"""
        return {k: getattr(self, k) for k in self.data_attr}
        
    
    ########## Basics ##########
    def __len__(self) -> int:
        """Implements length of LocSet."""
        return int(self.xyz.size(0))

    def __iter__(self) -> Union[FrameLocSet, EmitterLocSet]:
        """Implements iterator bookkeeping."""
        self.n = 0
        return self

    def __next__(self) -> Union[FrameLocSet, EmitterLocSet]:
        """Implements next element in iterator method. Returns the next localization (as a subset of the LocSet)"""
        if self.n < len(self):
            self.n += 1
            return self._get_subset(self.n - 1)
        else:
            raise StopIteration

    def __getitem__(self, ix) -> Union[FrameLocSet, EmitterLocSet]:
        """Implements array indexing for this class. Returns the subset of a LocSet indexed by item"""
        if isinstance(ix, int) and ix >= len(self):
            raise IndexError(f"Index {ix} out of bounds of LocSet of size {len(self)}")
        return self._get_subset(ix)
    
    def __add__(self, other) -> Union[FrameLocSet, EmitterLocSet]:
        return self.__class__.loc_cat((self, other))

    def __iadd__(self, other) -> Union[FrameLocSet, EmitterLocSet]:
        self._inplace_replace(self + other)
        return self

    @abstractmethod
    def __str__(self):
        """
        Friendly representation of the class
        Inheriting class can use _meta_str() to represent the meta, and add data information after
        """
        raise NotImplementedError
    
    
    ########## Public ##########
    def clone(self) -> Union[FrameLocSet, EmitterLocSet]:
        """Returns a deep copy of this LocSet."""
        return copy.deepcopy(self)
    
    def to_dict(self) -> dict:
        """
        Returns dictionary representation of this LocSet so that the keys and variables correspond to what an LocSet would be initialised.
        Example:
            >>> loc_dict = loc.to_dict()
            >>> loc_clone = LocSet(**loc_dict)
        """
        loc_dict = {}
        loc_dict.update(self.meta)
        loc_dict.update(self.data)
        return loc_dict
    
    def init_load(self, **data) -> None:
        """
        Loading data to an empty LocSet (LocSet with meta but no data)
        Example:
            >>> loc = LocSet(**meta_dict)  # initialize an empty LocSet
            >>> loc.init_load(**data_dict)
        """
        if any([len(getattr(self, k)) > 0 for k in self.data_attr]):
            raise ValueError("cannot load data to a non-empty LocSet")
        loc_dict = {k: v for k, v in data.items() if k in self.data_attr}
        loc_dict.update(self.meta)
        self.__init__(sanity_check=True, **loc_dict)

    def meta_isclose(self, other:Union[FrameLocSet, EmitterLocSet]) -> bool:
        """return ture if the meta of the two locsets are within the given precision"""
        if set(self.meta_attr) != set(other.meta_attr):
            return False
        for k in self.meta_attr:
            v_self = getattr(self, k)
            v_other = getattr(other, k)
            if k not in {'R2Ts', 'Ts2R'} and v_self != v_other:
                return False
            elif self.nchannels > 1 and k in {'R2Ts', 'Ts2R'} and \
                (not torch.isclose(v_self, v_other, rtol=self._rtol, atol=self._atol, equal_nan=True).all().item()):
                return False    
        return True
    
    def get_activeattr(self) -> List[str]:
        """Get the attributes for the active (length > 0) attributes"""
        return [key for key in self.data_attr if len(getattr(self, key)) > 0]
    
    @classmethod
    def loc_cat(cls, locsets:List[Union[FrameLocSet, EmitterLocSet]]) -> Union[FrameLocSet, EmitterLocSet]:
        """
        Concatenate multiple localization sets into a single set.
        All input localization sets must have identical metadata and active attributes.
        All input localization sets are assumed to be sanity checked already.
        Args:
            locsets: List of LocSet instances to concatenate
        Returns:
            Union[FrameLocSet, EmitterLocSet]: Concatenated LocSet instance
        Raises:
            RuntimeError: If metadata or active attributes don't match across input sets
            ValueError: If input list is empty
        Example:
            >>> combined = FrameLocSet.loc_cat([locset1, locset2, locset3])
        """
        if len(locsets) == 0:
            raise ValueError("empty list is not supported")
        
        # make sure the meta for the locsets are identical before concatenating
        if not all([locsets[0].meta_isclose(loc) for loc in locsets[1:]]): 
            raise RuntimeError("meta mismatch: locsets with different meta are not concatenatable")
        meta_dict = locsets[0].meta

        # make sure all the LocSet in locsets has the same active attributes
        active_attr = []
        for loc in locsets:
            active_attr = loc.get_activeattr()
            if len(active_attr) > 0:
                break
        
        if len(active_attr) == 0:
            return locsets[0]
        
        for loc in locsets:
            active_attr_tmp = loc.get_activeattr()
            if len(active_attr_tmp) > 0 and (set(active_attr) != set(active_attr_tmp)):
                raise RuntimeError("attribute mismatch: locsets with different active attributes are not concatenatable")
        
        # concatenate the active attributes
        data_collection = [loc.data for loc in locsets]
        data_dict = {k: torch.cat([d[k] for d in data_collection], dim=0) for k in active_attr}
        
        loc_dict = {}
        loc_dict.update(meta_dict)
        loc_dict.update(data_dict)
        return cls(sanity_check=False, **loc_dict)
    
    def split_to_chunks(self, nchunks:int) -> List[Union[FrameLocSet, EmitterLocSet]]:
        """
        Splits the locset into (almost) equal chunks
        https://stackoverflow.com/questions/2130016/splitting-a-list-into-n-parts-of-approximately-equal-length/37414115#37414115
        Args:
            nchunks: number of chunks to split
        Returns:
            List: list of sublocset
        """
        if nchunks <= 0:
            raise ValueError(f"nchunks must be positive, but got nchunks={nchunks}")
        if nchunks > len(self):
            raise ValueError(f"nchunks ({nchunks}) exceeds number of localizations ({len(self)})")
            
        n = len(self)
        k = nchunks
        return [self[i * (n // k) + min(i, n % k) : (i + 1) * (n // k) + min(i + 1, n % k)] for i in range(k)]

    def fov_filter(self, xextent:Tuple[int, int]=None, yextent:Tuple[int, int]=None, zextent:Tuple[int, int]=None) -> torch.BoolTensor:
        """
        Returns the boolean array of the localizations within the given extents
        No checking on whether the extents satisfy extent[0] < extent[1]
        Args:
            xextent:    [west, east], the x extent
            yextent:    [north, south], the y extent
            zextent:    [up, bottom], the z extent
        Returns:
            mask:       (len(self)) boolean tensor indicating the localizations within the given ranges
        """
        if xextent is not None and xextent[0] >= xextent[1]:
            raise ValueError("xextent should be ascendingly ordered")
        if yextent is not None and yextent[0] >= yextent[1]:
            raise ValueError("yextent should be ascendingly ordered")
        if zextent is not None and zextent[0] >= zextent[1]:
            raise ValueError("zextent should be ascendingly ordered")
        
        _mask = torch.ones((len(self),), dtype=torch.bool)
        
        xextent = self.xextent if xextent is None else xextent
        _mask = _mask & (self.xyz[:, 0] >= xextent[0]) & (self.xyz[:, 0] < xextent[1])

        yextent = self.yextent if yextent is None else yextent
        _mask = _mask & (self.xyz[:, 1] >= yextent[0]) & (self.xyz[:, 1] < yextent[1])

        if self.ndim == 3: 
            zextent = self.zextent if zextent is None else zextent
            _mask = _mask & (self.xyz[:, 2] >= zextent[0]) & (self.xyz[:, 2] < zextent[1])
        return _mask
    
    def get_infov(self, xextent:Tuple[int, int]=None, yextent:Tuple[int, int]=None, zextent:Tuple[int, int]=None) -> Union[FrameLocSet, EmitterLocSet]:
        """
        Returns the subset of the localizations within the given extents
        Args:
            xextent:    [west, east], the x extent
            yextent:    [north, south], the y extent
            zextent:    [up, bottom], the z extent
        Return:
            LocSet:     a sub_instance of the Locset
        """
        mask = self.fov_filter(xextent=xextent, yextent=yextent, zextent=zextent)
        sub_instance = self[mask]
        if xextent is not None:
            setattr(sub_instance, 'xextent', xextent)
        if yextent is not None:
            setattr(sub_instance, 'yextent', yextent)
        if zextent is not None:
            setattr(sub_instance, 'zextent', zextent)
        return sub_instance
    
    def phot_filter(self, threshold:float=0.0) -> Union[FrameLocSet, EmitterLocSet]:
        """filter the localization set with photon threshold"""
        return self[self.phot >= threshold]

    
    ########## Private ##########
    def _sanity_check(self) -> None:
        """sanity checks on the meta"""

        # sanity checks on the structual meta
        if self.xextent[0] >= self.xextent[1]: 
            raise ValueError(f"invalid xextent, xextent[0] should smaller than xextent[1], but the input is {self.xextent}")
        if self.yextent[0] >= self.yextent[1]:
            raise ValueError(f"invalid yextent, yextent[0] should smaller than yextent[1], but the input is {self.yextent}")
        if self.ndim == 3 and self.zextent[0] >= self.zextent[1]:
            raise ValueError(f"invalid zextent, zextent[0] should smaller than zextent[1], but the input is {self.zextent}")
        
        # sanity checks on the multi-channel meta
        if self.nchannels == 1:
            self.R2Ts = None
            self.Ts2R = None
        else:
            if (self.R2Ts is None and self.Ts2R is not None) or (self.R2Ts is not None and self.Ts2R is None):
                raise ValueError("R2Ts and Ts2R must all be None or not None")
            elif self.R2Ts is not None:
                if self.R2Ts.shape != self.Ts2R.shape:
                    raise ValueError(f"R2Ts and Ts2R should be of the same size, but R2Ts.size()={self.R2Ts.shape}, Ts2R.size()={self.Ts2R.shape}")
                if self.R2Ts.shape[0] != self.nchannels - 1:
                    raise ValueError(f"shape mismatch: R2Ts must be of shape ((nchannels-1, warpdeg**2, 2)), but R2Ts.size(0)={self.R2Ts.shape[0]} while nchannels={self.nchannels}")
                if self.R2Ts.shape[1] != math.isqrt(self.R2Ts.shape[1]) ** 2:
                    raise ValueError(f"shape mismatch: R2Ts must be of shape ((nchannels-1, warpdeg**2, 2)), but R2Ts.size(1)={self.R2Ts.shape[1]} is not perfect square")
                if self.R2Ts.shape[2] != 2:
                    raise ValueError(f"shape mismatch: R2Ts must be of shape ((nchannels-1, warpdeg**2, 2)), but R2Ts.size(2)={self.R2Ts.shape[2]}")
                self.R2Ts = to_tensor(self.R2Ts, dtype=self.f_type, device='cpu')
                self.Ts2R = to_tensor(self.Ts2R, dtype=self.f_type, device='cpu')
            else:
                warnings.warn(f"channels transformation acorss {self.nchannels}-channels were not specified. An identical transformation will be assumed")
                self.R2Ts = torch.zeros((self.nchannels-1, 4, 2), dtype=self.f_type, device='cpu')
                for i in range(self.nchannels - 1):
                    self.R2Ts[i, 1, 0] = 1.0
                    self.R2Ts[i, 2, 1] = 1.0
                self.Ts2R = self.R2Ts.detach().clone()

        # set data types
        for k in self.data_attr:
            dtype_ = self.i_type if k in {'indf', 'id'} else self.f_type
            setattr(self, k, to_tensor(getattr(self, k), dtype=dtype_, device='cpu'))
        
        # sanity checks on the dimensionality for xyz, xyz_sig, fracN, and fracN_sig
        if self.xyz.size(1) != self.ndim:
            raise ValueError(f"ndim mismatch: xyz.size(1)={self.xyz.size(1)}, ndim={self.ndim}")
        if hasattr(self, 'xyz_sig') and self.xyz_sig.size(1) != self.ndim:
            raise ValueError(f"ndim mismatch: xyz_sig.size(1)={self.xyz_sig.size(1)}, ndim={self.ndim}")
        if self.nchannels > 1:
            if self.fracN.size(1) != self.nchannels:
                raise ValueError(f"nchannels mismatch: fracN.size(1)={self.fracN.size(1)}, nchannels={self.nchannels}")
            if hasattr(self, 'fracN_sig') and self.fracN_sig.size(1) != self.nchannels:
                raise ValueError(f"nchannels mismatch: fracN_sig.size(1)={self.fracN_sig.shape[1]}, nchannels={self.nchannels}")
        
        # sanity checks on the length of the data
        _nlocs = self.xyz.size(0)
        for k in self.data_attr:
            val = getattr(self, k)
            if val.size(0) != 0 and val.size(0) != _nlocs:
                raise ValueError(f"nlocs mismatch: nlocs={_nlocs}, {k}.size(0)={val.size(0)}") 

    def _inplace_replace(self, other:Union[FrameLocSet, EmitterLocSet]) -> None:
        """
        Inplace replacement of the self instance with another
        NOTE: The substituting instance is assumed to be sanity checked
        """
        if not isinstance(other, type(self)):
            raise TypeError(f"type mismatch: type(self)={type(self)}, type(other)={type(other)}")
        self.__init__(sanity_check=False, **other.to_dict())

    def _get_subset(self, ix:Union[int, list, np.ndarray, torch.Tensor, slice]) -> Union[FrameLocSet, EmitterLocSet]:
        """
        Returns subset of a locset. Implementation of __getitem__ and __next__ methods.
        Inheriting class should overwrite this method by changing the return type to itself
        Args:
            ix: integer, or list of indices, or slice
        Returns:
            LocSet: the subset of the instance
        NOTE: the method creates new instance instead of using deepcopy for performance purpose
        """
        ix = to_tensorslicer(ix)
        if isinstance(ix, slice) and len(range(*ix.indices(len(self)))) == 0:
            return self.__class__(sanity_check=False, **self.meta)
        if isinstance(ix, list) and len(ix) == 0:
            return self.__class__(sanity_check=False, **self.meta)
        
        sub_instance = self.__class__(sanity_check=False, **self.meta)
        for key in self.data_attr:
            val = getattr(self, key)
            if val.size(0) > 0:
                setattr(sub_instance, key, val[ix])
        return sub_instance
    
    def _meta_str(self) -> str:
        """
        representation of the Meta of a LocSet, can be implemented in __str__ in inheriting class
        Returns:
            (string) representation of the Meta of this class
        """
        print_str = f"{self.__class__.__name__}"\
                    f"\n:: number of dimensions: {self.ndim}"\
                    f"\n:: xextent = [{self.xextent[0]}, {self.xextent[1]})"\
                    f"\n:: yextent = [{self.yextent[0]}, {self.yextent[1]})"
        if self.ndim == 3:
            print_str += f"\n:: zextent = [{self.zextent[0]}, {self.zextent[1]})"
                    
        print_str += f"\n:: number of channels: {self.nchannels}"
        print_str += f"\n:: integer type={self.i_type}, float type={self.f_type}"
        return print_str



class FrameLocSet(LocSet):
    """
    generic data type to represent a list of frame localizations as well as its properties
    Args (Meta):
        ndim:           number of dimension
        xextent:        [west, east], extents in x-axis
        yextent:        [north, south], extents in y-axis
        zextent:        [up, bottom], extents in z-axis
        nchannels:      number of channels
        R2Ts:           (nchannels - 1, 2, warpdeg**2), polynomial warpping coefficients from the reference (0-th) channel to the others
        Ts2R:           (nchannels - 1, 2, warpdeg**2), polynomial warpping coefficients from the others to the reference (0-th) channel
        i_type:         integer data type
        f_type:         float data type
    Args (Data):
        id:             (nlocs), ID of each emitter
        prob:           (nlocs), probability feature of each emitter
        indf:           (nlocs), frame index of each localization
        xyz:            (nlocs, ndim), [[x, y(, z)],...] of each localization in the reference (0-th) channel
        phot:           (nlocs), photon of each localization
        fracN:          (nlocs, nchannels), photon distributions across different channels
        xyz_sig:        (nlocs, ndim), [[x_sig, y_sig(, z_sig)],...] of each localization in the reference (0-th) channel
        phot_sig:       (nlocs), photon sigma of each localization
        fracN_sig:      (nlocs, nchannels), sigma of photon distributions across different channels
    """
    data_attr = ['id', 'prob', 'indf', 'xyz', 'phot', 'fracN', 'xyz_sig', 'phot_sig', 'fracN_sig']

    def __init__(self, ndim:int, xextent:Tuple[int, int], yextent:Tuple[int, int], zextent:Tuple[int, int]=None,
                 nchannels:int=1, R2Ts:torch.Tensor=None, Ts2R:torch.Tensor=None, 
                 i_type:torch.dtype=torch.int32, f_type:torch.dtype=torch.float32,
                 id:torch.IntTensor=None, prob:torch.Tensor=None, indf:torch.IntTensor=None, 
                 xyz:torch.Tensor=None, phot:torch.Tensor=None, fracN:torch.Tensor=None,
                 xyz_sig:torch.Tensor=None, phot_sig:torch.Tensor=None, fracN_sig:torch.Tensor=None, sanity_check:bool=True):
       
        super().__init__(ndim=ndim, xextent=xextent, yextent=yextent, zextent=zextent, 
                         nchannels=nchannels, R2Ts=R2Ts, Ts2R=Ts2R, i_type=i_type, f_type=f_type)
        self.id = torch.zeros((0,), dtype=self.i_type, device='cpu') if id is None else id
        self.prob = torch.zeros((0,), dtype=self.f_type, device='cpu') if prob is None else prob
        self.indf = torch.zeros((0,), dtype=self.i_type, device='cpu') if indf is None else indf
        self.xyz = torch.zeros((0, self.ndim), dtype=self.f_type, device='cpu') if xyz is None else xyz
        self.phot = torch.zeros((0,), dtype=self.f_type, device='cpu') if phot is None else phot
        self.fracN = torch.zeros((0, self.nchannels), dtype=self.f_type, device='cpu') if fracN is None else fracN
        self.xyz_sig = torch.zeros((0, self.ndim), dtype=self.f_type, device='cpu') if xyz_sig is None else xyz_sig
        self.phot_sig = torch.zeros((0,), dtype=self.f_type, device='cpu') if phot_sig is None else phot_sig
        self.fracN_sig = torch.zeros((0, self.nchannels), dtype=self.f_type, device='cpu') if fracN_sig is None else fracN_sig
        if sanity_check:
            self._sanity_check()

    @property
    def frame_rng(self) -> Tuple[int, int]:
        """frame range: left inclusive and right exclusive"""
        if len(self.indf) == 0:
            raise ValueError("self.indf.numel() == 0, frame range is not available")
        return self.indf.min(), self.indf.max() + 1 
    
    @property
    def nfrm(self) -> int:
        if len(self.indf) == 0:
            return 0
        return self.indf.max() - self.indf.min() + 1
    
    @property
    def active_frames(self) -> torch.IntTensor:
        return torch.unique(self.indf, sorted=True)
    
    def sort_by_frame(self) -> FrameLocSet:
        """Sort a deepcopy of this emitterset and return it"""
        if len(self.indf) == 0:
            return self
        _, ix = self.indf.sort()
        return self[ix]
    
    def sort_by_frame_(self):
        """Inplace sort this emitterset by its frame index."""
        if len(self.indf) == 0:
            return self
        frame_sorted = self.sort_by_frame()
        self._inplace_replace(frame_sorted)
        return
    
    def split_by_frame(self, loose=False) -> List[FrameLocSet]:
        """
        Splits the locset by frame
        Returns:
            List: list of sublocset
        """
        if len(self.indf) == 0:
            return [self]
        
        frame_set = torch.arange(*self.frame_rng) if loose else self.active_frames
        frame_sorted = self.sort_by_frame()
        ix_edges = torch.searchsorted(frame_sorted.indf, frame_set)
        ix_edges = torch.cat((ix_edges, torch.tensor([len(self)], dtype=ix_edges.dtype)))
        out = [frame_sorted[ix_edges[i] : ix_edges[i+1]] for i in range(len(ix_edges)-1)]
        return out

    def get_subframe(self, frame_rng:Tuple[int, int]) -> FrameLocSet:
        """Returns the subset of the instance according to the frame range"""
        if frame_rng[0] > frame_rng[1]:
            raise ValueError("frame_rng must be frame_rng[0] < frame_rng[1] (left-inclusive and right-exclusive)")
        if frame_rng[0] == frame_rng[1]:
            ix = self.indf == frame_rng[0]
        else:
            ix = (self.indf >= frame_rng[0]) & (self.indf < frame_rng[1])
        return self[ix]
    
    def realign_frame(self, frame_offset:int) -> FrameLocSet:
        """ Readjust the indf by adding the offset to the frames """         
        new = self.clone()
        new.indf += frame_offset
        return new
    
    def __str__(self) -> str:
        """Friendly representation of FrameLocSet"""
        print_str = self._meta_str()
        print_str += f"\n:: number of localizations: {len(self)}"
        if len(self) > 0:
            print_str += f"\n:: frame_rng: [{self.indf.min()}-{self.indf.max()+1})"
        return print_str
    
    

class EmitterLocSet(LocSet):
    """
    generic data type to represent a list of emitters as well as its properties
    Args (Meta):
        ndim:           number of dimension
        nchannels:      number of channels
        R2Ts:           (nchannels - 1, 2, warpdeg**2), polynomial warpping coefficients from the reference (0-th) channel to the others
        Ts2R:           (nchannels - 1, 2, warpdeg**2), polynomial warpping coefficients from the others to the reference (0-th) channel
        xextent:        [west, east], extents in x-axis
        yextent:        [north, south], extents in y-axis
        zextent:        [up, bottom], extents in z-axis
        i_type:         integer data type
        f_type:         float data type 
    Args (Data):
        id:             (nemitters), id of each emitter
        xyz:            (nemitters, ndim), [[x, y(, z)],...] of each emitter in the reference (0-th) channel
        phot:           (nemitters), photon flux (photon / sec) of each emitter
        fracN:          (nemitters, nchannels), photon distributions across different channels
        t0:             (nemitters), the time an emitter is turned on
        te:             (nemitters), the time an emitter is turned off 
    """
    data_attr = ['id', 'xyz', 'phot', 'fracN', 't0', 'te']

    def __init__(self, ndim:int, xextent:Tuple[int, int], yextent:Tuple[int, int], zextent:Tuple[int, int]=None,
                 nchannels:int=1, R2Ts:torch.Tensor=None, Ts2R:torch.Tensor=None, 
                 i_type:torch.dtype=torch.int32, f_type:torch.dtype=torch.float32,
                 id:torch.IntTensor=None, xyz:torch.Tensor=None, phot:torch.Tensor=None, fracN:torch.Tensor=None,
                 t0:torch.Tensor=None, te:torch.Tensor=None, sanity_check:bool=True):
       
        super().__init__(ndim=ndim, xextent=xextent, yextent=yextent, zextent=zextent, 
                         nchannels=nchannels, R2Ts=R2Ts, Ts2R=Ts2R, i_type=i_type, f_type=f_type)
        self.id = torch.zeros((0,), dtype=self.i_type, device='cpu') if id is None else id
        self.xyz = torch.zeros((0, self.ndim), dtype=self.f_type, device='cpu') if xyz is None else xyz
        self.phot = torch.zeros((0,), dtype=self.f_type, device='cpu') if phot is None else phot
        self.fracN = torch.zeros((0, self.nchannels), dtype=self.f_type, device='cpu') if fracN is None else fracN
        self.t0 = torch.zeros((0,), dtype=self.f_type, device='cpu') if t0 is None else t0
        self.te = torch.zeros((0,), dtype=self.f_type, device='cpu') if te is None else te
        if sanity_check:
            self._sanity_check()

    def to_frameloc(self, frame_rng:Tuple[int, int]) -> FrameLocSet:
        """
        convert an EmitterLocSet to a FrameLocSet
        Args:
            frame_rng:  start and end frame index to sample the emitters (inclusion for frame_rng[0], exclusion for frame_rng[1])
        Returns:
            FrameLocSet: The FrameLocSet corresponding to the EmitterLocSet instance
        """
        if frame_rng[0] > frame_rng[1]:
            raise ValueError("invalid frame_rng, should be frame_rng[0] <= frame_rng[1]")
        ix_low = frame_rng[0]
        ix_high = ix_low + 1 if frame_rng[0] == frame_rng[1] else frame_rng[1]

        indf = torch.zeros((0,), dtype=self.i_type, device='cpu')
        xyz = torch.zeros((0, self.ndim), dtype=self.f_type, device='cpu')
        phot = torch.zeros((0,), dtype=self.f_type, device='cpu')
        fracN = torch.zeros((0, self.nchannels), dtype=self.f_type, device='cpu')
        
        for f in range(ix_low, ix_high):
            _ind = (self.t0 < f + 1) & (self.te > f)
            if torch.any(_ind):
                durations = torch.clamp(self.te[_ind], max=f+1) - torch.clamp(self.t0[_ind], min=f)

                indf = torch.cat([indf, f + torch.zeros((_ind.sum(),), dtype=self.i_type)], dim=0)
                xyz = torch.cat([xyz, self.xyz[_ind]], dim=0)
                phot = torch.cat([phot, self.phot[_ind] * durations], dim=0)
                if self.nchannels > 1:
                    fracN = torch.cat([fracN, self.fracN[_ind]], dim=0)
        id = torch.arange(len(indf), dtype=self.i_type, device='cpu')
        frame_loc = FrameLocSet(**self.meta)
        frame_loc.init_load(id=id, indf=indf, xyz=xyz, phot=phot, fracN=fracN)
        return frame_loc
    
    def __str__(self) -> str:
        """Friendly representation of FrameLocSet"""
        print_str = self._meta_str()
        print_str += f"\n:: number of localizations: {len(self)}"
        return print_str