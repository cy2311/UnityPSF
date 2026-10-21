from types import SimpleNamespace
from typing import Union
import importlib.resources as pkg_resources
from pathlib import Path

import yaml
import copy

class RecursiveNamespace(SimpleNamespace):
    """
    Extension of SimpleNamespace to recursive dictionaries.
    Inspired by https://dev.to/taqkarim/extending-simplenamespace-for-nested-dictionaries-58e8
    """
    @staticmethod
    def map_entry(entry):
        if isinstance(entry, dict):
            return RecursiveNamespace(**entry)
        return entry

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for key, val in kwargs.items():
            if isinstance(val, dict):
                setattr(self, key, RecursiveNamespace(**val))
            elif isinstance(val, (tuple, list)):
                setattr(self, key, list(map(self.map_entry, val)))

    def to_dict(self):
        d = copy.deepcopy(self.__dict__)  # should not be inplace, otherwise shitty
        for k in d:  # nice little recursion
            if isinstance(d[k], RecursiveNamespace):
                d[k] = d[k].to_dict()
        return d

    def __getitem__(self, item):
        out = getattr(self, item)
        if isinstance(out, RecursiveNamespace):
            raise TypeError
        return out

    def keys(self):
        return self.__dict__.keys()
    


def load_template(ref_fname:str) -> dict:
    """Load the ref_fname.yaml file from reference_files"""
    from . import reference_files

    if not ref_fname.endswith('.yaml'):
        ref_fname = ref_fname + '.yaml'

    ref_template = pkg_resources.open_text(reference_files, ref_fname)
    ref_template = yaml.load(ref_template, Loader=yaml.SafeLoader)
    return ref_template



def autofill_dict(dst:dict, reference:dict, mode_missing:str='include') -> dict:
    """
    Fill the dst dict with keys and values of reference if they are not present in dst
    Args:
        dst:            input dict to fill
        reference:      reference dict
        mode_missing:   'include': directly fill in dst
                        'exclude': create a copy of dist and fill in
    """
    assert mode_missing in ['include', 'exclude']
    out = {} if mode_missing == 'exclude' else dst
    for key, val in reference.items():
        if isinstance(val, dict):
            out[key] = autofill_dict(out[key] if key in dst else {}, val, mode_missing='include')
        elif key in dst and dst[key] is not None:
            out[key] = dst[key]
        else:
            out[key] = reference[key]
    return out



def add_root_relative(path:Union[str, Path], root:Union[str, Path]) -> Path:
    """
    Add the root to a path if the path is not absolute
    Args:
        path:   path to a file
        root:   root path
    Returns:
        Path:   absolute path to a file
    """
    path = Path(path) if not isinstance(path, Path) else path
    root = Path(root) if not isinstance(root, Path) else root
    return path if path.is_absolute() else root / path