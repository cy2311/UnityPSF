from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Union
import copy
import importlib.resources as pkg_resources

import yaml


class RecursiveNamespace(SimpleNamespace):
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
        out = copy.deepcopy(self.__dict__)
        for key in out:
            if isinstance(out[key], RecursiveNamespace):
                out[key] = out[key].to_dict()
        return out

    def __getitem__(self, item):
        out = getattr(self, item)
        if isinstance(out, RecursiveNamespace):
            raise TypeError
        return out

    def keys(self):
        return self.__dict__.keys()


def load_template(ref_fname: str) -> dict:
    from neptune.smlm_v2a.utils import reference_files

    if not ref_fname.endswith(".yaml"):
        ref_fname = ref_fname + ".yaml"

    ref_template = pkg_resources.open_text(reference_files, ref_fname)
    return yaml.load(ref_template, Loader=yaml.SafeLoader)


def autofill_dict(dst: dict, reference: dict, mode_missing: str = "include") -> dict:
    assert mode_missing in ["include", "exclude"]
    out = {} if mode_missing == "exclude" else dst
    for key, val in reference.items():
        if isinstance(val, dict):
            out[key] = autofill_dict(out[key] if key in dst else {}, val, mode_missing="include")
        elif key in dst and dst[key] is not None:
            out[key] = dst[key]
        else:
            out[key] = reference[key]
    return out


def add_root_relative(path: Union[str, Path], root: Union[str, Path]) -> Path:
    path = Path(path) if not isinstance(path, Path) else path
    root = Path(root) if not isinstance(root, Path) else root
    return path if path.is_absolute() else root / path
