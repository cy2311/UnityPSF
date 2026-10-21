from typing import Union
import hashlib
from pathlib import Path
import time

import math
import torch


def hash_model(modelfile:Union[Path, str]):
    """
    Calculate hash of a model file
    https://www.pythoncentral.io/hashing-files-with-python/
    """
    blocksz = 65536
    hasher = hashlib.sha1()
    with open(modelfile, 'rb') as fid:
        buff = fid.read(blocksz)
        while len(buff) > 0:
            hasher.update(buff)
            buff = fid.read(blocksz)
    return hasher.hexdigest()



class LoadSaveModel:

    def __init__(self, model_instance:torch.nn.Module, output_file:Union[str, Path], input_file:Union[str, Path]=None, 
                 name_time_interval:float=3600.0, better_th:float=1e-6, max_num_files:int=3, state_dict_update:dict=None):
        
        self.model = model_instance
        self.output_file = None if output_file is None else Path(output_file) if not isinstance(output_file, Path) else output_file
        self.warmstart_file = None if input_file is None else Path(input_file) if not isinstance(input_file, Path) else input_file
        self.output_file_suffix = -1 # will be increased to 0 in the first round
        self.max_num_files = max_num_files if max_num_files > 0 else float('inf')
        
        self.name_time_interval = name_time_interval
        self._last_saved_time = 0.0 # float, timestamp when it was last saved
        self._new_name_time = 0.0   # float, timestamp when the name last changed
        
        self._best_metric_val = math.inf
        self.better_th = better_th
        
        self.state_dict_update = state_dict_update

    
    def _creat_target_dir(self):
        """Creates the target folder for the  model .pt file"""
        p = self.output_file
        if p is None:
            raise ValueError("output_file is set None during instantiation of the LoadSaveModel")
        
        try:
            p.parents[0].mkdir(parents=False, exist_ok=True)
        except FileNotFoundError:
            raise FileNotFoundError(f"cannot find folder {p.parents[1]} to make folder {p.parents[0]}")
        
    
    def load_init(self, device:Union[str, torch.device]) -> torch.nn.Module:
        """
        Initialize and warmstart model (if possible) and ship to sepcified device
        Args:
            device:             target device to ship the model to
        Returns:
            torch.nn.Module:    the initialized model
        """
        
        model = self.model
        print("Model instanitiated")
        if self.warmstart_file is None:
            print("Model instanitiated as specified in the constructor")
        else:
            hashv = hash_model(self.warmstart_file)
            print(f"Model SHA-1 hash: {hashv}")
            model.hashv = hashv
            
            state_dict = torch.load(self.warmstart_file, map_location=device)
            if self.state_dict_update is not None:
                state_dict.update(self.state_dict_update)
            model.load_state_dict(state_dict)
            print(f"Loaded pretrained model: {self.warmstart_file}")
        
        model.to(device)
        return model


    def save(self, model:torch.nn.Module, metric_val=None):
        """Save model (conditioned on a better metric if is provided)"""

        self._creat_target_dir()
        
        # examine if the relative difference to previous value is less than the threshold difference
        if metric_val is not None:
            rel_diff = metric_val / self._best_metric_val
            if rel_diff > 1 - self.better_th:
                return    
            self._best_metric_val = metric_val

        # After a certain period, change the suffix
        if metric_val is None or (time.time() > self._new_name_time + self.name_time_interval):
            self.output_file_suffix += 1
            if self.output_file_suffix > self.max_num_files - 1:
                self.output_file_suffix = 0
            self._new_name_time = time.time()

        # Determine file name and save
        fname = Path(str(self.output_file.with_suffix('')) + '_' + str(self.output_file_suffix) + '.pt')
        torch.save(model.state_dict(), fname)
        print(f"Saved model to file: {fname}")
        self._last_saved_time = time.time()        