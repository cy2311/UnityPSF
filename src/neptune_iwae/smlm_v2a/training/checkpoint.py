from typing import Union, Optional
from pathlib import Path

import torch

class CheckPoint:
    """
    Checkpointing intended to resume to an already started training.
    Warning:
        Checkpointing is not intended for long-term storage of models or other information.
        No version compatibility guarantees are given here at all.
    """
    
    def __init__(self, path:Union[str, Path]):
        """
        Args:
            path: filename / path where to dump the checkpoints
        """
        self.path = path
        self.step = None
        self.model_state = None
        self.optimizer_state = None
        self.lr_sched_state = None
        self.log_dict = None

    @property
    def dict(self):
        return {'step':self.step, 'model_state':self.model_state, 'optimizer_state':self.optimizer_state, 
                'lr_sched_state':self.lr_sched_state, 'log_dict':self.log_dict}

    def update(self, *, step:int, model_state:dict, optimizer_state:dict, lr_sched_state:dict, log_dict:dict=None):
        self.step = step
        self.model_state = model_state
        self.optimizer_state = optimizer_state
        self.lr_sched_state = lr_sched_state
        self.log_dict = log_dict

    def save(self):
        torch.save(self.dict, self.path)

    @classmethod
    def load(cls, path:Union[str, Path], path_out:Optional[Union[str, Path]]=None):
        
        ckpt_dict = torch.load(path)
        if path_out is None:
            path_out = path
        
        ckpt = cls(path = path_out)
        ckpt.update( step=ckpt_dict['step'], model_state=ckpt_dict['model_state'], optimizer_state=ckpt_dict['optimizer_state'],
                    lr_sched_state=ckpt_dict['lr_sched_state'], log_dict=ckpt_dict['log_dict'] if 'log_dict' in ckpt_dict.keys() else None)
        return ckpt
    
    def dump(self, step:int, model_state:dict, optimizer_state:dict, lr_sched_state:dict, log_dict:dict=None):
        """Updates and saves to file."""
        self.update(step=step, model_state=model_state, optimizer_state=optimizer_state, lr_sched_state=lr_sched_state, log_dict=log_dict)
        self.save()