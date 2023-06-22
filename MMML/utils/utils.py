import torch
from torch.nn import ModuleDict, ModuleList,Module
from torch import Tensor
import os
from typing import Any
from dataclasses import dataclass

def transfer_to(x, device):
     
    if isinstance(x, dict):
        for key in x:
            x[key] = transfer_to(x[key], device)
    elif isinstance(x, list):
        for i in range(len(x)):
            x[i] = transfer_to(x[i], device)
    elif any([parent_cls is Module for parent_cls in x.__class__.__mro__]):#TODO : test this
        x= x.to(device)
    elif isinstance(x, Tensor):
        x= x.to(device)
    else:
        x = x
    return x

from torch.utils.data import DataLoader, Dataset

@dataclass
class TrainingConfig:
    training_methode : str
    training_module:Any
    dataset : Dataset
    dataloader_kwargs: dict
    optimizer: Any
    scheduler: Any
    epochs : int
    callbacks : list
    start_epoch = 0
    name_split = ""
    
    @property
    def dataloader(self):
        return DataLoader(self.dataset, **self.dataloader_kwargs)

