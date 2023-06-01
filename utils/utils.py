import torch
from torch.nn import ModuleDict, ModuleList


def transfer_to(x, device):
    if isinstance(x, dict) or isinstance(x, ModuleDict):
        for key in x:
            x[key] = transfer_to(x[key], device)
    elif isinstance(x, list) or isinstance(x, ModuleList):
        for i in range(len(x)):
            x[i] = transfer_to(x[i], device)
    else:
        x = x.to(device)
    return x
