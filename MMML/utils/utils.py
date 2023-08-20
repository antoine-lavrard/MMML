import torch
from torch.nn import ModuleDict, ModuleList, Module
from torch import Tensor
import os
from typing import Any
from dataclasses import dataclass, field
from torch import nn
from torchinfo import summary


def transfer_to(
    x, device, non_blocking=True
):  # probably not a good idea to keep non_blocking = True as a default
    if isinstance(x, dict):
        for key in x:
            x[key] = transfer_to(x[key], device, non_blocking=non_blocking)
    elif isinstance(x, list):
        for i in range(len(x)):
            x[i] = transfer_to(x[i], device, non_blocking=non_blocking)
    elif any([parent_cls is Module for parent_cls in x.__class__.__mro__]):
        x = x.to(device, non_blocking=non_blocking)
    elif isinstance(x, Tensor):
        x = x.to(device, non_blocking=non_blocking)
    else:
        x = x
    return x


@torch.no_grad()
def update_bn(loader, model, device=None):
    r"""Updates BatchNorm running_mean, running_var buffers in the model.
    copied from torch impl, deleted the input formating


    It performs one pass over data in `loader` to estimate the activation
    statistics for BatchNorm layers in the model.
    Args:
        loader (torch.utils.data.DataLoader): dataset loader to compute the
            activation statistics on. Each data batch should be either a
            tensor, or a list/tuple whose first element is a tensor
            containing data.
        model (torch.nn.Module): model for which we seek to update BatchNorm
            statistics.
        device (torch.device, optional): If set, data will be transferred to
            :attr:`device` before being passed into :attr:`model`.

    """
    momenta = {}
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.running_mean = torch.zeros_like(module.running_mean)
            module.running_var = torch.ones_like(module.running_var)
            momenta[module] = module.momentum

    if not momenta:
        return

    was_training = model.training
    model.train()
    for module in momenta.keys():
        module.momentum = None
        module.num_batches_tracked *= 0

    for input in loader:
        if device is not None:
            input = transfer_to(input, device)

        model(input)

    for bn_module in momenta.keys():
        bn_module.momentum = momenta[bn_module]
    model.train(was_training)


def save_summary(path_output, backbone):
    sum = summary(backbone, input_size=(1, 3, 32, 32), device="cpu", verbose=0)

    with open(os.path.join(path_output, "summary.txt"), "wb") as f:
        sum_str = str(sum).encode("utf8")
        f.write(sum_str)

