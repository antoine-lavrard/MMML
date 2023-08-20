"""
There is a correspondence between the image space and the manifold space for somes transformations 
(ie the network is equivarient by disign to this transformations). For exemple for CNN :
    - translation 
    - scaling
For some transformation, this is not the case :
    - color transformation (will behave like a kernel estimation of the underlying density, set equivalent of bandwith)
    - shape transformation (crop)
    - dropping part of the image (croping)
"""
import torch
from torch import nn
import random
from typing import Union
import numpy as np
from torchvision.transforms import functional as F
import math


class ABCRandomMix(nn.Module):
    """
    Parent class for RandomMixup & RandomCutMix (avoid code duplication)

    """

    def __init__(self, p: float = 0.5, alpha: float = 1.0, inplace: bool = False):
        super().__init__()
        assert alpha > 0, "Alpha param can't be zero."

        self.p = p
        self.alpha = alpha
        self.inplace = inplace

    def assert_compat(self, batch):
        if batch.ndim != 4:
            raise ValueError(f"Batch ndim should be 4. Got {batch.ndim}")

        if not batch.is_floating_point():
            raise TypeError(f"Batch dtype should be a float tensor. Got {batch.dtype}.")

        if not self.inplace:
            batch = batch.clone()

    def mix_data(self, data, lmbd, index):
        raise NotImplementedError

    def forward(self, batch):
        if batch.ndim != 4:
            raise ValueError(f"Batch ndim should be 4. Got {batch.ndim}")

        if not batch.is_floating_point():
            raise TypeError(f"Batch dtype should be a float tensor. Got {batch.dtype}.")

        if not self.inplace:
            batch = batch.clone()

        # think it is slower than Hawkeye implementation without jit
        # random.randint(0, self.number_layer - 1)
        index = torch.arange(0, batch.shape[0], dtype=torch.long).roll(1, 0)

        lambda_param = float(
            torch._sample_dirichlet(torch.tensor([self.alpha, self.alpha]))[0]
        )
        if torch.rand(1).item() >= self.p:
            lmbd = torch.ones((batch.shape[0], 1))
            return batch, lmbd, index
        return self.mix_data(batch, lambda_param, index)


class RandomMixup(ABCRandomMix):
    """Randomly apply Mixup to the provided batch and targets.
    Adapted from https://github.com/Hawkeye-FineGrained/Hawkeye
    The class implements the data augmentations as described in the paper
    `"mixup: Beyond Empirical Risk Minimization" <https://arxiv.org/abs/1710.09412>`_.
    Args:
        p (float): probability of the batch being transformed. Default value is 0.5.
        alpha (float): hyperparameter of the Beta distribution used for mixup.
            Default value is 1.0.
        inplace (bool): boolean to make this transform inplace. Default set to False.
    """

    def mix_data(self, data, lmbd, index):
        data_rolled = data[index, ...]
        data_rolled.mul_(1.0 - lmbd)
        data.mul_(lmbd).add_(data_rolled)

        return data, lmbd, index


class RandomCutmix(ABCRandomMix):
    """
    Randomly apply Cutmix to the provided batch and targets.
    Awdapted from https://github.com/Hawkeye-FineGrained/Hawkeye
    The class implements the data augmentations as described in the paper
    `"CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features"
    <https://arxiv.org/abs/1905.04899>`_.
    Args:
        p (float): probability of the batch being transformed. Default value is 0.5.
        alpha (float): hyperparameter of the Beta distribution used for cutmix.
            Default value is 1.0.
        inplace (bool): boolean to make this transform inplace. Default set to False.
    """

    def mix_data(self, batch, lmbd, index):
        W, H = F.get_image_size(batch)
        r_x = torch.randint(W, (1,))
        r_y = torch.randint(H, (1,))

        r = 0.5 * math.sqrt(1.0 - lmbd)
        r_w_half = int(r * W)
        r_h_half = int(r * H)

        x1 = int(torch.clamp(r_x - r_w_half, min=0))
        y1 = int(torch.clamp(r_y - r_h_half, min=0))
        x2 = int(torch.clamp(r_x + r_w_half, max=W))
        y2 = int(torch.clamp(r_y + r_h_half, max=H))

        batch[:, :, y1:y2, x1:x2] = batch[index, :, y1:y2, x1:x2]
        lmbd = float(1.0 - (x2 - x1) * (y2 - y1) / (W * H))
        return batch, lmbd, index


import torchvision


class SequentialManifoldMix(nn.Module):
    """
    Define a Neural Network where augmentation will be applied on manifold.
    This aims to estimate the joint distribution X,y using some kind of interpolation between two realisation
    belonging to the training set.

    The augmentation will be applied only to one manifold (ie one randomly choosen layer of the network) at each iteration.

    """

    def __init__(
        self,
        modules: nn.Module,
        augmentation: callable = ABCRandomMix(),
    ):
        """
        Initialize the module

        augmentation function :
        The function returns (inputs, labels), in which the inputs is the tensor that contains the mixup images while the
        labels is a tensor that contains (label_batch, label_permuted_batch, lambda) for each image.

        Args:
            modules: iterable of module that will be applied sequentially
            augmentation: callable that return an interpolation between two images
        """
        super().__init__()
        module_list = [module for module in modules]
        self.module_list = nn.ModuleList(module_list)
        self.augmentation = augmentation
        self.number_layer = len(module_list)

    def _forward_train(self, input):
        manifold_id_to_augment = random.randint(0, self.number_layer - 1)

        for module_number, module in enumerate(self.module_list):
            if module_number == manifold_id_to_augment:
                input, lmbd, index = self.augmentation(input)
            input = module(input)

        return input, lmbd, index

    def _forward_eval(self, input):
        for module in self.module_list:
            input = module(input)
        return input

    def forward(self, input):
        if self.training:
            output, lmbd, index = self._forward_train(input)
            return (output, lmbd, index)
        else:
            return self._forward_eval(input)
