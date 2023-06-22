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

from torch import nn
import kornia
import random


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
        augmentation: str = "random_mixup",
    ):
        """
        Initialize the module

        augmentation function :
        The function returns (inputs, labels), in which the inputs is the tensor that contains the mixup images while the
        labels is a tensor that contains (label_batch, label_permuted_batch, lambda) for each image.

        Args:
            modules: iterable of module that will be applied sequentially
            augmentation (kornia.augmentation, optional): augmentation to apply manifold-wise. Should be called like : augmentation(input, label)
            Defaults to kornia.augmentation.RandomMixUpV2().
        """
        super().__init__()
        module_list = [module for module in modules]
        self.module_list = nn.ModuleList(module_list)
        if augmentation == "random_mixup":
            self.augmentation = kornia.augmentation.RandomMixUpV2(
                data_keys=["input", "class"]
            )
        else:
            raise NotImplementedError(
                f"methode {augmentation} is not implemented for manifold mixup"
            )

        self.number_layer = len(module_list)

    def _forward_train(self, input, target):
        manifold_id_to_augment = random.randint(0, self.number_layer - 1)
        for module_number, module in enumerate(self.module_list):
            if module_number == manifold_id_to_augment:
                input, target = self.augmentation(input, target)
            input = module(input)
        return input, target

    def _forward_eval(self, input):
        for module in self.module_list:
            input = module(input)
        return input

    def forward(self, input, target=None):
        if self.training:
            return self._forward_train(input, target)
        else:
            return self._forward_eval(input)
