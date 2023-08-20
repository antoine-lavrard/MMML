"""Contains Wrapper for normalized datasets"""
from torchvision.datasets import CIFAR10, CIFAR100
from torchvision import transforms
from typing import Optional, Callable

DICT_NORMALIZATION = dict(
    cifar=transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.247, 0.243, 0.261])
)


def get_resulting_transform(transform_pil, transform, normalization):
    list_transform = []

    if transform_pil is not None:
        list_transform.append(transform_pil)
    list_transform.append(transforms.ToTensor())
    if transform is not None:
        list_transform.append(transform)
    list_transform.append(normalization)
    return transforms.Compose(list_transform)


def get_normalized_cifar10(
    root: str,
    train: bool = True,
    transform_pil: Optional[Callable] = None,
    transform: Optional[Callable] = None,
    target_transform: Optional[Callable] = None,
    download: bool = False,
):
    """Get the following normalized dataset :

    `CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    Args:
        root (string): Root directory of dataset where directory
            ``cifar-10-batches-py`` exists or will be saved to if download is set to True.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that takes in a normalized torch tensor
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

    """
    transform = get_resulting_transform(
        transform_pil, transform, DICT_NORMALIZATION["cifar"]
    )
    return CIFAR10(root, train, transform, target_transform, download)


def get_normalized_cifar100(
    root: str,
    train: bool = True,
    pil_transform: Optional[Callable] = None,
    transform: Optional[Callable] = None,
    target_transform: Optional[Callable] = None,
    download: bool = False,
):
    """Get the following normalized dataset :

    `CIFAR100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    Args:
        root (string): Root directory of dataset where directory
            ``cifar-10-batches-py`` exists or will be saved to if download is set to True.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that takes in a normalized torch tensor
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

    """
    transform = get_resulting_transform(
        pil_transform, transform, DICT_NORMALIZATION["cifar"]
    )
    return CIFAR100(root, train, transform, target_transform, download)
