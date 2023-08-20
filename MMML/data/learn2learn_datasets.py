from learn2learn.vision.datasets import CIFARFS, MiniImagenet
from typing import Optional, Callable
from torchvision import transforms
from MMML.data import get_resulting_transform

DICT_NORMALIZATION = dict(
    cifar=transforms.Normalize(
        mean=[0.5088, 0.4917, 0.4437], std=[0.2003, 0.1985, 0.2031]
    ),
    miniimagenet=transforms.Normalize(
        mean=[120.0492, 114.6121, 102.6735], std=[58.7272, 57.8338, 57.9294]
    ),
)


def get_normalized_cifarfs(
    root,
    mode="train",
    transform_pil: Optional[Callable] = None,
    transform=None,
    target_transform=None,
    download=False,
):
    """Get the following normalized dataset :


    **Description**

    The CIFAR Few-Shot dataset as originally introduced by Bertinetto et al., 2019.

    It consists of 60'000 colour images of sizes 32x32 pixels. The dataset is divided in 3 splits of 64 training, 16 validation, and 20 testing classes each containing 600 examples. The classes are sampled from the CIFAR-100 dataset, and we use the splits from Bertinetto et al., 2019.

    **References**

    Bertinetto et al. 2019. "Meta-learning with differentiable closed-form solvers". ICLR.
    **Arguments**

    **root** (str) - Path to download the data.
    **mode** (str, *optional*, default='train') - Which split to use. Must be 'train', 'validation', or 'test'.
    **transform** (Transform, *optional*, default=None) - Input pre-processing.
    **target_transform** (Transform, *optional*, default=None) - Target pre-processing.
    **Example**

    ~~~python train_dataset = l2l.vision.datasets.CIFARFS(root='./data', mode='train') train_dataset = l2l.data.MetaDataset(train_dataset) train_generator = l2l.data.TaskGenerator(dataset=train_dataset, ways=ways)
    """

    transform = get_resulting_transform(
        transform_pil,
        transform,
        DICT_NORMALIZATION["cifar"],
    )
    return CIFARFS(
        root,
        mode=mode,
        transform=transform,
        target_transform=target_transform,
        download=download,
    )


def get_normalized_miniimagenet(
    root,
    mode="train",
    transform_pil: Optional[Callable] = None,
    transform=None,
    target_transform=None,
    download=False,
):
    """Get the following normalized dataset :

    Description

    The mini-ImageNet dataset was originally introduced by Vinyals et al., 2016.

    It consists of 60'000 colour images of sizes 84x84 pixels. The dataset is divided in 3 splits of 64 training, 16 validation, and 20 testing classes each containing 600 examples. The classes are sampled from the ImageNet dataset, and we use the splits from Ravi & Larochelle, 2017.

    References

    Vinyals et al. 2016. “Matching Networks for One Shot Learning.” NeurIPS.
    Ravi and Larochelle. 2017. “Optimization as a Model for Few-Shot Learning.” ICLR.
    Arguments

    root (str) - Path to download the data.
    mode (str, optional, default='train') - Which split to use. Must be 'train', 'validation', or 'test'.
    transform (Transform, optional, default=None) - Input pre-processing.
    target_transform (Transform, optional, default=None) - Target pre-processing.
    download (bool, optional, default=False) - Download the dataset if it's not available.
    """
    list_transform = []
    if transform_pil is not None:
        list_transform.append(transforms.ToPILImage())
        list_transform.append(transform_pil)
        list_transform.append(transforms.ToTensor())
    if transform is not None:
        list_transform.append(transform)
    transform = None
    if len(list_transform) > 0:
        transform = transforms.Compose(list_transform)

    return MiniImagenet(
        root,
        mode=mode,
        transform=transform,
        target_transform=target_transform,
        download=download,
    )


if __name__ == "__main__":
    from tqdm import tqdm

    def get_mean_std(loader):
        mean = 0.0
        std = 0.0
        for images, _ in tqdm(loader):
            batch_samples = images.size(
                0
            )  # batch size (the last batch can have smaller size!)
            images = images.view(batch_samples, images.size(1), -1)
            mean += images.mean(2).sum(0)
            std += images.std(2).sum(0)

        mean /= len(loader.dataset)
        std /= len(loader.dataset)
        return mean, std

    import torch
    from torch.utils.data import DataLoader

    dataset = CIFARFS("datasets", mode="train", transform=transforms.ToTensor())
    loader = DataLoader(dataset, batch_size=128)
    print("cifarfs mean/ std : ", get_mean_std(loader))

    dataset = MiniImagenet("datasets", mode="train")
    loader = DataLoader(dataset, batch_size=128)
    print("miniimagenet mean/ str : ", get_mean_std(loader))
