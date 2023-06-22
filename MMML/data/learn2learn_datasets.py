from learn2learn.vision.datasets.cifarfs import CIFARFS
from torchvision import transforms

DICT_NORMALIZATION = dict(
    cifar = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465], std=[0.247, 0.243, 0.261]
        ),
    ])
)


def get_normalized_cifarfs(root,
                 mode='train',
                 transform=None,
                 target_transform=None,
                 download=False):
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
    if transform is None:
        return CIFARFS(root, mode=mode, transform=DICT_NORMALIZATION["cifar"], target_transform= target_transform, download= download)
    transform = transforms.Compose([DICT_NORMALIZATION["cifar"], transform])
    return CIFARFS(root, mode=mode, transform=transform, target_transform= target_transform, download= download)