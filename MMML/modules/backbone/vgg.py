from torchvision.models.vgg import make_layers
from torchvision.models.vgg import cfgs as vgg_cfgs
from typing import Literal


def get_vgg(vgg_type: Literal["A", "B", "D", "E"], batch_norm=False):
    return make_layers(vgg_cfgs[vgg_type], batch_norm=batch_norm)
