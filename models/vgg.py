from torch import nn
from typing import Dict, List, Union, cast
from torchvision.models.vgg import make_layers
from torchvision.models.vgg import cfgs as vgg_cfgs
from models.backbone import SequentialManifoldMix
from torch.nn import Sequential

def make_layers_manifold(
    cfg: List[Union[str, int]], batch_norm: bool = False
) -> SequentialManifoldMix:
    """
    same function as make_layers, expept that it uses SequentialManifoldMix instead of nn.Sequential
    """
    layers: List[nn.Module] = []
    in_channels = 3
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            v = cast(int, v)
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return SequentialManifoldMix(*layers)

# TODO : functional to avoid all model definition
VGG_MODELS = dict(
    vgg11 = make_layers(vgg_cfgs["A"]),
    vgg11bn = make_layers(vgg_cfgs["A"], batch_norm=True),
    vgg13 = make_layers(vgg_cfgs["B"]),
    vgg13bn = make_layers(vgg_cfgs["B"], batch_norm=True),
    vgg16 = make_layers(vgg_cfgs["D"]),
    vgg16bn = make_layers(vgg_cfgs["D"], batch_norm=True),
    vgg19 = make_layers(vgg_cfgs["E"]),
    vgg19_bn = make_layers(vgg_cfgs["E"], batch_norm=True),
)

