"""Resnet backbone adapted from EASY"""
from torch import nn
import torch.nn.functional as F
from torchvision.ops import StochasticDepth


EASY_FEATURE_NUMBER = {"tiny": 32, "small": 45, "classic": 64}


class BasicBlockRN(nn.Module):
    def __init__(
        self,
        in_planes,
        planes,
        number_layer=3,
        stochastic_depth=None,
        mode_stochastic_depth="row",
        dropout=0,
        stride=1,
        slope_leaky=0.01,
        post_activate=True,
    ):
        super(BasicBlockRN, self).__init__()
        list_layer = []

        for i in range(number_layer):
            if i == 0:
                input_fmaps = in_planes
            else:
                input_fmaps = planes
            convolution = nn.Conv2d(
                input_fmaps, planes, kernel_size=3, padding=1, bias=False
            )
            bn = nn.BatchNorm2d(planes)
            list_layer.append(convolution)
            list_layer.append(bn)
            if i != (number_layer - 1):
                list_layer.append(nn.LeakyReLU(negative_slope=slope_leaky))

        use_stochastic_depth = stochastic_depth is not None
        if use_stochastic_depth:
            module_sd = StochasticDepth(stochastic_depth, mode_stochastic_depth)
            self.branch1 = nn.Sequential(module_sd, *list_layer)
        else:
            self.branch1 = nn.Sequential(*list_layer)

        if in_planes != planes or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )
        else:
            self.shortcut = nn.BatchNorm2d(planes)
        self.dropout = dropout

        self.post_activation = (
            nn.LeakyReLU(negative_slope=slope_leaky, inplace=True)
            if post_activate
            else nn.Identity()
        )

    def forward(self, x):
        out = self.branch1(x)
        out += self.shortcut(x)
        if self.dropout > 0:
            out = F.dropout(out, p=self.dropout, training=self.training, inplace=True)
        return self.post_activation(out)


def get_resnet_doubled_from_stage_number(
    number_blocks: list[int], fmaps, use_strides=False, number_layer=2, **kwargs
):
    layers = []
    for i, number_block in enumerate(number_blocks):
        if i == 0:
            in_maps = 3
            out_maps = fmaps
        if i == 1:
            in_maps = fmaps
            out_maps = 2 * fmaps
        if i == 2:
            in_maps = 2 * fmaps
            out_maps = 5 * fmaps
        if i == 3:
            in_maps = fmaps * 5
            out_maps = fmaps * 8
        current_in = in_maps

        add_downsample = i != len(number_blocks) - 1

        for block_id in range(number_block - 1):
            layers.append(
                BasicBlockRN(current_in, out_maps, number_layer=number_layer, **kwargs)
            )
            current_in = out_maps
        if add_downsample:
            if use_strides:
                layers.append(
                    BasicBlockRN(
                        current_in,
                        out_maps,
                        stride=2,
                        number_layer=number_layer,
                        **kwargs
                    )
                )
            else:
                layers.append(
                    BasicBlockRN(
                        current_in, out_maps, number_layer=number_layer, **kwargs
                    )
                )
                layers.append(nn.MaxPool2d((2, 2)))
        else:
            layers.append(
                BasicBlockRN(current_in, out_maps, number_layer=number_layer, **kwargs)
            )

    resnet = nn.Sequential(*layers)
    for m in resnet.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="leaky_relu")
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
    return resnet


def get_resnet_from_stage_number(
    number_blocks: list[int], fmaps, use_strides=False, number_layer=2, **kwargs
):
    layers = []
    for i, number_block in enumerate(number_blocks):
        if i == 0:
            in_maps = 3
            out_maps = fmaps
        if i == 1:
            in_maps = fmaps
            out_maps = int(fmaps * 2.5)
        if i == 2:
            in_maps = int(fmaps * 2.5)
            out_maps = fmaps * 5
        if i == 3:
            in_maps = fmaps * 5
            out_maps = fmaps * 10
        current_in = in_maps

        add_downsample = i != len(number_blocks) - 1

        for block_id in range(number_block - 1):
            layers.append(
                BasicBlockRN(current_in, out_maps, number_layer=number_layer, **kwargs)
            )
            current_in = out_maps
        if add_downsample:
            if use_strides:
                layers.append(
                    BasicBlockRN(
                        current_in,
                        out_maps,
                        stride=2,
                        number_layer=number_layer,
                        **kwargs
                    )
                )
            else:
                layers.append(
                    BasicBlockRN(
                        current_in, out_maps, number_layer=number_layer, **kwargs
                    )
                )
                layers.append(nn.MaxPool2d((2, 2)))
        else:
            layers.append(
                BasicBlockRN(current_in, out_maps, number_layer=number_layer, **kwargs)
            )

    resnet = nn.Sequential(*layers)
    for m in resnet.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="leaky_relu")
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
    return resnet


def get_resnet12(feature_maps, **kwargs):
    layers = []
    layers.append(BasicBlockRN(3, feature_maps, **kwargs))
    layers.append(nn.MaxPool2d((2, 2)))
    layers.append(BasicBlockRN(feature_maps, int(2.5 * feature_maps), **kwargs))
    layers.append(nn.MaxPool2d((2, 2)))
    layers.append(BasicBlockRN(int(2.5 * feature_maps), 5 * feature_maps, **kwargs))
    layers.append(nn.MaxPool2d((2, 2)))
    layers.append(BasicBlockRN(5 * feature_maps, 10 * feature_maps, **kwargs))
    resnet = nn.Sequential(*layers)
    for m in resnet.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="leaky_relu")
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
    return resnet


def get_resnet12_easy(type, **kwargs):
    return get_resnet12(EASY_FEATURE_NUMBER[type], **kwargs)


def get_resnet9(feature_maps, **kwargs):
    layers = []
    layers.append(BasicBlockRN(3, feature_maps, **kwargs))
    layers.append(nn.MaxPool2d((2, 2)))
    layers.append(BasicBlockRN(feature_maps, int(2.5 * feature_maps), **kwargs))
    layers.append(nn.MaxPool2d((2, 2)))
    layers.append(BasicBlockRN(int(2.5 * feature_maps), 5 * feature_maps, **kwargs))
    resnet = nn.Sequential(*layers)
    for m in resnet.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="leaky_relu")
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
    return resnet


def get_resnet9_easy(type, **kwargs):
    return get_resnet9(EASY_FEATURE_NUMBER[type], **kwargs)
