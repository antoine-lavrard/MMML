"""Resnet backbone adapted from EASY"""
from torch import nn
import torch.nn.functional as F


EASY_FEATURE_NUMBER = {
    "tiny" : 32,
    "small" : 45,
    "classic" : 64
}

class BasicBlockRN12(nn.Module):
    def __init__(self, in_planes, planes, dropout = 0):
        
        super(BasicBlockRN12, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential(
            nn.Conv2d(in_planes, planes, kernel_size=1, bias=False),
            nn.BatchNorm2d(planes)
        )
        self.dropout = dropout

    def forward(self, x):
        out = F.leaky_relu(self.bn1(self.conv1(x)), negative_slope = 0.1)
        out = F.leaky_relu(self.bn2(self.conv2(out)), negative_slope = 0.1)
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        if self.dropout > 0:
            out = F.dropout(out, p=self.dropout, training=self.training, inplace=True)
        return out
    
def get_resnet12(feature_maps, dropout):
    layers = []
    layers.append(BasicBlockRN12(3, feature_maps, dropout))
    layers.append(BasicBlockRN12(feature_maps, int(2.5 * feature_maps), dropout))
    layers.append(BasicBlockRN12(int(2.5 * feature_maps), 5 * feature_maps,dropout))
    layers.append(BasicBlockRN12(5 * feature_maps, 10 * feature_maps, dropout))   
    resnet = nn.Sequential(*layers)
    for m in resnet.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
    return resnet

def get_resnet12_easy(type, dropout = 0):
    return get_resnet12(EASY_FEATURE_NUMBER[type], dropout = dropout)