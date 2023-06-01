import torch
from torch import nn

class PretextRotation(nn.Module):
    """
    Produce a pretext task corresponding to rotation
    Adapted from EASY 
    """
    def forward(self, batch_image):

        bs = batch_image.shape[0] // 4
        target_rot = torch.LongTensor(batch_image.shape[0]).to(batch_image.device)
        target_rot[:bs] = 0
        batch_image[bs:] = batch_image[bs:].transpose(3,2).flip(2)
        target_rot[bs:2*bs] = 1
        batch_image[2*bs:] = batch_image[2*bs:].transpose(3,2).flip(2)
        target_rot[2*bs:3*bs] = 2
        batch_image[3*bs:] = batch_image[3*bs:].transpose(3,2).flip(2)
        target_rot[3*bs:] = 3

        return batch_image, target_rot