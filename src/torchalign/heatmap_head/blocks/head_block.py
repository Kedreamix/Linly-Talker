import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = [ 'BinaryHeadBlock' ]


class BinaryHeadBlock(nn.Module):
    """BinaryHeadBlock
    """
    def __init__(self, in_channels, proj_channels, out_channels, **kwargs):
        super(BinaryHeadBlock, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, proj_channels, 1, bias=False),
            nn.BatchNorm2d(proj_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(proj_channels, out_channels*2, 1, bias=False),
        )
        
    def forward(self, input):
        N, C, H, W = input.shape
        return self.layers(input).view(N, 2, -1, H, W)

