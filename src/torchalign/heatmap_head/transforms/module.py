
import torch
import torch.nn as nn

from . import functional as F


__all__ = [ 'BinaryHeatmap2Coordinate' ]


class BinaryHeatmap2Coordinate(nn.Module):
    """BinaryHeatmap2Coordinate
    """
    def __init__(self, stride=4.0, topk=5, **kwargs):
        super(BinaryHeatmap2Coordinate, self).__init__()
        self.topk = topk
        self.stride = stride
        
    def forward(self, input):
        return self.stride * F.heatmap2coord(input[:,1,...], self.topk)
        
    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'topk={}, '.format(self.topk)
        format_string += 'stride={}'.format(self.stride)
        format_string += ')'
        return format_string
        
