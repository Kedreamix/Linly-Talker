import torch
import torch.nn as nn
import torch.nn.functional as F

from . import blocks, transforms


__all__ = [ 'HeatmapHead' ]


class HeatmapHead(nn.Module):
    """HeatmapHead
    """
    def __init__(self, cfg, **kwargs):
        super(HeatmapHead, self).__init__()
        self.decoder = transforms.__dict__[cfg.HEATMAP.DECODER](
            topk=cfg.HEATMAP.TOPK,
            stride=cfg.HEATMAP.STRIDE,
        )
        self.head = blocks.__dict__[cfg.HEATMAP.BLOCK](
            in_channels=cfg.HEATMAP.IN_CHANNEL,
            proj_channels=cfg.HEATMAP.PROJ_CHANNEL,
            out_channels=cfg.HEATMAP.OUT_CHANNEL,
        )
        
    def forward(self, input):
        return self.decoder(self.head(input))
        
