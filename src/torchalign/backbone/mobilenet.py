import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo


__all__ = ['MobileNetV2', 'mobilenetv2']


class Block(nn.Module):
    """ 
    Bottleneck Residual Block
    
    """
    def __init__(self, in_channels, out_channels, expansion=1, stride=1):
        super(Block, self).__init__()
        if expansion == 1:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, 3, stride, 1, groups=in_channels, bias=False),
                nn.BatchNorm2d(in_channels),
                nn.ReLU6(inplace=True),
                nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            channels = expansion * in_channels
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, channels, 1, 1, 0, bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU6(inplace=True),
                nn.Conv2d(channels, channels, 3, stride, 1, groups=channels, bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU6(inplace=True),
                nn.Conv2d(channels, out_channels, 1, 1, 0, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        self.residual = (stride == 1) and (in_channels == out_channels)

    def forward(self, x):
        out = self.conv(x)
        if self.residual:
            out = out + x
        return out


class MobileNetV2(nn.Module):
    def __init__(self, config):
        super(MobileNetV2, self).__init__()
        in_channels = config[0][1]
        features = [nn.Sequential(
            nn.Conv2d(3, in_channels, 3, 2, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU6(inplace=True)
        )]
        for expansion, out_channels, blocks, stride in config[1:]:
            for i in range(blocks):
                features.append(Block(in_channels, out_channels, expansion, stride if i == 0 else 1))
                in_channels = out_channels
        self.features = nn.Sequential(*features)

    def forward(self, x):
        c2 = self.features[:4](x)
        c3 = self.features[4:7](c2)
        c4 = self.features[7:14](c3)
        kwargs = {'size': c2.shape[-2:],'mode': 'bilinear','align_corners': False}
        return torch.cat([F.interpolate(xx,**kwargs) for xx in [c2,c3,c4]], 1)


def mobilenetv2(pretrained=False, **kwargs):
    """Constructs a MobileNetv2 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    config = [
        (1,  32, 1, 1),
        (1,  16, 1, 1),
        (6,  24, 2, 2), 
        (6,  32, 3, 2),
        (6,  64, 4, 2),
        (6,  96, 3, 1),
    ]
    model = MobileNetV2(config, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['mobilenetv2']), strict=False)
    return model



