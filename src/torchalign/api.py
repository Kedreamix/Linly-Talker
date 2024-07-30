import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import transforms

from . import backbone, heatmap_head


__all__ = [ 'FacialLandmarkDetector' ]


class FacialLandmarkDetector(nn.Module):
    """FacialLandmarkDetector
    """
    def __init__(self, root, pretrained=True):
        super(FacialLandmarkDetector, self).__init__()
        self.config = self.config_from_file(os.path.join(root, 'config.yaml'))
        self.backbone = backbone.__dict__[self.config.BACKBONE.ARCH](pretrained=False)
        self.heatmap_head = heatmap_head.__dict__[self.config.HEATMAP.ARCH](self.config)
        self.transform = transforms.Compose([
            transforms.Resize(self.config.INPUT.SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        ])
        if pretrained:
            self.load_state_dict(torch.load(os.path.join(root, 'model.pth')))
        
    def config_from_file(self, filename):
        from .cfg import cfg
        if os.path.isfile(filename):
            cfg.merge_from_file(filename)
        return cfg
        
    def resized_crop(self, img, bbox):
        rect = torch.Tensor([[0, 0, img.width, img.height]])
        if bbox is not None:
            wh = (bbox[:,2:] - bbox[:,:2] + 1).max(1)[0] * self.config.INPUT.SCALE
            xy = (bbox[:,:2] + bbox[:,2:] - wh.unsqueeze(1) + 1) / 2.0
            rect = torch.cat([xy, xy+wh.unsqueeze(1)], 1)
        data = torch.stack([self.transform(img.crop(x.tolist())) for x in rect])
        return data, rect
        
    def resized_crop_inverse(self, landmark, rect):
        scale = torch.stack([
            self.config.INPUT.SIZE[0] / (rect[:,2]-rect[:,0]),
            self.config.INPUT.SIZE[1] / (rect[:,3]-rect[:,1])
        ]).t()
        return landmark / scale[:,None,:] + rect[:,None,:2]
        
    def flip_landmark(self, landmark, img_width):
        landmark[..., 0] = img_width - 1 -landmark[...,0]
        return landmark[...,self.config.INPUT.FLIP_ORDER,:]

    def forward(self, img, bbox=None, device=None):
        data, rect = self.resized_crop(img, bbox)
        if device is not None:
            data, rect = data.to(device), rect.to(device)
        landmark = self.heatmap_head(self.backbone(data))
        if self.config.INPUT.FLIP:
            data = data.flip(dims=[-1])
            landmark_ = self.heatmap_head(self.backbone(data))
            landmark_ = self.flip_landmark(landmark_, data.shape[-1])
            landmark = (landmark + landmark_) / 2.0
        landmark = self.resized_crop_inverse(landmark, rect)
        return landmark
