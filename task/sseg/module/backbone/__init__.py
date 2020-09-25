"""
This file is adapted from https://github.com/jfzhang95/pytorch-deeplab-xception
"""

from . import resnet


def build_backbone(backbone, output_stride, BatchNorm, pretrained_backbone_url):
    if backbone == 'resnet50':
        return resnet.ResNet50(output_stride, BatchNorm, pretrained_backbone_url)
    elif backbone == 'resnet101':
        return resnet.ResNet101(output_stride, BatchNorm, pretrained_backbone_url)
    else:
        raise NotImplementedError
