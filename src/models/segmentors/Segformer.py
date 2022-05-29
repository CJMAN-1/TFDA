from base64 import decode
import torch
import torch.nn as nn
import torch.nn.functional as F
import hydra
import logging
from src.utils import get_logger


class Segformer(nn.Module):
    def __init__(self,
                 backbone,
                 decode_head,
                 pretrained_backbone = None,
                 pretrained_decode_head = None):
        super(Segformer, self).__init__()
        self.backbone = backbone
        self.decode_head = decode_head
        self._init_decode_head(self.decode_head)
        self.init_weights(pretrained_backbone, pretrained_decode_head)

    def _init_decode_head(self, decode_head):
        """Initialize ``decode_head``"""
        self.align_corners = decode_head.align_corners
        self.num_classes = decode_head.num_classes

    def init_weights(self, pre_backbone=None, pre_decode_head=None):
        """Initialize the weights in backbone and heads.
        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        logger = get_logger(__name__)
        if pre_backbone is not None:
            logger.info(f'load backbone model from: {pre_backbone}')
            self.backbone.init_weights(pretrained=pre_backbone)
        if pre_decode_head is not None:
            logger.info(f'load decode_head model from: {pre_decode_head}')
            self.decode_head.init_weights(pretrained=pre_decode_head)

    def _forward_train(self, img):
        self.backbone.train()
        self.decode_head.train()
        output = self.backbone(img)
        output = self.decode_head(output)
        return output
    
    def _forward_infer(self, img):
        self.backbone.eval()
        self.decode_head.eval()
        with torch.no_grad():
            output = self.backbone(img)
            output = self.decode_head(output)
        return output
    
    def _forward_feat(self, img):
        self.backbone.eval()
        self.decode_head.eval()
        with torch.no_grad():
            output = self.backbone(img)
            output, feat = self.decode_head(output, return_feat=True)
        return output, feat

    def forward(self, img, mode=None):
        if mode == 'train' or mode is None:
            return self._forward_train(img)
        elif mode == 'infer':
            return self._forward_infer(img)
        elif mode == 'feat':
            return self._forward_feat(img)
