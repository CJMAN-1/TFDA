# ---------------------------------------------------------------
# Copyright (c) 2021, NVIDIA Corporation. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# ---------------------------------------------------------------
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F


class MLP(nn.Module):
    """
    Linear Embedding
    """
    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2).contiguous()
        x = self.proj(x)
        return x


class SegFormerHead(nn.Module):
    """
    SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
    """
    def __init__(self, in_channels, in_index, feature_strides, dropout_ratio, num_classes, norm, align_corners, decoder_params):
        super(SegFormerHead, self).__init__()
        assert len(feature_strides) == len(in_channels)
        assert min(feature_strides) == feature_strides[0]
        self.input_transform='multiple_select'
        self.in_index = in_index
        self.feature_strides = feature_strides
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.align_corners = align_corners


        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels

        decoder_params = decoder_params
        embedding_dim = decoder_params['embed_dim']

        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=embedding_dim)

        self.linear_fuse = nn.Conv2d(in_channels=embedding_dim*4,
                                    out_channels=embedding_dim,
                                    kernel_size=1)
        if norm == 'SyncBN':
            self.norm = nn.SyncBatchNorm(num_features=embedding_dim)
        elif norm == 'BN':
            self.norm = nn.BatchNorm2d(num_features=embedding_dim)
            
        if dropout_ratio > 0:
            self.dropout = nn.Dropout2d(dropout_ratio)
        else:
            self.dropout = None
        self.linear_pred = nn.Conv2d(embedding_dim, self.num_classes, kernel_size=1)

    def forward(self, inputs, return_feat=False):
        x = self._transform_inputs(inputs)  # len=4, 1/4,1/8,1/16,1/32
        c1, c2, c3, c4 = x

        ############## MLP decoder on C1-C4 ###########
        n, _, h, w = c4.shape

        _c4 = self.linear_c4(c4).permute(0,2,1).reshape(n, -1, c4.shape[2], c4.shape[3]).contiguous()
        _c4 = F.interpolate(_c4, size=c1.size()[2:],mode='bilinear',align_corners=self.align_corners)

        _c3 = self.linear_c3(c3).permute(0,2,1).reshape(n, -1, c3.shape[2], c3.shape[3]).contiguous()
        _c3 = F.interpolate(_c3, size=c1.size()[2:],mode='bilinear',align_corners=self.align_corners)

        _c2 = self.linear_c2(c2).permute(0,2,1).reshape(n, -1, c2.shape[2], c2.shape[3]).contiguous()
        _c2 = F.interpolate(_c2, size=c1.size()[2:],mode='bilinear',align_corners=self.align_corners)

        _c1 = self.linear_c1(c1).permute(0,2,1).reshape(n, -1, c1.shape[2], c1.shape[3]).contiguous()

        _c_cat = torch.cat([_c4, _c3, _c2, _c1], dim=1)

        _c = self.linear_fuse(_c_cat)
        _c = self.norm(_c)

        x = self.dropout(_c)
        x = self.linear_pred(x)
        if return_feat:
            return x, _c_cat
        else:
            return x

    def _transform_inputs(self, inputs):
        """Transform inputs for decoder.

        Args:
            inputs (list[Tensor]): List of multi-level img features.

        Returns:
            Tensor: The transformed inputs
        """

        if self.input_transform == 'resize_concat':
            inputs = [inputs[i] for i in self.in_index]
            upsampled_inputs = [
                F.interpolate(
                    input=x,
                    size=inputs[0].shape[2:],
                    mode='bilinear',
                    align_corners=self.align_corners) for x in inputs
            ]
            inputs = torch.cat(upsampled_inputs, dim=1)
        elif self.input_transform == 'multiple_select':
            inputs = [inputs[i] for i in self.in_index]
        else:
            inputs = inputs[self.in_index]

        return inputs

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            self.load_state_dict(torch.load(pretrained, map_location='cpu'))