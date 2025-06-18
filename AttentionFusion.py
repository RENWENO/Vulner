#! -*- coding: utf-8 -*-
# @Time    : 2025/6/18 23:20
# @Author  : xx
import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionFusion(nn.Module):
    def __init__(self, in_channels):
        super(AttentionFusion, self).__init__()
        self.attn = nn.Sequential(
            nn.Linear(in_channels, in_channels // 2),
            nn.ReLU(),
            nn.Linear(in_channels // 2, in_channels)
        )

    def forward(self, x1, x2):
        # 计算注意力权重
        attn_1 = self.attn(x1)  # [batch_size, in_channels]
        attn_2 = self.attn(x2)  # [batch_size, in_channels]

        # 归一化
        attn_1 = F.softmax(attn_1, dim=0)
        attn_2 = F.softmax(attn_2, dim=0)

        # 加权融合
        fused = attn_1 * x1 + attn_2 * x2
        return fused

