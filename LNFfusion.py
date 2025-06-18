#! -*- coding: utf-8 -*-
# @Time    : 2025/6/18 23:19
# @Author  : xx
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import GlobalAttentionPooling


class DotProductAttention(nn.Module):
    def __init__(self, in_channels):
        super(DotProductAttention, self).__init__()
        self.attn = nn.Sequential(
            nn.Linear(in_channels, in_channels // 2),
            nn.ReLU(),
            nn.Linear(in_channels // 2, in_channels)
        )

    def forward(self, x1, x2):
        # 计算注意力权重
        # 计算注意力权重
        attn_1 = self.attn(x1)  # [batch_size, in_channels]
        attn_2 = self.attn(x2)  # [batch_size, in_channels]

        # 归一化
        attn_1 = F.softmax(attn_1, dim=0)
        attn_2 = F.softmax(attn_2, dim=0)

        # 加权融合
        fused = attn_1 * x1 + attn_2 * x2
        return fused




class LayerNormDotProductAttentionFusion(nn.Module):
    def __init__(self, input_dim,hidden_size, num_heads):
        super(LayerNormDotProductAttentionFusion, self).__init__()
        # 定义点积注意力层
        self.attention_layer = DotProductAttention(768)
        # 定义层归一化层
        self.layer_norm = nn.LayerNorm((input_dim,), elementwise_affine=True)
        self.liner3 = nn.Linear(hidden_size * num_heads[-1],768)
        self.liner = nn.Linear(768, 1, bias=True)
        self.pool = GlobalAttentionPooling(self.liner)

    def forward(self, x, y,g):
        # 对x和y执行层归一化
        y=self.liner3(y)
        y = self.attention_layer(y, x)
        # 将上下文向量和原始输入相加来融合特征
        fusedxy = self.layer_norm(y)
        inputs = self.pool(g,fusedxy)
        return inputs