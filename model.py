#! -*- coding: utf-8 -*-
# @Time    : 2025/6/13 16:21
# @Author  : xx

#! -*- coding: utf-8 -*-
# @Time    : 2024/7/5 23:44
# @Author  : LiuGan
"""This model shows an example of using dgl.metapath_reachable_graph on the original heterogeneous
graph.
Because the original HAN implementation only gives the preprocessed homogeneous graph, this model
could not reproduce the result in HAN as they did not provide the preprocessing code, and we
constructed another dataset from ACM with a different set of papers, connections, features and
labels.
"""

import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GATConv
import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel
from Vulner.LNFfusion import LayerNormDotProductAttentionFusion
from Vulner.AttentionFusion import AttentionFusion

#from HAN.utils import load_data, EarlyStopping
# from HAN.utils  import setup
from dgl.nn import GlobalAttentionPooling
# from dgl.nn import Set2Set
# from dgl.nn import WeightAndSum
# from dgl.nn import SumPooling

class GraphAttentionPooling(nn.Module):
    def __init__(self, embedding_dim):
        super(GraphAttentionPooling, self).__init__()
        self.linear = nn.Linear(embedding_dim, 1)  # 用于计算注意力分数的线性层

    def forward(self, node_embeddings):
        # node_embeddings 是形状为 [num_nodes, embedding_dim] 的矩阵
        # 计算每个节点的注意力分数
        attention_scores = self.linear(node_embeddings).squeeze(1)
        attention_weights = F.softmax(attention_scores, dim=0)
        # 加权求和以聚合节点嵌入
        graph_embedding = (node_embeddings * attention_weights.unsqueeze(1)).sum(dim=0)
        return graph_embedding
class SemanticAttention(nn.Module):
    def __init__(self, in_size, hidden_size=128):
        super(SemanticAttention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False),
        )

    def forward(self, z):
        w = self.project(z).mean(0)  # (M, 1)
        beta = torch.softmax(w, dim=0)  # (M, 1)
        beta = beta.expand((z.shape[0],) + beta.shape)  # (N, M, 1)

        return (beta * z).sum(1)  # (N, D * K)


class HANLayer(nn.Module):
    """
    HAN layer.

    Arguments
    ---------
    meta_paths : list of metapaths, each as a list of edge types
    in_size : input feature dimension
    out_size : output feature dimension
    layer_num_heads : number of attention heads
    dropout : Dropout probability

    Inputs
    ------
    g : DGLGraph
        The heterogeneous graph（异构网络）
    h : tensor
        Input features

    Outputs
    -------
    tensor
        The output feature
    """

    def __init__(self, meta_paths, in_size, out_size, layer_num_heads, dropout):
        super(HANLayer, self).__init__()

        # One GAT layer for each meta path based adjacency matrix
        self.gat_layers = nn.ModuleList()
        for i in range(len(meta_paths)):
            self.gat_layers.append(
                GATConv(
                    in_size,
                    out_size,
                    layer_num_heads,
                    dropout,
                    dropout,
                    activation=F.elu,
                    allow_zero_in_degree=True,
                )
            )
        self.semantic_attention = SemanticAttention(
            in_size=out_size * layer_num_heads
        )
        self.meta_paths = list(tuple(meta_path) for meta_path in meta_paths)

        self._cached_graph = None
        self._cached_coalesced_graph = {}#字典用来存储图

    def forward(self, g, h):
        semantic_embeddings = []

        if self._cached_graph is None or self._cached_graph is not g:
            self._cached_graph = g
            self._cached_coalesced_graph.clear()#字典清除
            for meta_path in self.meta_paths:
                self._cached_coalesced_graph[
                    meta_path
                ] = dgl.metapath_reachable_graph(g, meta_path)

        for i, meta_path in enumerate(self.meta_paths):
            new_g = self._cached_coalesced_graph[meta_path]
            semantic_embeddings.append(self.gat_layers[i](new_g, h).flatten(1))
        semantic_embeddings = torch.stack(
            semantic_embeddings, dim=1
        )  # (N, M, D * K)

        return self.semantic_attention(semantic_embeddings)  # (N, D * K)


class HAN(nn.Module):
    def __init__(
        self, meta_paths, in_size, hidden_size, out_size, num_heads, dropout
    ):
        super(HAN, self).__init__()
        #self.bert = BertModel.from_pretrained('../BERT')
        self.layers = nn.ModuleList()
        self.layers.append(
            HANLayer(meta_paths, in_size, hidden_size, num_heads[0], dropout)
        )
        for l in range(1, len(num_heads)):
            self.layers.append(
                HANLayer(
                    meta_paths,
                    hidden_size * num_heads[l - 1],
                    hidden_size,
                    num_heads[l],
                    dropout,
                )
            )
        #self.predict = nn.Linear(hidden_size * num_heads[-1], out_size)
        #self.predict = nn.Linear(768, out_size)
        #self.pool = GraphAttentionPooling(out_size)

        self.liner = nn.Linear(hidden_size * num_heads[-1], 1, bias=True)
        self.pool = GlobalAttentionPooling(self.liner)
        #self.pool1 = Set2Set(hidden_size * num_heads[-1],2,1)
        #self.pool2 = WeightAndSum(hidden_size * num_heads[-1])
        #self.pool3 = SumPooling()
        self.liner1 = nn.Linear(768,out_size,bias=True)
        self.liner2 = nn.Linear(out_size,350 ,bias=True)
        self.liner3 = nn.Linear(768,hidden_size * num_heads[-1])
        self.liner4 = nn.Linear(hidden_size * num_heads[-1],768)
        self.yce = nn.Linear(350,2,bias=True)
        self.bn1 = nn.BatchNorm1d(num_features=hidden_size * num_heads[-1])
        self.bn2 = nn.BatchNorm1d(num_features=350)
        self.bn3 = nn.BatchNorm1d(num_features=768)
        self.fusion = LayerNormDotProductAttentionFusion(input_dim=768,hidden_size=hidden_size, num_heads=num_heads)
        self.attention = AttentionFusion(768)
        self.dropout = nn.Dropout(0.6)

    def forward(self, g, h):
        h2 = h
        #outputs = self.bert(**h)
        #last_hidden_states = outputs.last_hidden_state
        #h = last_hidden_states[:, 0, :]
        h3 = self.liner3(h)
        for gnn in self.layers:
            h1 = gnn(g, h)
            h=h3+h1
            h = self.bn1(h)
            h = F.elu(h)
        Tp = self.pool(g,h)
        T = self.liner4(Tp)#768
        T = self.dropout(T)
        newT = self.fusion(h2,h,g)#768
        newT = self.bn3(newT)
        RH = self.attention(T,newT)
        RH = self.dropout(RH)
        Trh = self.liner2(RH)#350
        Tbn = self.bn2(Trh)
        Tbn = self.dropout(Tbn)
        Tel = F.elu(Tbn)
        yc = self.yce(Tel)
        return yc
if __name__ == '__main__':
    #from HAN.model_hetero import HAN
    model = HAN(meta_paths=[["uu"], ["uf"],["uaf"],["hyper"]],
                in_size=768,
                hidden_size=384,
                out_size=720,
                num_heads=[8, 8],
                dropout=0.8)
    print("ssh -p 24351 root@i-2.gpushare.com")