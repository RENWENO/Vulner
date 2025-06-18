#! -*- coding: utf-8 -*-
# @Time    : 2025/6/11 0:11
# @Author  : xx
#生成节点直接连接图路径1
import os
from pathlib import Path
import pydotplus as pydot  # 使用 pydotplus 替代
#import dgl
import torch
import networkx as nx
import pickle

def get_filename_set(folder):
    """获取文件夹中所有文件名集合（包含子目录）"""
    return {f.name for f in Path(folder).rglob('*') if f.is_file()}

def get_file_set(folder):
    """获取文件夹中所有文件名集合"""
    filedir = os.listdir(folder)
    return filedir

def get_networkx(PathString):
    # 读取 .dot 文件
    graphs = pydot.graph_from_dot_file(PathString)
    nodes = graphs.obj_dict["nodes"]
    edges = graphs.obj_dict["edges"]

    # print(nodes['0'])
    # print(nodes['1'])
    # print(edges)
    nx_graph = nx.Graph()
    #print(len(nodes))
    for node in nodes:
        # print(node)
        # print(nodes[node][0]['attributes']['label'])
        att =  nodes[node][0]['attributes']
        nx_graph.add_node(node,attr = att)
    for edge in edges:
        # print(edge[0])
        # print(edge[1])
        source = edge[0]
        destination = edge[1]
        nx_graph.add_edge(source,destination)
    return nx_graph

# String = "./CFG/DoS_CFG/0x000983ba1a675327f0940b56c2d49cd9c042dfbf/0x000983ba1a675327f0940b56c2d49cd9c042dfbf.sol-CappedVault-constructor().dot"
# netG = get_networkx(String)
# print(netG.nodes["1"])

#修改路径


all = get_file_set('./CFG/DoS_CFG')

for j in all:
    #修改路径
    testdir = './CFG/DoS_CFG/'+j

    alldot = get_filename_set(testdir)
    for i,ad in enumerate(alldot):
        dotstring = testdir+'/'+ad
        try:
            if i == 0:
                G1 = get_networkx(dotstring)
            else:
                G = get_networkx(dotstring)
                G1 = nx.disjoint_union(G1, G)
        except Exception as e:
            continue
    print(G1)
    #修改路径
    outstring ='./CFG/Pickce_DOS/'+j+'.pickle'
    with open(outstring, "wb") as f:
        pickle.dump(G1, f, protocol=pickle.HIGHEST_PROTOCOL)
