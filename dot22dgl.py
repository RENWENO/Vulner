#! -*- coding: utf-8 -*-
# @Time    : 2025/6/13 11:02
# @Author  : xx
#生成节点直接的间接连接图路径2
import os
from pathlib import Path
import pydotplus as pydot  # 使用 pydotplus 替代
#import dgl
import torch
import networkx as nx
import pickle
from collections import defaultdict
import pickle


def find_transitive_relations(relations):
    """
    找出关系列表中的传递关系

    参数:
        relations: 关系列表，如 [(0,1), (1,2), ...]

    返回:
        transitive_relations: 所有传递关系的集合
        components: 所有连通分量（链式结构）的列表
    """
    # 构建图结构（邻接表表示）
    graph = defaultdict(list)
    for a, b in relations:
        graph[a].append(b)

    # 找出所有链式结构（连通分量）
    components = []
    visited = set()

    for node in list(graph.keys()):
        if node not in visited:
            # 从当前节点开始DFS遍历
            stack = [node]
            component = []

            while stack:
                current = stack.pop()
                if current not in visited:
                    visited.add(current)
                    component.append(current)
                    # 添加所有未访问的邻居
                    for neighbor in graph.get(current, []):
                        if neighbor not in visited:
                            stack.append(neighbor)

            # 只保留长度大于1的链
            if len(component) > 1:
                components.append(component)

    # 找出所有传递关系
    transitive_relations = set()

    for chain in components:
        # 对每个链式结构，找出所有可能的传递关系
        n = len(chain)
        for i in range(n):
            for j in range(i + 2, n):  # 至少间隔一个节点
                # 添加传递关系 (chain[i], chain[j])
                transitive_relations.add((chain[i], chain[j]))

    return transitive_relations, components
def getnetworkx(edges,nodes):

    nx_graph = nx.Graph()
    for node in nodes:
        # print(node)
        # print(nodes[node][0]['attributes']['label'])
        #att1 = nodes[node]
        att =  nodes[node]['attr']
        nx_graph.add_node(node,attr = att)
    for edge in edges:
        # print(edge[0])
        # print(edge[1])
        source = edge[0]
        destination = edge[1]
        nx_graph.add_edge(source,destination)
    return nx_graph
pickle_path ="/home/Vulner/CFG/Pickce_nobug/0x00000000e86b5156e8fd624255bf7a6d722a8f1f.pickle"
def readgraph(pickle_path):
    with open(pickle_path, 'rb') as f:
        graph_data = pickle.load(f)
        edgess = graph_data.edges
        listedges = []
        for edgeslist in edgess:
            listedges.append(edgeslist)
        # print(edgess)
        # print(graph_data)
        nodes = graph_data.nodes
        #print(nodes[0])

        chuandiedges,chaotu = find_transitive_relations(listedges)
        G = getnetworkx(chuandiedges,nodes)
        print(G)
        return G

def get_filename_set(folder):
    """获取文件夹中所有文件名集合（包含子目录）"""
    return {f.name for f in Path(folder).rglob('*') if f.is_file()}

#allfile = get_filename_set("/home/Vulner/CFG/Pickce_nobug")
#print(allfile)

if __name__ == '__main__':
    inputpath = "/home/Vulner/CFG/Pickce_nobug"
    outputpath = "/home/Vulner/CFG/Pickce_nobug1/"
    filename = get_filename_set(inputpath)
    for name in filename:
        graphPath = inputpath+'/'+name
        G = readgraph(graphPath)
        print(G)
        outstring = outputpath + name
        with open(outstring, "wb") as f:
            pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)