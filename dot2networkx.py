#! -*- coding: utf-8 -*-
# @Time    : 2025/5/29 23:26
# @Author  : xx

import os
import networkx as nx
from pathlib import Path
import pygraphviz as pgv
import pickle

def get_filename_set(folder):
    """获取文件夹中所有文件名集合（包含子目录）"""
    return {f.name for f in Path(folder).rglob('*') if f.is_file()}

def get_file_set(folder):
    """获取文件夹中所有文件名集合"""
    filedir = os.listdir(folder)
    return filedir

#修改路径
all = get_file_set('./CFG/DoS_nobug_CFG')

for j in all:
    #修改路径
    testdir = './CFG/DoS_nobug_CFG/'+j

    alldot = get_filename_set(testdir)
    for i,ad in enumerate(alldot):
        dotstring = testdir+'/'+ad
        try:
            graph = pgv.AGraph(dotstring)
        except Exception as e:
            continue
        if i == 0:
            G1 = nx.nx_agraph.from_agraph(graph)
        else:
            G = nx.nx_agraph.from_agraph(graph)
            G1 = nx.disjoint_union(G1, G)
    print(G1)
    #修改路径
    outstring ='./CFG/Pickce_nobug/'+j+'.pickle'
    with open(outstring, "wb") as f:
        pickle.dump(G1, f, protocol=pickle.HIGHEST_PROTOCOL)


