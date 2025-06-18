#! -*- coding: utf-8 -*-
# @Time    : 2025/6/14 15:16
# @Author  : xx
import pickle
import dgl
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
import torch
from pathlib import Path
import random
import numpy as np
from Vulner.model import HAN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def edgefenli(edges,nodes):
    BF=[]
    AF=[]
    for edge in edges:
        BF.append(int(edge[0]))
        AF.append(int(edge[1]))
    for n in nodes:
        BF.append(int(n))
        AF.append(int(n))
    return BF,AF

def readgraph(StringPath):
    # 读取图结构文件并将其节点（nodes）和边（edges）输出
    with open(StringPath, 'rb') as f:
        graph_data = pickle.load(f)
    nodes = graph_data.nodes
    edges = graph_data.edges
    return nodes,edges

def gener_hetero_graph(nodes,edges1,edges2,lable=0):
    #输出单个文件的异质图结构和节点特征。
    zjBF, zjAF = edgefenli(edges1,nodes)
    jjBF, jjAF = edgefenli(edges2,nodes)
    dict1 = {
        ('node', 'zj', 'node'): (torch.tensor(zjBF), torch.tensor(zjAF)),
        ('node', 'jj', 'node'): (torch.tensor(jjBF), torch.tensor(jjAF))
    }
    hetero_graph = dgl.heterograph(dict1)
    allgraph1Node = []
    #print(hetero_graph)
    tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
    for node in nodes:
        try:
            code = nodes[node]['attr']['label']
            # tokens = tokenizer.tokenize(xulietezheng)
            inputs = tokenizer.encode_plus(
                code,  # 代码片段（必选）
                add_special_tokens=True,  # 自动添加 <s>, </s> 等标记
                max_length=100,  # 最大序列长度（超过则截断）
                padding="max_length",  # 填充到 max_length
                truncation=True,  # 启用截断
                return_tensors="pt",  # 返回 PyTorch 张量（可选 "tf" 为 TensorFlow）
            )
        except Exception as e:
            print("errr")
            print(e)
            code ="Node Type:uint uninitializedUint;"
            inputs = tokenizer.encode_plus(
                code,  # 代码片段（必选）
                add_special_tokens=True,  # 自动添加 <s>, </s> 等标记
                max_length=100,  # 最大序列长度（超过则截断）
                padding="max_length",  # 填充到 max_length
                truncation=True,  # 启用截断
                return_tensors="pt",  # 返回 PyTorch 张量（可选 "tf" 为 TensorFlow）
            )

        allgraph1Node.append(inputs)

    data = [hetero_graph,allgraph1Node,lable]
    return data
def get_filename_set(folder):
    """获取文件夹中所有文件名集合（包含子目录）"""
    return {f.name for f in Path(folder).rglob('*') if f.is_file()}


def split_data(data):
    # 复制列表避免修改原始数据
    data_copy = data.copy()
    # 随机打乱数据
    random.shuffle(data_copy)

    # 计算划分点（90%训练集，10%测试集）
    split_index = int(0.9 * len(data_copy))

    # 划分训练集和测试集
    train_set = data_copy[:split_index]
    test_set = data_copy[split_index:]

    return train_set, test_set
class VulnerDataset(Dataset):
    def __init__(self,data):
        self.data = data
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        heterograph = data[0]
        feature = data[1]
        lable = data[2]
        return heterograph, feature,lable
def BEACH(graph):
    dantu = []
    for i in range(len(graph)):
        g = graph[i]
        print(g)
        dantu.append(g)
    BBhg = dgl.batch(dantu)
    print(BBhg)
    return BBhg

def Cat(feart):
    codebertmodel = AutoModel.from_pretrained("microsoft/codebert-base").to(device)
    allids = []
    alltoken = []
    allatt = []
    for encoded_input in feart:
        print("节点个数：", len(encoded_input))
        for en in encoded_input:

            allids.append(en['input_ids'])
            allatt.append(en['attention_mask'])

        #tokenerlist.append(encoded_input)

    id = torch.cat(allids, dim=0)
    att = torch.cat(allatt, dim=0)

    input = {}
    input['input_ids'] = id.to(device)
    input['attention_mask'] = att.to(device)

    codebertmodel.eval()
    with torch.no_grad():
        outputs = codebertmodel(**input)
    last_hidden_states = outputs.last_hidden_state
    sencefeat = last_hidden_states[:, 0, :]
    #return sencefeat
    #f = torch.tensor(input['input_ids'], dtype=torch.float32).clone().detach().requires_grad_(True)
    return sencefeat
def collate_fn(batch):
    hetero_graph,faeture,lable = map(list, zip(*batch))
    #graph,faeture,lable = GuiYi(graphs, features, labels)
    #print(faeture.shape)


    batched_graph = BEACH(hetero_graph)
    #print(batched_graph.number_of_nodes())
    batched_features = Cat(faeture)
    #print(batched_features.shape)
    lable1=[]
    for i in lable:
        #combined_array = np.concatenate(i, axis=0)

        encoded_labels = np.eye(2)[i]
        torch_tensor = torch.from_numpy(encoded_labels)
        tensor_unsqueezed = torch_tensor.unsqueeze(0)
        lable1.append(tensor_unsqueezed)

    batched_labels = torch.cat(lable1,dim=0)

    return batched_graph, batched_features, batched_labels
def score(logits, labels): #  micro_f1 和 macro_f1

    prediction = torch.gt(logits,0).float()
    labels =labels

    #print(prediction)
    #print(labels)

    fz =2 * torch.sum(labels * prediction).float()
    fm =torch.sum(labels + prediction).float()
    #print(fz)
    #print(fm)
    f1= fz / fm
    return f1, f1, f1


# 评估
def evaluate(model, g, features, labels, mask, loss_func):
    model.eval()
    with torch.no_grad():
        logits = model(g, features)
    loss = loss_func(logits[mask], labels[mask])
    accuracy, micro_f1, macro_f1 = score(logits[mask], labels[mask])

    return loss, accuracy, micro_f1, macro_f1


def multilabel_categorical_crossentropy(y_true, y_pred):
    y_pred = (1 - 2 * y_true) * y_pred
    y_pred_neg = y_pred - y_true * 1e-12
    y_pred_pos = y_pred - (1 - y_true) * 1e-12  # 这里应该使用.clone()方法
    y_pred_pos = y_pred_pos.clone()  # 添加这行代码

    zeros = torch.zeros_like(y_pred[..., :1])
    y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
    y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
    neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
    pos_loss = torch.logsumexp(y_pred_pos, dim=-1)

    loss = (neg_loss + pos_loss).mean()
    Loss = loss.requires_grad_(True)

    return Loss

if __name__ == "__main__":
    mate1Path = "/home/Vulner/CFG/Pickce_DOS1"
    mate2Path = "/home/Vulner/CFG/Pickce_DOS"
    mate3Path = "/home/Vulner/CFG/Pickce_nobug1"
    mate4Path = "/home/Vulner/CFG/Pickce_nobug"
    matefile = get_filename_set(mate1Path)
    mateNobugfile = get_filename_set(mate3Path)
    allDATA=[]
    i=0
    for file in matefile:
        i=i+1
        M1P = mate1Path+'/'+file
        M2P = mate2Path+'/'+file
        print(M1P)
        #print(M2P)
        nodes,egeds1 = readgraph(M1P)
        _,egeds2 = readgraph(M2P)
        # print(nodes)
        # print(_)
        data = gener_hetero_graph(nodes,egeds1,egeds2,lable=1)
        #print(data[0])
        allDATA.append(data)
        if i == 40:
            break
    print("漏洞类结束")
    j=0
    for file1 in mateNobugfile:
        j=j+1
        try:
            M3P = mate3Path+'/'+file1
            M4P = mate4Path+'/'+file1
            print(M3P)
            #print(M2P)
            nodes,egeds3 = readgraph(M3P)
            _,egeds4 = readgraph(M4P)
            # print(nodes)
            # print(_)
            data = gener_hetero_graph(nodes,egeds3,egeds4,lable=0)
            #print(data[0])
            allDATA.append(data)
        except Exception as e:
            print("errr")
            print(e)
            continue
        if j==40:
            break
    print("负类结束")
    print(len(allDATA))
    train,test =split_data(allDATA)
    #print(train)
    traindataset = VulnerDataset(train)
    traindataload = DataLoader(traindataset, batch_size=8, collate_fn=collate_fn,drop_last=True)
    testdataset = VulnerDataset(test)
    testdataload = DataLoader(testdataset, batch_size=8, collate_fn=collate_fn, drop_last=True)
    model = HAN(meta_paths=[["jj"], ["zj"]],
                in_size=768,
                hidden_size=384,
                out_size=768,
                num_heads=[4],
                dropout=0.5).to(device)
    loss_fcn = torch.nn.CrossEntropyLoss()
    loss_BCfcn = torch.nn.BCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.1)
    for epch in range(460):
        model.train()
        allacc = []
        f1mi = []
        f1ma = []
        allloss = []
        alllogits = []
        for step,(G,F,L) in enumerate(traindataload):
            print(step)
            print(G)
            print(F.shape)
            print(L)
            batched_graph1 = G.to(device)
            batched_labels1 = L.type(torch.float32).to(device)
            batched_features1 = F.to(device)
            optimizer.zero_grad()
            logits = model(batched_graph1, batched_features1)
            print(logits)
            loss = multilabel_categorical_crossentropy(batched_labels1, logits)
            allloss.append(loss)
            accuracy, micro_f1, macro_f1 = score(logits, batched_labels1)
            allacc.append(accuracy)
            f1ma.append(macro_f1)
            f1mi.append(micro_f1)
            loss.backward()
            optimizer.step()
        print(step)
        loss1 = sum(allloss) / j
        acc = sum(allacc) / j
        f1ma = sum(f1ma) / j
        f1mi = sum(f1mi) / j
        print("---------train-loss-------", loss1.item(), "-----train-F1-----", f1ma.item(), "-----次数-----", epch)

        model.eval()
        with torch.no_grad():
            allacc1 = []
            f1mi1 = []
            f1ma1 = []
            allloss1 = []
            alllogits1 = []
            for step, (G, F, L) in enumerate(traindataload):
                print(step)
                print(G)
                print(F.shape)
                print(L)
                batched_graph1 = G.to(device)
                batched_labels1 = L.type(torch.float32).to(device)
                batched_features1 = F.to(device)
                optimizer.zero_grad()
                logits = model(batched_graph1, batched_features1)
                print(logits)
                loss = multilabel_categorical_crossentropy(batched_labels1, logits)
                allloss.append(loss)
                accuracy, micro_f1, macro_f1 = score(logits, batched_labels1)
                allacc.append(accuracy)
                f1ma.append(macro_f1)
                f1mi.append(micro_f1)
                loss.backward()
                optimizer.step()

