#! -*- coding: utf-8 -*-
# @Time    : 2025/5/11 23:34
# @Author  : xx
import os

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
import math
import torch.nn as nn
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len = 30):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        return x + self.pe[:x.size(0)]


class AttentionPooling(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Tanh(),
            nn.Linear(d_model, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, d_model]
        Returns:
            pooled: Tensor, shape [batch_size, d_model]
        """
        # 计算注意力分数
        attn_scores = self.attention(x)  # [seq_len, batch_size, 1]
        attn_weights = torch.softmax(attn_scores, dim=0)  # 沿序列维度归一化

        # 加权求和
        pooled = torch.sum(x * attn_weights, dim=0)  # [batch_size, d_model]
        return pooled

class CodeProcessor:
    def __init__(self, model_name="microsoft/codebert-base"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_chunk_len = 510  # 为特殊token预留空间

    def chunk_code(self, code: str) -> list:
        """动态分块策略"""
        tokens = self.tokenizer.tokenize(code)
        if not tokens:
            return [self._create_empty_chunk()]

        # 自动计算窗口参数
        chunk_size = min(len(tokens), self.max_chunk_len)
        stride = max(chunk_size // 2, 1)  # 确保至少1个token的步长

        chunks = []
        for i in range(0, len(tokens), stride):
            chunk = tokens[i:i + chunk_size]
            processed = self._process_chunk(chunk)
            chunks.append(processed)
            if len(chunks) >= 30:  # 安全限制
                break
        return chunks

    def _process_chunk(self, tokens: list) -> dict:
        """处理单个代码块"""
        # 添加特殊token并转换为ID
        input_ids = self.tokenizer.build_inputs_with_special_tokens(
            self.tokenizer.convert_tokens_to_ids(tokens))

        # 截断和填充处理
        if len(input_ids) > 512:
            input_ids = input_ids[:510] + [self.tokenizer.sep_token_id]
        pad_len = 512 - len(input_ids)
        attention_mask = [1] * len(input_ids) + [0] * pad_len
        input_ids += [self.tokenizer.pad_token_id] * pad_len

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long)
        }

    def _create_empty_chunk(self) -> dict:
        """生成空分块"""
        return {
            "input_ids": torch.zeros(512, dtype=torch.long),
            "attention_mask": torch.zeros(512, dtype=torch.long)
        }


class SmartContractDataset(Dataset):
    def __init__(self, file_paths, labels, processor):
        self.samples = []
        self.processor = processor

        for path, label in zip(file_paths, labels):
            with open(path, 'r') as f:
                code = f.read()
                chunks = self.processor.chunk_code(code)
                self.samples.append({
                    "chunks": chunks,
                    "label": label
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def dynamic_collate(batch):
    """动态批次对齐函数"""
    # 计算当前批次最大分块数
    chunk_counts = [len(item["chunks"]) for item in batch]
    batch_max = max(chunk_counts)

    # 初始化批次容器
    processed = {
        "input_ids": [],
        "attention_mask": [],
        "chunk_masks": [],
        "labels": torch.stack([torch.tensor(item["label"]) for item in batch])
    }

    # 处理每个分块位置
    for chunk_idx in range(batch_max):
        chunk_inputs = []
        chunk_attn = []
        chunk_mask = []

        for sample in batch:
            # 获取当前分块或填充空分块
            if chunk_idx < len(sample["chunks"]):
                chunk = sample["chunks"][chunk_idx]
                mask = 1.0
            else:
                chunk = processor._create_empty_chunk()
                mask = 0.0

            chunk_inputs.append(chunk["input_ids"])
            chunk_attn.append(chunk["attention_mask"])
            chunk_mask.append(mask)

        # 转换为张量
        processed["input_ids"].append(torch.stack(chunk_inputs))
        processed["attention_mask"].append(torch.stack(chunk_attn))
        processed["chunk_masks"].append(torch.tensor(chunk_mask))

    return processed


class ChunkFusionModel(nn.Module):
    def __init__(self, num_labels=2):
        super().__init__()
        self.encoder = AutoModel.from_pretrained("microsoft/codebert-base")
        self.position_embed = PositionalEncoding(768)
        self.transformer = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(
                d_model=768,
                nhead=4,
                dim_feedforward=768*2
            ),
            num_layers=2
        )
        self.pooling = AttentionPooling(d_model=768)
        self.classifier = torch.nn.Linear(768, num_labels)
        self.device = device
        # 冻结底层参数
        for layer in self.encoder.encoder.layer:
            for param in layer.parameters():
                param.requires_grad = False

    def forward(self, input_ids_list, attention_mask_list, chunk_masks):
        # batch_size = input_ids_list[0].size(0)
        # device = device

        # 处理每个分块
        chunk_features = []
        for chunk_idx, (input_ids, attn_mask) in enumerate(zip(input_ids_list, attention_mask_list)):
            outputs = self.encoder(
                input_ids=input_ids.to(self.device),
                attention_mask=attn_mask.to(self.device)
            )
            cls_features = outputs.last_hidden_state[:, 0, :]  # [batch, 768]
            mask = chunk_masks[chunk_idx].unsqueeze(1).to(self.device)
            chunk_features.append(cls_features * mask)

        # 添加位置编码并融合
        features = torch.stack(chunk_features)  # [seq, batch, 768]
        print(features.shape)
        features = self.position_embed(features)
        fused = self.transformer(features)
        pooled = self.pooling(fused)

        return self.classifier(pooled)

file_path1 =[]
labels1=[]
file_path2 =[]
labels2=[]
file_names1 = os.listdir("./DoS/train/sol_have_bug/")
for file_name in file_names1:
    file_path = os.path.join("./DoS/train/sol_have_bug/", file_name)
    if os.path.isfile(file_path):  # 确保是文件（非子目录）
        #print("文件名:", file_name)
        #print("完整路径:", file_path)
        file_path1.append(file_path)
        labels1.append(1)

file_names2 = os.listdir("./DoS/train/sol_no_bug/")
for file_name in file_names2:
    file_path = os.path.join("./DoS/train/sol_no_bug/", file_name)
    if os.path.isfile(file_path):  # 确保是文件（非子目录）
        #print("文件名:", file_name)
        #print("完整路径:", file_path)
        file_path2.append(file_path)
        labels2.append(0)
labels = labels1+labels2
files = file_path1+file_path2
# 使用示例
processor = CodeProcessor()
dataset = SmartContractDataset(
    file_paths=files,
    labels=labels,
    processor=processor
)

dataloader = DataLoader(
    dataset,
    batch_size=2,
    collate_fn=lambda b: dynamic_collate(b),
    shuffle=True,
    pin_memory=True
)

model = ChunkFusionModel().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

# 训练循环
for epoch in range(5):
    ALLloss = []
    i=1
    for batch in dataloader:
        i=i+1
        # 数据迁移到GPU
        inputs = [chunk.to(device) for chunk in batch["input_ids"]]
        #inputs = inputs.to(device)
        masks = [chunk.to(device) for chunk in batch["attention_mask"]]
        #masks = masks.to(device)
        chunk_masks = [mask.to(device) for mask in batch["chunk_masks"]]
        #chunk_masks = chunk_masks.to(device)
        labels = batch["labels"].to(device)
        one_hot = torch.nn.functional.one_hot(labels, num_classes=2).float()
        #one_hot = one_hot.to(device)

        # 前向传播
        outputs = model(inputs, masks, chunk_masks)
        loss = torch.nn.CrossEntropyLoss()(outputs, one_hot)
        #print(loss)
        ALLloss.append(loss)
        # 反向传播
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    AL = sum(ALLloss)
    print("loss is :"+AL/i)
