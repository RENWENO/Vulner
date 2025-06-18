#! -*- coding: utf-8 -*-
# @Time    : 2025/5/3 11:56
# @Author  : xx

import torch
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        # 分词并编码
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'label': torch.tensor(label)
        }


# 自定义批处理函数
def collate_fn(batch):
    input_ids = [item['input_ids'] for item in batch]
    attention_mask = [item['attention_mask'] for item in batch]
    labels = torch.stack([item['label'] for item in batch])

    # 填充到批次内最大长度
    padded_input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True)
    padded_attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True)

    return {
        'input_ids': padded_input_ids,
        'attention_mask': padded_attention_mask,
        'labels': labels
    }


# 使用示例
from transformers import AutoTokenizer

texts = ["Hello world!", "PyTorch is awesome", "Deep learning"]
labels = [0, 1, 0]
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

dataset = TextDataset(texts, labels, tokenizer)
dataloader = DataLoader(
    dataset,
    batch_size=2,
    collate_fn=collate_fn,
    shuffle=True
)

for batch in dataloader:
    print(f"Input IDs shape: {batch['input_ids'].shape}")
    print(f"Attention mask shape: {batch['attention_mask'].shape}")
    print(f"Labels: {batch['labels']}")
    break