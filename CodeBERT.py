#! -*- coding: utf-8 -*-
# @Time    : 2025/5/4 10:10
# @Author  : xx
import torch
from transformers import AutoModel, AutoTokenizer
from torch import nn
from torch.utils.data import Dataset, DataLoader


# 1. 长代码处理模块
def split_code(code, chunk_size=512, stride=256):
    """
    使用滑动窗口拆分长代码
    """
    tokens = tokenizer.tokenize(code)
    chunks = []

    start = 0
    while start < len(tokens):
        end = start + chunk_size
        chunk = tokens[start:end]
        chunks.append(tokenizer.convert_tokens_to_string(chunk))

        if end >= len(tokens):
            break
        start += stride

    return chunks


# 2. 数据集类
class ContractDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_chunks=20):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_chunks = max_chunks

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        code = self.texts[idx]
        chunks = split_code(code)[:self.max_chunks]  # 截取最大chunk数

        # 编码每个chunk
        chunk_features = []
        for chunk in chunks:
            inputs = self.tokenizer(
                chunk,
                return_tensors="pt",
                max_length=512,
                padding="max_length",
                truncation=True
            )
            chunk_features.append({
                "input_ids": inputs["input_ids"].squeeze(0),
                "attention_mask": inputs["attention_mask"].squeeze(0)
            })

        # 填充不足的chunks
        while len(chunk_features) < self.max_chunks:
            chunk_features.append({
                "input_ids": torch.zeros(512, dtype=torch.long),
                "attention_mask": torch.zeros(512, dtype=torch.long)
            })

        return {
            "chunks": chunk_features,
            "label": torch.tensor(self.labels[idx], dtype=torch.long)
        }


# 3. 模型架构
class CodeBERTWithFusion(nn.Module):
    def __init__(self, num_labels=5):
        super().__init__()
        self.codebert = AutoModel.from_pretrained("microsoft/codebert-base")
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=768, nhead=8),
            num_layers=2
        )
        self.classifier = nn.Linear(768, num_labels)

    def forward(self, input_chunks):
        # 处理每个chunk
        chunk_embeddings = []
        for chunk in input_chunks:
            outputs = self.codebert(
                input_ids=chunk["input_ids"].unsqueeze(0),
                attention_mask=chunk["attention_mask"].unsqueeze(0)
            )
            # 取CLS token作为chunk表示
            cls_embedding = outputs.last_hidden_state[0, 0, :]
            chunk_embeddings.append(cls_embedding)

        # 拼接所有chunk特征 [seq_len, batch, features]
        chunk_tensor = torch.stack(chunk_embeddings).unsqueeze(1)

        # Transformer特征融合
        fused_features = self.transformer(chunk_tensor)

        # 取融合后的CLS特征进行分类
        pooled_output = fused_features.mean(dim=0)
        logits = self.classifier(pooled_output)
        return logits


# 4. 训练准备
tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
model = CodeBERTWithFusion(num_labels=5)

# 示例数据
with open("./DoS/train/sol_have_bug/0x0000000000b3f879cb30fe243b4dfee438691c04.sol",'r') as f1:
    text1 = f1.read()
with open("./DoS/train/sol_no_bug/0x0000000000027f6d87be8ade118d9ee56767d993.sol",'r') as f2:
    text2 = f2.read()
train_texts = [text1, text2]  # 你的合约代码列表
train_labels = [1, 0]  # 对应的漏洞标签

dataset = ContractDataset(train_texts, train_labels, tokenizer)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# 5. 训练循环
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
criterion = nn.CrossEntropyLoss()

for epoch in range(5):
    for batch in dataloader:
        optimizer.zero_grad()

        chunks = batch["chunks"]
        labels = batch["label"]
        aa = chunks[0]
        # 重组输入结构
        inputs = []
        for i in range(len(chunks[0])):  # 遍历每个chunk位置
            chunk_batch = {
                "input_ids": torch.stack([c[i]["input_ids"] for c in chunks]),
                "attention_mask": torch.stack([c[i]["attention_mask"] for c in chunks])
            }
            inputs.append(chunk_batch)

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        print(f"Loss: {loss.item()}")