#! -*- coding: utf-8 -*-
# @Time    : 2025/5/1 10:50
# @Author  : xx
# Load model directly
from transformers import AutoTokenizer, AutoModel
import torch

# 加载模型和分词器
tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
model = AutoModel.from_pretrained("microsoft/codebert-base")
def chunk_code(code, tokenizer, chunk_size=500, overlap=50):
    tokens = tokenizer.tokenize(code)
    chunks = []
    start = 0
    while start < len(tokens):
        end = start + chunk_size
        chunks.append(tokenizer.convert_tokens_to_string(tokens[start:end]))
        start += (chunk_size - overlap)
    return chunks


# 长Java代码示例
java_code = """
public class Calculator {
    private int result;

    public Calculator() {
        result = 0;
    }

    public void add(int x) {
        result += x;
    }

    public void subtract(int x) {
        result -= x;
    }

    public int getResult() {
        return result;
    }
}
"""

# 分块处理
chunks = chunk_code(java_code, tokenizer, chunk_size=100, overlap=20)

# 提取各块特征并聚合
chunk_embeddings = []
for chunk in chunks:
    inputs = tokenizer(chunk, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    chunk_embeddings.append(outputs.last_hidden_state[:, 0, :])  # 取各块的[CLS]

final_embedding = torch.mean(torch.stack(chunk_embeddings), dim=0)  # 均值聚合
print(f"聚合后向量维度: {final_embedding.shape}")
#print(final_embedding)