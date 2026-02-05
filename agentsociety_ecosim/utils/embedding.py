from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
import numpy as np
import os
from typing import List

# 为 MCP 服务器设置：如果环境变量 MCP_MODE 存在，强制使用 CPU
if os.getenv('MCP_MODE'):
    device = "cpu"
else:
    device = "cuda" if torch.cuda.is_available() else "cpu"

def embedding(text: str, tokenizer: AutoTokenizer, model: AutoModel) -> np.ndarray:
    """
    Generate an embedding for the given text using the specified tokenizer and model.
    
    Args:
        text (str): The input text to be embedded.
        tokenizer (AutoTokenizer): The tokenizer to preprocess the text.
        model (AutoModel): The model to generate embeddings.
    
    Returns:
        np.ndarray: The normalized embedding vector.
    """
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Mean pooling
    pooled_output = mean_pooling(outputs, inputs['attention_mask']).squeeze(0)

    # Normalize the output
    normalized_embedding = F.normalize(pooled_output, p=2, dim=0)
    
    return normalized_embedding.cpu().numpy().tolist()


def batch_embedding(texts: List[str], tokenizer: AutoTokenizer, model: AutoModel, batch_size: int = 32) -> List[List[float]]:
    """
    批量生成文本的 embedding 向量（加速 5-10 倍）
    
    Args:
        texts: 文本列表
        tokenizer: 分词器
        model: 模型
        batch_size: 批处理大小（默认 32，可根据内存调整）
    
    Returns:
        List[List[float]]: 归一化的 embedding 向量列表
    """
    all_embeddings = []
    
    # 分批处理
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        
        # 批量分词（padding=True 会自动对齐长度）
        inputs = tokenizer(batch_texts, return_tensors='pt', truncation=True, padding=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Mean pooling（处理批量）
        pooled_output = mean_pooling(outputs, inputs['attention_mask'])
        
        # Normalize（处理批量）
        normalized_embeddings = F.normalize(pooled_output, p=2, dim=1)
        
        # 转换为列表
        batch_embeddings = normalized_embeddings.cpu().numpy().tolist()
        all_embeddings.extend(batch_embeddings)
    
    return all_embeddings


# mean pooling
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
