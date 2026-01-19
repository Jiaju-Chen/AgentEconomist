from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
import numpy as np
import os

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

# mean pooling
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
