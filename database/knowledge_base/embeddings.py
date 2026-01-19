"""
SPECTER2 Embedding 模型封装

SPECTER2 是 Allen AI 专门为学术论文设计的 Embedding 模型，
支持多种任务适配器（proximity, adhoc_query, classification）。
"""

import os
import logging
import threading
from typing import List, Optional, Union
from dataclasses import dataclass
import numpy as np

from .config import EmbeddingConfig

logger = logging.getLogger(__name__)

# 设置 HuggingFace 镜像（中国大陆访问）
HF_MIRROR = os.getenv("HF_ENDPOINT", "https://hf-mirror.com")
os.environ["HF_ENDPOINT"] = HF_MIRROR

# 本地缓存优先路径（ModelScope 下载）
LOCAL_MODEL_PATHS = {
    "sentence-transformers/allenai-specter": "/root/.cache/modelscope/sentence-transformers/allenai-specter",
    "allenai/specter2": "/root/.cache/modelscope/sentence-transformers/allenai-specter",
}


@dataclass
class EmbeddingResult:
    """Embedding 结果"""
    embeddings: np.ndarray
    texts: List[str]
    model_name: str


class SPECTER2Embeddings:
    """
    SPECTER2 学术论文 Embedding 模型
    
    支持的适配器:
    - allenai/specter2: 基础模型
    - allenai/specter2_proximity: 论文相似度检索（推荐用于 RAG）
    - allenai/specter2_adhoc_query: 查询-文档匹配
    - allenai/specter2_classification: 分类任务
    
    使用示例:
        embeddings = SPECTER2Embeddings()
        vectors = embeddings.embed_documents(["paper title. paper abstract"])
        query_vector = embeddings.embed_query("research topic")
    """
    
    def __init__(self, config: Optional[EmbeddingConfig] = None):
        """
        初始化 SPECTER2 Embedding 模型
        
        Args:
            config: Embedding 配置，如果为 None 则使用默认配置
        """
        self.config = config or EmbeddingConfig()
        self._model = None
        self._tokenizer = None
        self._adapter_loaded = False
        self._load_lock = threading.Lock()  # 线程锁，防止并发加载
        
        # 延迟加载模型
        logger.info(f"SPECTER2Embeddings initialized (lazy loading)")
    
    def _load_model(self) -> None:
        """加载模型（延迟加载）- 使用 sentence-transformers，支持并发安全"""
        # 快速路径：如果已加载，直接返回
        if self._model is not None:
            return
        
        # 使用锁确保只有一个线程/进程加载模型
        with self._load_lock:
            # 双重检查：可能在等待锁时其他线程已完成加载
            if self._model is not None:
                return
            
            logger.info(f"Loading SPECTER2 model: {self.config.model_name}")
            
            try:
                from sentence_transformers import SentenceTransformer
                import torch
                
                # 确定设备（sentence-transformers 不接受 "auto"）
                if self.config.device == "auto":
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                else:
                    device = self.config.device
                
                # 使用 sentence-transformers 加载模型
                # 注意：allenai/specter2 需要特殊处理，使用 SciBERT 作为替代
                model_mapping = {
                    "allenai/specter2": "sentence-transformers/allenai-specter",  # 原始 SPECTER
                    "allenai/specter2_base": "sentence-transformers/allenai-specter",
                    "allenai/specter2_proximity": "sentence-transformers/allenai-specter",
                }
                
                model_name = model_mapping.get(self.config.model_name, self.config.model_name)
                
                # 如果配置了单独的 HF 端点，优先应用（否则沿用全局 HF_ENDPOINT）
                if getattr(self.config, "hf_endpoint", None):
                    os.environ["HF_ENDPOINT"] = self.config.hf_endpoint  # type: ignore
                
                # 优先使用本地 ModelScope 缓存
                local_path = LOCAL_MODEL_PATHS.get(model_name)
                load_path = local_path if local_path and os.path.exists(local_path) else model_name
                if local_path and os.path.exists(local_path):
                    logger.info(f"Using local cached model: {local_path}")
                else:
                    logger.info(f"Using remote/mirror model: {model_name}")
                
                logger.info(f"Using model: {load_path} on device: {device}")
                
                # 修复 meta tensor 问题：先在 CPU 加载，再移动到目标设备
                if device == "cuda" and torch.cuda.is_available():
                    # 先在 CPU 加载
                    self._model = SentenceTransformer(load_path, device="cpu")
                    # 然后移动到 CUDA
                    self._model = self._model.to(device)
                else:
                    self._model = SentenceTransformer(load_path, device=device)
                
                logger.info(f"Model loaded successfully on {device}")
                
            except Exception as e:
                logger.error(f"Failed to load SPECTER2 model: {e}")
                raise
    
    def embed_documents(
        self,
        texts: List[str],
        batch_size: Optional[int] = None,
        show_progress: bool = True,
    ) -> np.ndarray:
        """
        对文档列表进行 Embedding
        
        Args:
            texts: 文档文本列表
            batch_size: 批处理大小，默认使用配置值
            show_progress: 是否显示进度条
        
        Returns:
            Embedding 向量矩阵，形状为 (n_docs, embedding_dim)
        """
        self._load_model()
        
        batch_size = batch_size or self.config.batch_size
        
        return self._model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=True,
        )
    
    def embed_query(self, query: str) -> np.ndarray:
        """
        对单个查询进行 Embedding
        
        Args:
            query: 查询文本
        
        Returns:
            Embedding 向量，形状为 (embedding_dim,)
        """
        embeddings = self.embed_documents([query], show_progress=False)
        return embeddings[0]
    
    def embed_paper(
        self,
        title: str,
        abstract: str = "",
    ) -> np.ndarray:
        """
        对论文进行 Embedding（使用标准格式）
        
        SPECTER2 推荐的论文格式: "title [SEP] abstract"
        
        Args:
            title: 论文标题
            abstract: 论文摘要
        
        Returns:
            Embedding 向量
        """
        # SPECTER2 推荐格式
        text = f"{title} [SEP] {abstract}" if abstract else title
        return self.embed_query(text)
    
    @property
    def embedding_dimension(self) -> int:
        """返回 Embedding 维度"""
        return 768  # SPECTER2 固定输出 768 维
    
    def __repr__(self) -> str:
        return (
            f"SPECTER2Embeddings("
            f"model={self.config.model_name}, "
            f"adapter={self.config.adapter_name}, "
            f"device={self.config.device})"
        )


class FallbackEmbeddings:
    """
    回退 Embedding 模型（当 SPECTER2 不可用时使用）
    
    使用 sentence-transformers 的通用模型作为备选。
    """
    
    # ModelScope 缓存路径映射
    LOCAL_MODEL_PATHS = {
        "all-MiniLM-L6-v2": "/root/.cache/modelscope/sentence-transformers/all-MiniLM-L6-v2",
        "sentence-transformers/all-MiniLM-L6-v2": "/root/.cache/modelscope/sentence-transformers/all-MiniLM-L6-v2",
    }
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        初始化回退模型
        
        Args:
            model_name: sentence-transformers 模型名称或本地路径
        """
        self.model_name = model_name
        self._model = None
        logger.info(f"FallbackEmbeddings initialized with {model_name}")
    
    def _load_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            import os
            
            # 优先使用本地缓存路径
            local_path = self.LOCAL_MODEL_PATHS.get(self.model_name)
            if local_path and os.path.exists(local_path):
                logger.info(f"Loading from local cache: {local_path}")
                self._model = SentenceTransformer(local_path)
                return
            
            # 尝试从 HuggingFace 加载
            try:
                self._model = SentenceTransformer(self.model_name)
            except Exception as e:
                logger.warning(f"Failed to load model: {e}")
                raise
    
    def embed_documents(
        self,
        texts: List[str],
        show_progress: bool = True,
    ) -> np.ndarray:
        """对文档列表进行 Embedding"""
        self._load_model()
        return self._model.encode(
            texts,
            show_progress_bar=show_progress,
            normalize_embeddings=True,
        )
    
    def embed_query(self, query: str) -> np.ndarray:
        """对查询进行 Embedding"""
        self._load_model()
        return self._model.encode(
            query,
            normalize_embeddings=True,
        )
    
    @property
    def embedding_dimension(self) -> int:
        self._load_model()
        return self._model.get_sentence_embedding_dimension()


def get_embeddings(config: Optional[EmbeddingConfig] = None) -> Union[SPECTER2Embeddings, FallbackEmbeddings]:
    """
    获取 Embedding 模型（带自动回退）
    
    Args:
        config: Embedding 配置
    
    Returns:
        Embedding 模型实例
    """
    config = config or EmbeddingConfig()
    
    # 如果配置了使用回退模型，直接使用 FallbackEmbeddings
    if config.use_fallback:
        model_name = config.model_name if "specter" not in config.model_name.lower() else "all-MiniLM-L6-v2"
        logger.info(f"Using fallback embeddings: {model_name}")
        return FallbackEmbeddings(model_name)
    
    try:
        return SPECTER2Embeddings(config)
    except Exception as e:
        logger.warning(f"Failed to load SPECTER2, falling back to MiniLM: {e}")
        return FallbackEmbeddings()
 