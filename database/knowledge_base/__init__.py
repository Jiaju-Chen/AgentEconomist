"""
Academic Paper Knowledge Base
=============================

基于 SPECTER2 + Qdrant 的学术论文知识库系统。

主要组件:
- config: 配置管理
- embeddings: SPECTER2 Embedding 模型封装
- document_loader: 论文 JSON 文档加载器
- vector_store: Qdrant 向量存储
- retriever: 语义检索器
- indexer: 索引构建器
- tool: Agent 工具封装

使用示例:
    from knowledge_base import KnowledgeBaseConfig, PaperRetriever
    
    config = KnowledgeBaseConfig()
    retriever = PaperRetriever(config)
    results = retriever.search("urban air pollution health impact", top_k=5)
"""

from .config import KnowledgeBaseConfig
from .embeddings import SPECTER2Embeddings
from .document_loader import PaperDocument, PaperDocumentLoader
from .vector_store import QdrantVectorStore
from .retriever import PaperRetriever
from .indexer import PaperIndexer

__version__ = "1.0.0"
__all__ = [
    "KnowledgeBaseConfig",
    "SPECTER2Embeddings",
    "PaperDocument",
    "PaperDocumentLoader",
    "QdrantVectorStore",
    "PaperRetriever",
    "PaperIndexer",
]


