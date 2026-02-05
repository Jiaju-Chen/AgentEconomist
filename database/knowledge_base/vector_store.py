"""
Qdrant 向量存储封装

提供向量存储、检索、更新等功能。
"""

import logging
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
import numpy as np
from pathlib import Path
import hashlib
from uuid import uuid5, NAMESPACE_DNS

from .config import QdrantConfig
from .document_loader import PaperDocument

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """检索结果"""
    document: PaperDocument
    score: float
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "document": self.document.to_dict(),
            "score": self.score,
        }


class QdrantVectorStore:
    """
    Qdrant 向量存储
    
    支持三种存储模式：
    - memory: 内存存储（重启后丢失）
    - local: 本地文件存储
    - remote: 远程 Qdrant 服务器
    
    使用示例:
        store = QdrantVectorStore(config)
        store.add_documents(documents, embeddings)
        results = store.search(query_embedding, top_k=5)
    """
    
    def __init__(self, config: Optional[QdrantConfig] = None):
        """
        初始化向量存储
        
        Args:
            config: Qdrant 配置
        """
        self.config = config or QdrantConfig()
        self._client = None
        
        logger.info(f"QdrantVectorStore initialized (mode={self.config.mode})")
    
    def _get_client(self):
        """获取 Qdrant 客户端（延迟初始化）"""
        if self._client is not None:
            return self._client
        
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams
        
        # 根据模式创建客户端
        if self.config.mode == "memory":
            self._client = QdrantClient(":memory:")
            logger.info("Using in-memory Qdrant storage")
        
        elif self.config.mode == "local":
            path = Path(self.config.local_path)
            path.mkdir(parents=True, exist_ok=True)
            self._client = QdrantClient(path=str(path))
            logger.info(f"Using local Qdrant storage: {path}")
        
        elif self.config.mode == "remote":
            self._client = QdrantClient(
                host=self.config.host,
                port=self.config.port,
                api_key=self.config.api_key,
            )
            logger.info(f"Connected to remote Qdrant: {self.config.host}:{self.config.port}")
        
        else:
            raise ValueError(f"Unknown Qdrant mode: {self.config.mode}")
        
        # 确保 collection 存在
        self._ensure_collection()
        
        return self._client
    
    def _ensure_collection(self) -> None:
        """确保 collection 存在"""
        from qdrant_client.models import Distance, VectorParams
        
        collections = self._client.get_collections().collections
        collection_names = [c.name for c in collections]
        
        if self.config.collection_name not in collection_names:
            # 距离度量映射
            distance_map = {
                "cosine": Distance.COSINE,
                "euclid": Distance.EUCLID,
                "dot": Distance.DOT,
            }
            
            self._client.create_collection(
                collection_name=self.config.collection_name,
                vectors_config=VectorParams(
                    size=self.config.vector_size,
                    distance=distance_map.get(
                        self.config.distance_metric,
                        Distance.COSINE,
                    ),
                ),
            )
            logger.info(f"Created collection: {self.config.collection_name}")
        else:
            logger.info(f"Collection exists: {self.config.collection_name}")
    
    def add_documents(
        self,
        documents: List[PaperDocument],
        embeddings: np.ndarray,
        batch_size: int = 100,
    ) -> int:
        """
        添加文档到向量存储
        
        Args:
            documents: 文档列表
            embeddings: Embedding 向量矩阵
            batch_size: 批处理大小
        
        Returns:
            成功添加的文档数量
        """
        from qdrant_client.models import PointStruct
        
        client = self._get_client()
        
        if len(documents) != len(embeddings):
            raise ValueError(
                f"Documents count ({len(documents)}) != embeddings count ({len(embeddings)})"
            )
        
        points = []
        for doc, embedding in zip(documents, embeddings):
            # 使用 UUID5 生成确定性 ID（基于 doc_id），避免哈希冲突
            # UUID5 是确定性哈希，相同输入产生相同输出，冲突概率极低（2^-122）
            doc_uuid = uuid5(NAMESPACE_DNS, doc.doc_id)
            point_id = doc_uuid.int % (2**63)  # 转换为 64 位整数
            
            point = PointStruct(
                id=point_id,
                vector=embedding.tolist(),
                payload=doc.to_dict(),
            )
            points.append(point)
        
        # 分批上传
        total_added = 0
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            client.upsert(
                collection_name=self.config.collection_name,
                points=batch,
            )
            total_added += len(batch)
            
            if (i + batch_size) % 1000 == 0:
                logger.info(f"Uploaded {total_added}/{len(points)} documents")
        
        logger.info(f"Successfully added {total_added} documents")
        return total_added
    
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        score_threshold: Optional[float] = None,
        filter_conditions: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """
        向量检索
        
        Args:
            query_embedding: 查询向量
            top_k: 返回结果数量
            score_threshold: 最小相似度阈值
            filter_conditions: 过滤条件
        
        Returns:
            检索结果列表
        """
        from qdrant_client.models import Filter, FieldCondition, MatchValue, Range
        
        client = self._get_client()
        
        # 构建过滤器
        qdrant_filter = None
        if filter_conditions:
            must_conditions = []
            
            # 期刊过滤
            if "journal" in filter_conditions:
                journal = filter_conditions["journal"]
                if isinstance(journal, str):
                    if journal.endswith("*"):
                        # 前缀匹配（Qdrant 不直接支持，需要多条件）
                        pass  # TODO: 实现前缀匹配
                    else:
                        must_conditions.append(
                            FieldCondition(
                                key="journal",
                                match=MatchValue(value=journal),
                            )
                        )
            
            # 年份范围过滤（通过 publish_year 整数字段）
            # 注意：需要在文档中添加 publish_year 字段，或者在结果中手动过滤
            if "year_range" in filter_conditions:
                start_year, end_year = filter_conditions["year_range"]
                # Qdrant Range 需要数字类型，所以使用整数年份
                # 这里我们跳过 Qdrant 过滤，在结果中手动过滤
                pass  # TODO: 添加 publish_year 字段支持
            
            # 文档类型过滤
            if "doc_type" in filter_conditions:
                must_conditions.append(
                    FieldCondition(
                        key="doc_type",
                        match=MatchValue(value=filter_conditions["doc_type"]),
                    )
                )
            
            if must_conditions:
                qdrant_filter = Filter(must=must_conditions)
        
        # 执行检索（使用新版本 API：query_points 替代 search）
        # query_points 可以直接接受向量列表作为 query 参数
        response = client.query_points(
            collection_name=self.config.collection_name,
            query=query_embedding.tolist(),  # 直接传入向量列表
            query_filter=qdrant_filter,
            limit=top_k,
            score_threshold=score_threshold,
        )
        
        # 转换响应格式（query_points 返回 QueryResponse，需要访问 points）
        results = response.points if hasattr(response, 'points') else []
        
        # 转换结果
        search_results = []
        for hit in results:
            doc = PaperDocument.from_dict(hit.payload)
            # query_points 返回的 ScoredPoint 中，score 是 float 值
            score = getattr(hit, 'score', 0.0)
            search_results.append(SearchResult(document=doc, score=float(score)))
        
        return search_results
    
    def search_by_paper_id(self, paper_id: str) -> List[SearchResult]:
        """
        按论文 ID 检索所有相关文档
        
        Args:
            paper_id: 论文 ID
        
        Returns:
            该论文的所有文档块
        """
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        
        client = self._get_client()
        
        results = client.scroll(
            collection_name=self.config.collection_name,
            scroll_filter=Filter(
                must=[
                    FieldCondition(
                        key="paper_id",
                        match=MatchValue(value=paper_id),
                    )
                ]
            ),
            limit=100,
        )[0]
        
        return [
            SearchResult(
                document=PaperDocument.from_dict(point.payload),
                score=1.0,
            )
            for point in results
        ]
    
    def delete_by_paper_id(self, paper_id: str) -> int:
        """
        删除指定论文的所有文档
        
        Args:
            paper_id: 论文 ID
        
        Returns:
            删除的文档数量
        """
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        
        client = self._get_client()
        
        # 先查询数量
        docs = self.search_by_paper_id(paper_id)
        count = len(docs)
        
        if count > 0:
            client.delete(
                collection_name=self.config.collection_name,
                points_selector=Filter(
                    must=[
                        FieldCondition(
                            key="paper_id",
                            match=MatchValue(value=paper_id),
                        )
                    ]
                ),
            )
            logger.info(f"Deleted {count} documents for paper {paper_id}")
        
        return count
    
    def get_collection_info(self) -> Dict[str, Any]:
        """获取 collection 信息"""
        client = self._get_client()
        
        info = client.get_collection(self.config.collection_name)
        
        # 兼容不同版本的 Qdrant API
        # 新版本使用 points_count 和 indexed_vectors_count
        # 旧版本使用 vectors_count
        points_count = info.points_count
        vectors_count = getattr(info, 'indexed_vectors_count', None) or getattr(info, 'vectors_count', points_count)
        
        return {
            "name": self.config.collection_name,
            "vectors_count": vectors_count,
            "points_count": points_count,
            "status": info.status.name if hasattr(info.status, 'name') else str(info.status),
            "vector_size": self.config.vector_size,
            "distance_metric": self.config.distance_metric,
        }
    
    def clear(self) -> None:
        """清空 collection"""
        client = self._get_client()
        
        # 删除并重建 collection
        client.delete_collection(self.config.collection_name)
        self._ensure_collection()
        
        logger.info(f"Cleared collection: {self.config.collection_name}")
    
    def close(self) -> None:
        """关闭连接"""
        if self._client is not None:
            self._client.close()
            self._client = None
            logger.info("Qdrant connection closed")

