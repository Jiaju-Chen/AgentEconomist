"""
配置管理模块

提供统一的配置类，支持环境变量覆盖和配置验证。
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List
import logging

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingConfig:
    """Embedding 模型配置"""
    
    # SPECTER2 模型名称
    model_name: str = "allenai/specter2"
    
    # 适配器名称（用于不同任务）
    adapter_name: str = "allenai/specter2_proximity"
    
    # 设备配置 (cuda / cpu / auto)
    device: str = "auto"
    
    # 批处理大小
    batch_size: int = 32
    
    # 最大序列长度
    max_length: int = 512
    
    # 是否使用 FP16（半精度）
    use_fp16: bool = True
    
    # 是否使用回退模型（sentence-transformers）
    use_fallback: bool = False
    
    # HuggingFace 镜像或自建端点
    hf_endpoint: Optional[str] = None
    
    def __post_init__(self):
        if self.device == "auto":
            import torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Auto-detected device: {self.device}")


@dataclass
class QdrantConfig:
    """Qdrant 向量数据库配置"""
    
    # 存储模式: "memory" | "local" | "remote"
    mode: str = "local"
    
    # 本地存储路径（mode="local" 时使用）
    # 使用相对路径，基于当前文件位置（database 目录下）
    local_path: str = str(Path(__file__).parent.parent / "qdrant_data")
    
    # 远程服务器配置（mode="remote" 时使用）
    host: str = "localhost"
    port: int = 6333
    api_key: Optional[str] = None
    
    # Collection 配置
    collection_name: str = "academic_papers"
    
    # 向量维度（SPECTER2 输出 768 维）
    vector_size: int = 768
    
    # 距离度量: "cosine" | "euclid" | "dot"
    distance_metric: str = "cosine"
    
    def __post_init__(self):
        if self.mode == "local":
            Path(self.local_path).mkdir(parents=True, exist_ok=True)


@dataclass
class ChunkingConfig:
    """文档分块配置"""
    
    # 分块策略: "section" | "paragraph" | "fixed"
    strategy: str = "section"
    
    # 固定分块大小（字符数，strategy="fixed" 时使用）
    chunk_size: int = 1000
    
    # 分块重叠（字符数）
    chunk_overlap: int = 100
    
    # 是否包含元数据
    include_metadata: bool = True
    
    # 是否索引摘要
    index_abstract: bool = True
    
    # 是否索引章节
    index_sections: bool = True

    # 是否只索引引言章节（Introduction）
    only_intro_sections: bool = False
    
    # 跳过的章节标题（正则匹配）
    skip_sections: List[str] = field(default_factory=lambda: [
        r"(?i)references?",
        r"(?i)acknowledg(e)?ments?",
        r"(?i)supplementary",
        r"(?i)data\s+availability",
        r"(?i)author\s+contributions?",
        r"(?i)competing\s+interests?",
    ])


@dataclass 
class RetrievalConfig:
    """检索配置"""
    
    # 默认返回结果数
    default_top_k: int = 5
    
    # 最大返回结果数
    max_top_k: int = 50
    
    # 最小相似度阈值
    score_threshold: float = 0.0
    
    # 是否启用重排序
    enable_reranking: bool = False
    
    # 重排序模型（如果启用）
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@dataclass
class KnowledgeBaseConfig:
    """知识库总配置"""
    
    # 数据目录（使用相对路径）
    data_dir: str = str(Path(__file__).parent.parent.parent / "Crawl_Results")
    
    # 元数据索引文件（使用相对路径）
    metadata_file: str = str(Path(__file__).parent.parent.parent / "Crawl_Results" / "crawl_info_filter.json")
    
    # 日志级别
    log_level: str = "INFO"
    
    # HuggingFace 端点（用于国内镜像或自建）
    hf_endpoint: Optional[str] = None
    
    # 子配置
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    qdrant: QdrantConfig = field(default_factory=QdrantConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    
    def __post_init__(self):
        # 从环境变量覆盖配置
        self._load_from_env()
        # 配置日志
        self._setup_logging()
        # 配置 HF 端点（优先配置文件，再看环境变量）
        if self.hf_endpoint:
            os.environ["HF_ENDPOINT"] = self.hf_endpoint
        elif os.getenv("KB_HF_ENDPOINT"):
            os.environ["HF_ENDPOINT"] = os.getenv("KB_HF_ENDPOINT")  # type: ignore
    
    def _load_from_env(self):
        """从环境变量加载配置"""
        env_mappings = {
            "KB_DATA_DIR": "data_dir",
            "KB_METADATA_FILE": "metadata_file",
            "KB_LOG_LEVEL": "log_level",
            "KB_HF_ENDPOINT": "hf_endpoint",
            "KB_QDRANT_MODE": ("qdrant", "mode"),
            "KB_QDRANT_HOST": ("qdrant", "host"),
            "KB_QDRANT_PORT": ("qdrant", "port"),
            "KB_QDRANT_API_KEY": ("qdrant", "api_key"),
            "KB_QDRANT_PATH": ("qdrant", "local_path"),
            "KB_QDRANT_COLLECTION": ("qdrant", "collection_name"),
            "KB_EMBEDDING_DEVICE": ("embedding", "device"),
            "KB_EMBEDDING_MODEL": ("embedding", "model_name"),
            "KB_EMBEDDING_USE_FALLBACK": ("embedding", "use_fallback"),
        }
        
        for env_key, attr_path in env_mappings.items():
            value = os.getenv(env_key)
            if value is not None:
                if isinstance(attr_path, tuple):
                    sub_config = getattr(self, attr_path[0])
                    current_value = getattr(sub_config, attr_path[1])
                    # 类型转换
                    if isinstance(current_value, bool):
                        value = str(value).lower() in ("1", "true", "yes", "on")
                    elif isinstance(current_value, int):
                        value = int(value)
                    elif isinstance(current_value, float):
                        value = float(value)
                    setattr(sub_config, attr_path[1], value)
                else:
                    setattr(self, attr_path, value)
    
    def _setup_logging(self):
        """配置日志"""
        logging.basicConfig(
            level=getattr(logging, self.log_level.upper()),
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> "KnowledgeBaseConfig":
        """从 YAML 文件加载配置"""
        import yaml
        
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        
        # 递归构建配置
        embedding = EmbeddingConfig(**data.pop("embedding", {}))
        qdrant = QdrantConfig(**data.pop("qdrant", {}))
        chunking = ChunkingConfig(**data.pop("chunking", {}))
        retrieval = RetrievalConfig(**data.pop("retrieval", {}))
        
        return cls(
            embedding=embedding,
            qdrant=qdrant,
            chunking=chunking,
            retrieval=retrieval,
            **data,
        )
    
    def to_yaml(self, yaml_path: str) -> None:
        """保存配置到 YAML 文件"""
        import yaml
        from dataclasses import asdict
        
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(asdict(self), f, default_flow_style=False, allow_unicode=True)

