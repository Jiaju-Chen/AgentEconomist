"""
状态类型定义

定义与前端对接的 FSState 类型（纯 TypedDict，不依赖任何框架）。
"""

from typing import List, Dict, Optional
try:
    from typing_extensions import TypedDict
except ImportError:
    from typing import TypedDict  # Python 3.12+


class KnowledgeBaseItem(TypedDict):
    """文献支撑项"""
    title: str                    # 文献标题
    source: str                   # 出处（期刊名 + 年份，如 "Nature 2023"）
    url: Optional[str]            # 论文链接（PDF/DOI链接）


class ConfigurationItem(TypedDict):
    """实验配置项"""
    filename: str                 # 配置文件名（如 "treatmentA.yaml"）
    url: str                      # 配置文件URL路径


class ImageItem(TypedDict):
    """实验结果图片项"""
    name: str                     # 图片名称/描述（格式：{group_name}_{filename}）
    url: str                      # 图片URL路径


class PathsDict(TypedDict):
    """文件路径字典（所有路径都是相对路径）"""
    experiment_directory: str     # 实验目录路径
    manifest_file: str            # manifest.yaml 路径
    config_files: Dict[str, str]  # 组名: 配置文件路径


class FSState(TypedDict):
    """
    Field Simulation Agent 状态结构
    
    这是与前端共享的业务状态，不依赖任何框架。
    """
    # ========== 实验基本信息 ==========
    experiment_id: str            # 实验唯一标识符
    name: str                     # 实验名称
    description: str              # 实验描述
    created_date: str             # 创建日期 (ISO 8601: "2026-01-01")
    tags: List[str]               # 标签列表
    
    # ========== 文献支撑 ==========
    knowledge_base: List[KnowledgeBaseItem]
    
    # ========== 实验状态 ==========
    status: str                   # "pending" | "planning" | "running" | "completed" | "failed" | "analysis_pending"
    start_time: Optional[str]     # 开始时间 (ISO 8601)
    end_time: Optional[str]       # 结束时间 (ISO 8601)
    duration_seconds: Optional[float]  # 总耗时（秒）
    
    # ========== 研究设计 ==========
    research_question: str        # 研究问题
    hypothesis: str              # 研究假设
    expected_outcome: str        # 预期结果
    
    # ========== 实验配置 ==========
    configurations: List[ConfigurationItem]
    
    # ========== 实验结果 ==========
    images: List[ImageItem]
    
    # ========== 文件路径 ==========
    paths: PathsDict


def create_empty_fs_state() -> FSState:
    """创建空的 FSState"""
    return FSState(
        experiment_id="",
        name="",
        description="",
        created_date="",
        tags=[],
        knowledge_base=[],
        status="pending",
        start_time=None,
        end_time=None,
        duration_seconds=None,
        research_question="",
        hypothesis="",
        expected_outcome="",
        configurations=[],
        images=[],
        paths=PathsDict(
            experiment_directory="",
            manifest_file="",
            config_files={}
        )
    )
