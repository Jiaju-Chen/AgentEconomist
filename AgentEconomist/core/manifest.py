"""
Manifest 管理模块

提供 manifest.yaml 文件的 CRUD 操作。
"""

import os
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from ..utils.path import to_absolute, get_project_root


# 北京时区 (UTC+8)
BEIJING_TZ = timezone(timedelta(hours=8))


def beijing_now() -> datetime:
    """获取北京时间"""
    return datetime.now(BEIJING_TZ)


def load_manifest(manifest_path: str) -> Dict[str, Any]:
    """
    加载 manifest.yaml 文件
    
    Args:
        manifest_path: manifest 文件路径
    
    Returns:
        manifest 数据字典，如果文件不存在返回空字典
    """
    abs_path = to_absolute(manifest_path)
    if not abs_path.exists():
        return {}
    with open(abs_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data


def save_manifest(manifest_path: str, data: Dict[str, Any]) -> None:
    """
    保存 manifest.yaml 文件
    
    Args:
        manifest_path: manifest 文件路径
        data: 要保存的数据
    """
    abs_path = to_absolute(manifest_path)
    abs_path.parent.mkdir(parents=True, exist_ok=True)
    with open(abs_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(
            data,
            f,
            allow_unicode=True,
            sort_keys=False,
            default_flow_style=False,
        )


def ensure_manifest_structure(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    确保 manifest 数据结构完整
    
    Args:
        data: manifest 数据
    
    Returns:
        补充完整的 manifest 数据
    """
    data.setdefault("experiment_info", {})
    data.setdefault("metadata", {})
    
    metadata = data["metadata"]
    metadata.setdefault("status", "planned")
    metadata.setdefault("runtime", {
        "start_time": None,
        "end_time": None,
        "duration_seconds": None,
    })
    
    data.setdefault("experiment_intervention", {
        "intervention_type": "none",
        "intervention_parameters": {},
    })
    data.setdefault("configurations", {})
    data.setdefault("runs", {})
    data.setdefault("results_summary", {
        "comparison_status": "pending",
        "conclusion": "",
        "insights": [],
    })
    
    # 确保 knowledge_base 字段存在
    if "knowledge_base" not in metadata:
        metadata["knowledge_base"] = []
    
    return data


def add_knowledge_base_items(
    manifest_path: str,
    items: List[Dict[str, Any]],
    merge: bool = True
) -> None:
    """
    向 manifest 添加 knowledge_base 项
    
    Args:
        manifest_path: manifest 文件路径
        items: 要添加的文献项列表，每项包含 title, source, url
        merge: 如果为 True，则合并到现有列表（去重），否则替换
    """
    manifest = ensure_manifest_structure(load_manifest(manifest_path))
    metadata = manifest["metadata"]
    
    if not merge:
        metadata["knowledge_base"] = []
    
    existing_items = metadata.get("knowledge_base", [])
    existing_urls = {item.get("url") for item in existing_items if item.get("url")}
    
    # 添加新项（去重：基于 url）
    for item in items:
        url = item.get("url")
        if url and url not in existing_urls:
            # 只保留 title, source, url 三个字段
            new_item = {
                "title": item.get("title", ""),
                "source": item.get("source", ""),
                "url": url
            }
            existing_items.append(new_item)
            existing_urls.add(url)
        elif not url:
            # 没有 url 的项也添加（可能是不同的文献）
            existing_items.append({
                "title": item.get("title", ""),
                "source": item.get("source", ""),
                "url": None
            })
    
    metadata["knowledge_base"] = existing_items
    save_manifest(manifest_path, manifest)


def get_knowledge_base_items(manifest_path: str) -> List[Dict[str, Any]]:
    """
    从 manifest 获取 knowledge_base 项
    
    Args:
        manifest_path: manifest 文件路径
    
    Returns:
        文献项列表
    """
    manifest = load_manifest(manifest_path)
    metadata = manifest.get("metadata", {})
    return metadata.get("knowledge_base", [])
