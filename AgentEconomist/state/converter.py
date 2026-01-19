"""
状态转换模块

提供 manifest.yaml ↔ FSState 的双向转换。
"""

import os
from pathlib import Path
from typing import Any, Dict, List

from .types import (
    FSState,
    PathsDict,
    KnowledgeBaseItem,
    ConfigurationItem,
    ImageItem,
    create_empty_fs_state,
)
from ..core.manifest import load_manifest, get_knowledge_base_items
from ..utils.path import to_absolute, to_relative, get_project_root


def _flatten_parameters(params: Dict[str, Any], parent_key: str = '', sep: str = '.') -> Dict[str, Any]:
    """
    将嵌套的参数字典扁平化
    
    Example:
        {"system_scale": {"num_households": 5}}
        -> {"system_scale.num_households": 5}
    """
    items = []
    for key, value in params.items():
        new_key = f"{parent_key}{sep}{key}" if parent_key else key
        if isinstance(value, dict):
            items.extend(_flatten_parameters(value, new_key, sep=sep).items())
        else:
            items.append((new_key, value))
    return dict(items)


def _generate_experiment_id(manifest_path: str, experiment_name: str) -> str:
    """生成实验唯一标识符"""
    manifest_dir = os.path.dirname(manifest_path)
    dir_name = os.path.basename(manifest_dir)
    
    # 清理名称
    clean_name = experiment_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
    clean_name = "".join(c for c in clean_name if c.isalnum() or c == "_")
    
    return f"{clean_name}_{dir_name}"


def _scan_experiment_images(manifest: Dict[str, Any]) -> List[ImageItem]:
    """
    扫描实验目录下的图片文件
    
    从 manifest 的 configurations 中获取每个 group 的 output 目录，
    扫描图片并生成带 group_name 前缀的名称。
    
    查找路径：
    - <experiment_output_dir>/charts/*.png
    - <experiment_output_dir>/charts/*.jpg
    
    Args:
        manifest: manifest 数据字典
    
    Returns:
        图片项列表，name 格式为 {group_name}_{filename}
    """
    images: List[ImageItem] = []
    configurations = manifest.get("configurations", {})
    
    for group_name, group_config in configurations.items():
        experiment_output_dir = group_config.get("experiment_output_dir", "")
        experiment_name = group_config.get("experiment_name", "")
        
        if not experiment_output_dir:
            continue
        
        output_path = to_absolute(experiment_output_dir)
        charts_dir = output_path / "charts"
        
        if not charts_dir.exists() or not charts_dir.is_dir():
            continue
        
        # 扫描图片文件
        for pattern in ["*.png", "*.jpg"]:
            for img_file in sorted(charts_dir.glob(pattern)):
                try:
                    # 图片名称：{group_name}_{filename}
                    img_name = f"{group_name}_{img_file.stem}"
                    
                    # 图片URL：/{experiment_name}/{filename}.{ext}
                    img_url = f"/{experiment_name}/{img_file.name}"
                    
                    images.append(ImageItem(
                        name=img_name,
                        url=img_url
                    ))
                except Exception as e:
                    print(f"Warning: Failed to process image {img_file}: {e}")
                    continue
    
    return images


def manifest_to_fs_state(manifest_path: str) -> FSState:
    """
    从 manifest.yaml 转换为 FSState
    
    Args:
        manifest_path: manifest.yaml 文件路径
    
    Returns:
        FSState 对象
    
    Raises:
        FileNotFoundError: 如果 manifest.yaml 不存在
        ValueError: 如果 manifest.yaml 格式不正确
    """
    abs_path = to_absolute(manifest_path)
    if not abs_path.exists():
        raise FileNotFoundError(f"Manifest file not found: {abs_path}")
    
    manifest = load_manifest(manifest_path)
    if not manifest:
        raise ValueError(f"Empty or invalid manifest file: {abs_path}")
    
    # 提取信息
    experiment_info = manifest.get("experiment_info", {})
    metadata = manifest.get("metadata", {})
    configurations = manifest.get("configurations", {})
    experiment_dir = experiment_info.get("directory", str(abs_path.parent))
    
    # 基本信息
    experiment_name = experiment_info.get("name", "Unnamed Experiment")
    experiment_id = _generate_experiment_id(str(abs_path), experiment_name)
    
    # 状态信息
    runtime = metadata.get("runtime", {})
    status = metadata.get("status", "pending")
    
    # 转换 configurations: dict -> List[ConfigurationItem]
    config_items: List[ConfigurationItem] = []
    config_files: Dict[str, str] = {}
    
    for group_name, group_config in configurations.items():
        config_path = group_config.get("path", "")
        if config_path:
            config_path_rel = to_relative(config_path)
            config_files[group_name] = config_path_rel
            
            # 生成配置文件URL（相对路径，前端会处理）
            # 格式：/experiment_files/{experiment_dir_name}/{group_name}.yaml
            experiment_dir_name = os.path.basename(experiment_dir)
            config_url = f"/experiment_files/{experiment_dir_name}/{group_name}.yaml"
            
            config_items.append(ConfigurationItem(
                filename=f"{group_name}.yaml",
                url=config_url
            ))
    
    # 扫描图片（从 manifest 中获取配置信息）
    images = _scan_experiment_images(manifest)
    
    # 提取 knowledge_base 从 manifest
    knowledge_base_raw = get_knowledge_base_items(manifest_path)
    knowledge_base: List[KnowledgeBaseItem] = []
    for item in knowledge_base_raw:
        knowledge_base.append(KnowledgeBaseItem(
            title=item.get("title", ""),
            source=item.get("source", ""),
            url=item.get("url")
        ))
    
    # 构建 FSState
    return FSState(
        experiment_id=experiment_id,
        name=experiment_name,
        description=experiment_info.get("description", ""),
        created_date=experiment_info.get("created_date", ""),
        tags=experiment_info.get("tags", []),
        knowledge_base=knowledge_base,
        status=status,
        start_time=runtime.get("start_time"),
        end_time=runtime.get("end_time"),
        duration_seconds=runtime.get("duration_seconds"),
        research_question=metadata.get("research_question", ""),
        hypothesis=metadata.get("hypothesis", ""),
        expected_outcome=metadata.get("expected_outcome", ""),
        configurations=config_items,
        images=images,
        paths=PathsDict(
            experiment_directory=to_relative(experiment_dir),
            manifest_file=to_relative(str(abs_path)),
            config_files=config_files
        )
    )


def update_fs_state_from_manifest(manifest_path: str) -> FSState:
    """
    从 manifest.yaml 更新 FSState（manifest_to_fs_state 的别名）
    
    Args:
        manifest_path: manifest.yaml 文件路径
    
    Returns:
        更新后的 FSState
    """
    return manifest_to_fs_state(manifest_path)


def fs_state_to_dict(fs_state: FSState) -> Dict[str, Any]:
    """
    将 FSState 转换为普通字典（用于序列化）
    
    Args:
        fs_state: FSState 对象
    
    Returns:
        字典表示
    """
    return dict(fs_state)
