"""
Agent State 转换模块

从 manifest.yaml 转换为前端可用的 agent_state 字典。
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime


# 项目根目录（相对于此文件的路径）
_PROJECT_ROOT = Path(__file__).parent.parent


def _abspath(path: str) -> str:
    """将路径转换为绝对路径。"""
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(_PROJECT_ROOT, path))


def _relpath(path: str, base: Optional[str] = None) -> str:
    """
    将绝对路径转换为相对路径（相对于项目根目录）。
    
    Args:
        path: 绝对路径或相对路径
        base: 基准路径，默认为项目根目录
    
    Returns:
        相对路径字符串
    """
    if base is None:
        base = str(_PROJECT_ROOT)
    
    abs_path = _abspath(path)
    abs_base = _abspath(base)
    
    try:
        rel_path = os.path.relpath(abs_path, abs_base)
        # 确保使用正斜杠（跨平台兼容）
        return rel_path.replace(os.sep, '/')
    except ValueError:
        # 如果路径不在基准目录下，返回原路径
        return path


def _flatten_parameters(params: Dict[str, Any], parent_key: str = '', sep: str = '.') -> Dict[str, Any]:
    """
    将嵌套的参数字典扁平化。
    
    Args:
        params: 嵌套参数字典
        parent_key: 父键名（用于递归）
        sep: 分隔符
    
    Returns:
        扁平化的参数字典
    
    Example:
        {
            "system_scale": {"num_households": 5}
        }
        ->
        {
            "system_scale.num_households": 5
        }
    """
    items = []
    for key, value in params.items():
        new_key = f"{parent_key}{sep}{key}" if parent_key else key
        if isinstance(value, dict):
            items.extend(_flatten_parameters(value, new_key, sep=sep).items())
        else:
            items.append((new_key, value))
    return dict(items)


def _format_duration(seconds: float) -> str:
    """
    格式化耗时（秒）为可读字符串。
    
    Args:
        seconds: 秒数
    
    Returns:
        格式化字符串，如 "1h 40m 40s"
    """
    if seconds is None:
        return "0s"
    
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    parts = []
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0:
        parts.append(f"{minutes}m")
    if secs > 0 or len(parts) == 0:
        parts.append(f"{secs}s")
    
    return " ".join(parts)


def _get_group_display_name(group_name: str) -> str:
    """
    根据组名生成显示名称。
    
    Args:
        group_name: 组名（如 "control", "treatment_A"）
    
    Returns:
        显示名称（如 "控制组", "实验组 A"）
    """
    display_names = {
        "control": "控制组",
        "treatment_a": "实验组 A",
        "treatment_b": "实验组 B",
    }
    
    # 尝试直接匹配
    if group_name.lower() in display_names:
        return display_names[group_name.lower()]
    
    # 尝试匹配 treatment_* 模式
    if group_name.lower().startswith("treatment_"):
        suffix = group_name.lower().replace("treatment_", "")
        # 尝试提取字母（如 "A", "B"）
        if suffix and suffix[0].isalpha():
            return f"实验组 {suffix[0].upper()}"
        return f"实验组 {suffix}"
    
    # 默认：首字母大写
    return group_name.replace("_", " ").title()


def _generate_experiment_id(manifest_path: str, experiment_name: str) -> str:
    """
    生成实验唯一标识符。
    
    Args:
        manifest_path: manifest.yaml 路径
        experiment_name: 实验名称
    
    Returns:
        实验 ID
    """
    # 使用实验目录名和时间戳生成 ID
    manifest_dir = os.path.dirname(manifest_path)
    dir_name = os.path.basename(manifest_dir)
    
    # 清理名称
    clean_name = experiment_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
    clean_name = "".join(c for c in clean_name if c.isalnum() or c == "_")
    
    return f"{clean_name}_{dir_name}"


def manifest_to_agent_state(manifest_path: str) -> Dict[str, Any]:
    """
    从 manifest.yaml 转换为 agent_state 字典。
    
    Args:
        manifest_path: manifest.yaml 文件路径（绝对路径或相对路径）
    
    Returns:
        agent_state 字典
    
    Raises:
        FileNotFoundError: 如果 manifest.yaml 不存在
        ValueError: 如果 manifest.yaml 格式不正确
    """
    # 加载 manifest
    manifest_path_abs = _abspath(manifest_path)
    if not os.path.exists(manifest_path_abs):
        raise FileNotFoundError(f"Manifest file not found: {manifest_path_abs}")
    
    with open(manifest_path_abs, "r", encoding="utf-8") as f:
        manifest = yaml.safe_load(f) or {}
    
    if not manifest:
        raise ValueError(f"Empty or invalid manifest file: {manifest_path_abs}")
    
    # 获取实验信息
    experiment_info = manifest.get("experiment_info", {})
    metadata = manifest.get("metadata", {})
    configurations = manifest.get("configurations", {})
    experiment_dir = experiment_info.get("directory", os.path.dirname(manifest_path_abs))
    
    # 转换为相对路径（相对于项目根目录）
    experiment_dir_rel = _relpath(experiment_dir)
    manifest_file_rel = _relpath(manifest_path_abs)
    
    # 提取基本信息
    experiment_name = experiment_info.get("name", "Unnamed Experiment")
    experiment_id = _generate_experiment_id(manifest_path_abs, experiment_name)
    
    # 提取状态信息
    runtime = metadata.get("runtime", {})
    status = metadata.get("status", "pending")
    start_time = runtime.get("start_time")
    end_time = runtime.get("end_time")
    duration_seconds = runtime.get("duration_seconds")
    
    # 提取研究设计
    research_question = metadata.get("research_question", "")
    hypothesis = metadata.get("hypothesis", "")
    expected_outcome = metadata.get("expected_outcome", "")
    
    # 构建 configurations 字典
    configurations_dict = {}
    config_files_dict = {}
    
    for group_name, group_config in configurations.items():
        # 提取参数（已扁平化）
        parameters_changed = group_config.get("parameters_changed", {})
        parameters = _flatten_parameters(parameters_changed) if parameters_changed else {}
        
        # 获取配置文件路径
        config_path = group_config.get("path", "")
        config_path_rel = _relpath(config_path) if config_path else ""
        
        configurations_dict[group_name] = {
            "name": group_name,
            "display_name": _get_group_display_name(group_name),
            "parameters": parameters,
        }
        
        config_files_dict[group_name] = config_path_rel
    
    # 构建 agent_state
    agent_state = {
        # 实验基本信息
        "experiment_id": experiment_id,
        "name": experiment_name,
        "description": experiment_info.get("description", ""),
        "created_date": experiment_info.get("created_date", ""),
        "author": experiment_info.get("author", ""),
        "tags": experiment_info.get("tags", []),
        
        # 实验状态
        "status": status,
        "start_time": start_time,
        "end_time": end_time,
        "duration_seconds": duration_seconds,
        
        # 研究设计
        "research_question": research_question,
        "hypothesis": hypothesis,
        "expected_outcome": expected_outcome,
        
        # 实验配置
        "configurations": configurations_dict,
        
        # 文件路径
        "paths": {
            "experiment_directory": experiment_dir_rel,
            "manifest_file": manifest_file_rel,
            "config_files": config_files_dict,
        },
    }
    
    return agent_state


def get_agent_state(experiment_directory: str) -> Dict[str, Any]:
    """
    从实验目录获取 agent_state。
    
    Args:
        experiment_directory: 实验目录路径（绝对路径或相对路径）
    
    Returns:
        agent_state 字典
    
    Raises:
        FileNotFoundError: 如果 manifest.yaml 不存在
    """
    experiment_dir_abs = _abspath(experiment_directory)
    manifest_path = os.path.join(experiment_dir_abs, "manifest.yaml")
    
    return manifest_to_agent_state(manifest_path)


def update_agent_state(manifest_path: str) -> Dict[str, Any]:
    """
    更新 agent_state（重新从 manifest.yaml 读取）。
    
    Args:
        manifest_path: manifest.yaml 文件路径
    
    Returns:
        更新后的 agent_state 字典
    """
    return manifest_to_agent_state(manifest_path)


# 示例使用
if __name__ == "__main__":
    # 测试转换
    test_manifest = "/root/project/agentsociety-ecosim/economist/experiments/ai_adoption_unemployment_smallscale/AI Adoption Intensity and Unemployment (Small Scale)/manifest.yaml"
    
    try:
        agent_state = manifest_to_agent_state(test_manifest)
        
        # 打印结果
        import json
        print(json.dumps(agent_state, indent=2, ensure_ascii=False))
        
    except Exception as e:
        print(f"Error: {e}")


