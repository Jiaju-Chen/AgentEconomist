"""
路径工具模块

提供路径转换和项目根目录获取功能。
"""

from pathlib import Path
from typing import Union


def get_project_root() -> Path:
    """
    获取项目根目录
    
    Returns:
        项目根目录的 Path 对象
    """
    # AgentEconomist 目录的父目录就是项目根目录
    current_file = Path(__file__)
    # AgentEconomist/utils/path.py -> AgentEconomist -> project root
    return current_file.parent.parent.parent


def to_absolute(path: Union[str, Path]) -> Path:
    """
    将路径转换为绝对路径
    
    Args:
        path: 相对路径（相对于项目根目录）或绝对路径
    
    Returns:
        绝对路径的 Path 对象
    """
    path_obj = Path(path)
    if path_obj.is_absolute():
        return path_obj
    
    # 相对路径：相对于项目根目录
    project_root = get_project_root()
    return project_root / path_obj


def to_relative(path: Union[str, Path]) -> str:
    """
    将路径转换为相对路径（相对于项目根目录）
    
    Args:
        path: 绝对路径或相对路径
    
    Returns:
        相对路径字符串
    """
    path_obj = Path(path)
    if not path_obj.is_absolute():
        return str(path_obj)
    
    project_root = get_project_root()
    try:
        return str(path_obj.relative_to(project_root))
    except ValueError:
        # 如果路径不在项目根目录下，返回绝对路径
        return str(path_obj)
