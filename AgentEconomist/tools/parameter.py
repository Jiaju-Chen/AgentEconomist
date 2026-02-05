"""
参数查询工具

提供仿真参数的查询功能。
"""

import sys
import yaml
from pathlib import Path
from typing import Optional, Dict, Any

from ..utils.path import get_project_root
from ..utils.format import format_tool_output
from ..utils.response import handle_tool_error, extract_response_text


def _load_default_yaml() -> Dict[str, Any]:
    """加载 default.yaml 配置文件"""
    project_root = get_project_root()
    default_yaml_path = project_root / "agentsociety_ecosim" / "default.yaml"
    
    if not default_yaml_path.exists():
        raise FileNotFoundError(f"default.yaml not found at {default_yaml_path}")
    
    with open(default_yaml_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f) or {}


def _organize_parameters_by_category(yaml_config: Dict[str, Any]) -> Dict[str, list]:
    """将参数按类别组织"""
    categories = {
        "tax_policy": [],
        "labor_market": [],
        "production": [],
        "market": [],
        "system_scale": [],
        "redistribution": [],
        "performance": [],
        "monitoring": [],
        "innovation": [],
    }
    
    # 从 YAML 配置中提取参数
    for category, params in yaml_config.items():
        if isinstance(params, dict):
            for param_name in params.keys():
                if category in categories:
                    # 使用点号格式：category.param_name
                    categories[category].append(f"{category}.{param_name}")
    
    return categories


def get_available_parameters(category: str = "all") -> str:
    """
    获取所有可配置的经济仿真参数列表
    
    Args:
        category: 参数类别过滤 (all/tax_policy/production/labor_market/market/system_scale/redistribution/performance/monitoring/innovation)
    
    Returns:
        XML格式字符串，包含参数分类和名称
    """
    try:
        # 加载 default.yaml
        yaml_config = _load_default_yaml()
        
        # 按类别组织参数
        organized_params = _organize_parameters_by_category(yaml_config)
        
        # 构建输出
        result_parts = []
        result_parts.append("Available Simulation Parameters:")
        result_parts.append("=" * 60)
        
        if category == "all":
            # 显示所有类别
            for cat, params in organized_params.items():
                if params:
                    result_parts.append(f"\n{cat.upper().replace('_', ' ')} ({len(params)} parameters):")
                    for param in sorted(params):
                        result_parts.append(f"  - {param}")
        else:
            # 只显示指定类别
            if category in organized_params:
                params = organized_params[category]
                if params:
                    result_parts.append(f"\n{category.upper().replace('_', ' ')} ({len(params)} parameters):")
                    for param in sorted(params):
                        result_parts.append(f"  - {param}")
                else:
                    result_parts.append(f"\nNo parameters found in category: {category}")
            else:
                result_parts.append(f"\nUnknown category: {category}")
                result_parts.append("Available categories: tax_policy, production, labor_market, market, system_scale, redistribution, performance, monitoring, innovation")
        
        result_text = "\n".join(result_parts)
        
        return format_tool_output(
            "success",
            "Parameters retrieved successfully",
            parameters=result_text,
            total_categories=len([c for c in organized_params.values() if c])
        )
        
    except Exception as e:
        return handle_tool_error("get_available_parameters", e)


def _parse_yaml_with_comments(file_path: Path) -> Dict[str, Dict[str, Any]]:
    """解析 YAML 文件，提取参数值和注释信息（注释作为字符串直接返回）"""
    param_info = {}
    
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    current_category = None
    i = 0
    
    while i < len(lines):
        line = lines[i].strip()
        
        # 检测类别（如 tax_policy:）
        if line and not line.startswith('#') and line.endswith(':'):
            current_category = line.rstrip(':')
            i += 1
            continue
        
        # 检测参数行（包含冒号，但不是类别）
        if line and ':' in line and not line.startswith('#') and current_category:
            # 提取参数名和值
            if '#' in line:
                param_part, comment_part = line.split('#', 1)
            else:
                param_part = line
                comment_part = ""
            
            param_part = param_part.strip()
            if ':' in param_part:
                param_name, param_value = param_part.split(':', 1)
                param_name = param_name.strip()
                param_value = param_value.strip()
                
                # 解析值
                try:
                    # 尝试转换为数字
                    if '.' in param_value:
                        param_value = float(param_value)
                    else:
                        param_value = int(param_value)
                except ValueError:
                    # 布尔值或字符串
                    if param_value.lower() in ('true', 'yes', 'on'):
                        param_value = True
                    elif param_value.lower() in ('false', 'no', 'off'):
                        param_value = False
                    else:
                        # 保持字符串
                        param_value = param_value.strip('"\'')
                
                # 直接保存注释字符串，不进行解析
                comment = comment_part.strip() if comment_part else ""
                
                # 存储参数信息
                full_param_name = f"{current_category}.{param_name}"
                param_info[full_param_name] = {
                    "name": param_name,
                    "full_name": full_param_name,
                    "category": current_category,
                    "value": param_value,
                    "comment": comment,  # 直接保存注释字符串
                    "type": type(param_value).__name__
                }
        
        i += 1
    
    return param_info


def get_parameter_info(parameter_name: str) -> str:
    """
    获取特定参数的详细信息
    
    Args:
        parameter_name: 参数名称（点号格式，如 "tax_policy.income_tax_rate" 或扁平格式如 "income_tax_rate"）
    
    Returns:
        XML格式字符串，包含参数的详细信息
    """
    try:
        project_root = get_project_root()
        default_yaml_path = project_root / "agentsociety_ecosim" / "default.yaml"
        
        if not default_yaml_path.exists():
            raise FileNotFoundError(f"default.yaml not found at {default_yaml_path}")
        
        # 解析 YAML 文件（带注释）
        param_info_dict = _parse_yaml_with_comments(default_yaml_path)
        
        # 查找参数（支持点号格式和扁平格式）
        param_info = None
        
        # 首先尝试直接匹配
        if parameter_name in param_info_dict:
            param_info = param_info_dict[parameter_name]
        else:
            # 尝试扁平格式匹配（如果用户只提供了参数名）
            for full_name, info in param_info_dict.items():
                if info["name"] == parameter_name or full_name.endswith(f".{parameter_name}"):
                    param_info = info
                    break
        
        if not param_info:
            # 如果找不到，尝试从 YAML 中直接读取
            yaml_config = _load_default_yaml()
            
            # 解析点号格式的参数名
            if '.' in parameter_name:
                category, param_name = parameter_name.split('.', 1)
            else:
                # 尝试在所有类别中查找
                category = None
                param_name = parameter_name
                for cat, params in yaml_config.items():
                    if isinstance(params, dict) and param_name in params:
                        category = cat
                        break
            
            if category and category in yaml_config:
                params = yaml_config[category]
                if isinstance(params, dict) and param_name in params:
                    param_info = {
                        "name": param_name,
                        "full_name": f"{category}.{param_name}",
                        "category": category,
                        "value": params[param_name],
                        "comment": "",  # 如果从 YAML 直接读取，没有注释信息
                        "type": type(params[param_name]).__name__
                    }
        
        if not param_info:
            return format_tool_output(
                "error",
                f"Parameter '{parameter_name}' not found",
                parameter=parameter_name,
                suggestion="Use get_available_parameters() to see all available parameters"
            )
        
        # 构建详细信息输出
        info_parts = []
        info_parts.append(f"Parameter: {param_info['full_name']}")
        info_parts.append("=" * 60)
        info_parts.append(f"Name: {param_info['name']}")
        info_parts.append(f"Category: {param_info['category']}")
        info_parts.append(f"Type: {param_info['type']}")
        info_parts.append(f"Default Value: {param_info['value']}")
        
        # 直接显示注释字符串
        if param_info['comment']:
            info_parts.append(f"Comment: {param_info['comment']}")
        
        info_text = "\n".join(info_parts)
        
        return format_tool_output(
            "success",
            f"Parameter info for {parameter_name}",
            parameter=param_info['full_name'],
            details=info_text,
            name=param_info['name'],
            category=param_info['category'],
            type=param_info['type'],
            default_value=param_info['value'],
            comment=param_info['comment']
        )
        
    except Exception as e:
        return handle_tool_error("get_parameter_info", e)
